import json
from pathlib import Path

import torch
import torch.nn as nn

from nsnet2.layers import make_linear, make_gru

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_stats_path(path):
    """Resolve an ``input_norm.path``: as given (absolute or cwd-relative),
    else relative to the repo root, so a config copied into ``cp_<run>/``
    still finds ``configs/stats/...`` from any working directory."""
    p = Path(path)
    if p.is_file():
        return p
    q = _REPO_ROOT / p
    if not p.is_absolute() and q.is_file():
        return q
    raise FileNotFoundError(
        f"input_norm stats file {path!r} not found (tried cwd and {_REPO_ROOT})")


# Config keys the network input depends on; a stats file measured with other
# values describes other features and is refused.
_FEATURE_KEYS = ("sampling_rate", "n_fft", "hop_size", "win_size", "compress_factor")


def load_input_mean(cfg, n_freq, h=None):
    """Read the frozen per-bin input mean described by an ``input_norm`` block."""
    kind = cfg.get("kind", "mean")
    if kind != "mean":
        raise ValueError(f"input_norm kind {kind!r} unsupported (only 'mean')")
    with open(_resolve_stats_path(cfg["path"])) as f:
        stats = json.load(f)
    mean = stats["mean"]
    if h is not None:
        feat = stats.get("config", {})
        bad = {k: (feat.get(k), getattr(h, k, None)) for k in _FEATURE_KEYS
               if k in feat and getattr(h, k, None) is not None and feat[k] != getattr(h, k, None)}
        if bad:
            raise ValueError(f"input_norm stats {cfg['path']!r} were measured on other features: {bad}")
    if len(mean) != n_freq:
        raise ValueError(f"input_norm mean has {len(mean)} bins, model expects {n_freq}")
    return torch.tensor(mean, dtype=torch.float32)


class NSNet2(nn.Module):
    """NSNet2 magnitude-mask predictor with pluggable layer backends.

    Reference: Braun & Tashev, "Towards efficient models for real-time deep
    noise suppression" (ICASSP 2021). Standard pipeline (FC + 2-layer GRU + FC
    stack producing a per-T/F gain in [0, 1]) — but every linear and the GRU
    are built via the ``models.layers`` factories so the implementation can be
    toggled from the config without touching this file.

    Config blocks (all optional, default to dense / cuDNN):

        "linear": {"kind": "linear" | "butterfly" | "monarch", ...kwargs}
        "gru":    {"kind": "gru"    | "butterfly" | "monarch", ...kwargs}

    See ``models/layers.py`` for the per-backend kwargs.

    Block-design options (both default OFF; when absent the module tree and
    state_dict are exactly the pre-existing ones):

        "input_norm": {"kind": "mean", "path": "configs/stats/<file>.json"}
            Subtract a frozen per-frequency-bin mean (buffer ``in_mean``,
            n_freq values, computed on the training split by
            ``nsnet2.input_stats``) from the network input. The predicted
            mask still multiplies the ORIGINAL noisy magnitude.
        "block_norm": "batch"
            ``BatchNorm1d`` (``bn_in``/``bn1``/``bn2``) directly after
            ``fc_in``/``fc1``/``fc2`` and before their ReLU, normalising the
            feature axis of the (B, T, C) activations.

    ``nsnet2.fold.fold_for_export`` folds both back into the linears, giving
    a plain NSNet2 for the existing export paths.
    """

    def __init__(self, h):
        super().__init__()
        self.h = h
        n_freq = h.n_fft // 2 + 1
        hidden = getattr(h, "hidden_dim", 400)
        fc_hidden = getattr(h, "fc_hidden_dim", 600)
        num_gru_layers = getattr(h, "num_gru_layers", 2)

        linear_cfg = getattr(h, "linear", None) or {"kind": "linear"}
        gru_cfg = getattr(h, "gru", None) or {"kind": "gru"}
        self.linear_kind = linear_cfg.get("kind", "linear")
        self.gru_kind = gru_cfg.get("kind", "gru")

        self.fc_in = make_linear(n_freq, hidden, cfg=linear_cfg)
        self.gru = make_gru(hidden, hidden, num_gru_layers, cfg=gru_cfg)
        self.fc1 = make_linear(hidden, fc_hidden, cfg=linear_cfg)
        self.fc2 = make_linear(fc_hidden, fc_hidden, cfg=linear_cfg)
        self.fc_out = make_linear(fc_hidden, n_freq, cfg=linear_cfg)
        self.act = nn.ReLU()

        # Everything below is registered only when its flag is on, so a
        # flags-off model has the exact pre-existing modules / state_dict.
        input_norm = getattr(h, "input_norm", None)
        self.input_norm = bool(input_norm)
        if self.input_norm:
            self.register_buffer("in_mean", load_input_mean(input_norm, n_freq, h))

        block_norm = getattr(h, "block_norm", None)
        if block_norm not in (None, False, "none", "batch"):
            raise ValueError(f"block_norm {block_norm!r} unsupported (only 'batch')")
        self.block_norm = block_norm == "batch"
        if self.block_norm:
            self.bn_in = nn.BatchNorm1d(hidden)
            self.bn1 = nn.BatchNorm1d(fc_hidden)
            self.bn2 = nn.BatchNorm1d(fc_hidden)

    @staticmethod
    def _bn(bn, x):
        # x: (B, T, C); BatchNorm1d normalises dim 1 of (B, C, T).
        return bn(x.transpose(1, 2)).transpose(1, 2)

    def _hidden(self, fc, bn, x):
        y = fc(x)
        if self.block_norm:
            y = self._bn(bn, y)
        return self.act(y)

    def forward(self, noisy_mag, noisy_pha):
        x = noisy_mag.transpose(1, 2)
        if self.input_norm:
            x = x - self.in_mean
        h = self._hidden(self.fc_in, getattr(self, "bn_in", None), x)
        h, _ = self.gru(h)
        h = self._hidden(self.fc1, getattr(self, "bn1", None), h)
        h = self._hidden(self.fc2, getattr(self, "bn2", None), h)
        mask = torch.sigmoid(self.fc_out(h))
        mask = mask.transpose(1, 2)

        denoised_mag = noisy_mag * mask
        denoised_pha = noisy_pha
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha),
             denoised_mag * torch.sin(denoised_pha)),
            dim=-1,
        )
        return denoised_mag, denoised_pha, denoised_com
