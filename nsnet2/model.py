import torch
import torch.nn as nn

from nsnet2.layers import make_linear, make_gru


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

    def forward(self, noisy_mag, noisy_pha):
        x = noisy_mag.transpose(1, 2)
        h = self.act(self.fc_in(x))
        h, _ = self.gru(h)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
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
