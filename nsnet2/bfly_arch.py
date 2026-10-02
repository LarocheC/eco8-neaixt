"""Butterfly-native NSNet2 variants, built for the FlexMan butterfly engine.

Every arm uses only operations the engine runs (BUTTERFLY_ENGINE.md, branch
butterfly-engine of jabra-private): radix-2 butterfly tasks with ANY stride order
and several output stacks (§8), a per-output epilogue -- bias, an added "seed"
vector, then RAW / ReLU / 4096-entry LUT (§9) -- element-wise tasks, and the
Hadamard unit's element-wise products. BatchNorm is used only where it folds into
the neighbouring butterfly at export (a fixed per-feature affine); there is no
LayerNorm (data-dependent) and no permutation inside the network.

Shared by every arm: 256 frequency bins (Nyquist dropped, its gain copied from bin
255) so the input butterfly is an exact 256-point one, and the A1 block design
(frozen per-bin input mean, generator LR warmup) that lifted butterfly NSNet2 by
~+0.06 PESQ on branch butterfly-a1.

``h.arch`` selects the arm:

    {"type": "bflynet",                       # R-family: NSNet2 topology
     "hidden": 512, "nblocks": 1,             # width (F: 2048), BB^T depth (R2b: 2)
     "encoder": "butterfly" | "cepstral",     # B: [x ; DCT x] in, s + iDCT(q) out
     "fc": {"kind": "plain" | "residual", "n": 2, "norm": "batch"},   # A
     "rnn": "gru" | "mingru" | "lru"}         # C

    {"type": "bflygrid",                      # D / E: frequency axis kept inside
     "channels": 4, "skips": false,           # E: skips = true
     "residual": false}                       # Dres: x + ReLU(BN(B x)) per square level

The cepstral path's DCT is a fixed orthonormal DCT-II: a real butterfly cannot hold
it (a 256-point fit leaves >= 48 % relative error even with input reorderings), but
the engine computes it exactly with the FFT pair words it already has for the STFT.
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint

from nsnet2.layers import make_linear
from nsnet2.model import NSNet2, _resolve_stats_path

try:
    from torch_structured import Butterfly as _TSButterfly
except ImportError:          # pragma: no cover
    _TSButterfly = None

N_BINS = 256

# torch_structured's Triton butterfly launches grid (row_tiles, rows * nstacks), and
# CUDA caps a grid's y dimension at 65535: a layer with more than two stacks at the
# training batch (256 x 126 frames = 32,256 rows) fails with "invalid argument".
# bfly() splits the rows so rows * nstacks stays under the cap.
_GRID_Y = 65535


def bfly(layer, x):
    st = getattr(layer, "nstacks", 1)
    cap = _GRID_Y // st
    rows = x.reshape(-1, x.shape[-1])
    if rows.shape[0] <= cap:
        return layer(x)
    out = torch.cat([layer(c) for c in rows.split(cap)], dim=0)
    return out.reshape(*x.shape[:-1], out.shape[-1])


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class StageButterfly(nn.Module):
    """Real butterfly with an explicit stride list (any order, repeats allowed).

    ``n`` is a power of two; the input (``in_size <= n``) is zero-padded and fed to
    each of ``nstacks`` stacks; the ``nstacks * n`` outputs (stack-major) are cropped
    to ``out_size``. Stage ``k`` with stride ``s`` pairs ``i0 = g*2s + j`` with
    ``i1 = i0 + s`` (pair ``p = g*s + j``) and writes row ``c`` to ``i0 + c*s`` --
    the engine's stage contract (BUTTERFLY_ENGINE.md §2), so a stride subset is one
    engine task with fewer stages.
    """

    def __init__(self, n, strides, nstacks=1, in_size=None, out_size=None,
                 bias=True, init="randn", checkpoint=True):
        super().__init__()
        assert n & (n - 1) == 0, "n must be a power of two"
        self.checkpoint = checkpoint     # recompute stage activations in backward (big inputs only)
        assert all(s & (s - 1) == 0 and 1 <= s < n for s in strides), strides
        self.n, self.strides, self.nstacks = n, list(strides), nstacks
        self.in_size = n if in_size is None else in_size
        self.out_size = nstacks * n if out_size is None else out_size
        t = torch.empty(len(strides), nstacks, n // 2, 2, 2)
        if init == "randn":          # torch_structured's variance-preserving init
            t.normal_(0.0, 1.0 / math.sqrt(2.0))
        elif init == "ortho":        # random rotation / reflection per pair
            th = torch.rand(t.shape[:-2]) * 2 * math.pi
            c, s = torch.cos(th), torch.sin(th)
            det = torch.randint(0, 2, th.shape).float() * 2 - 1
            t = torch.stack((torch.stack((det * c, -det * s), -1), torch.stack((s, c), -1)), -2)
        elif init == "identity":
            t.zero_(); t[..., 0, 0] = 1.0; t[..., 1, 1] = 1.0
        else:
            raise ValueError(init)
        self.twiddle = nn.Parameter(t)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_size).uniform_(
                -1 / math.sqrt(self.in_size), 1 / math.sqrt(self.in_size)))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        if self.checkpoint and torch.is_grad_enabled() and x.numel() > (1 << 22):
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def _forward(self, x):
        lead = x.shape[:-1]
        if x.shape[-1] < self.n:
            x = nn.functional.pad(x, (0, self.n - x.shape[-1]))
        S, n = self.nstacks, self.n
        v = x.unsqueeze(-2).expand(*lead, S, n)
        for k, s in enumerate(self.strides):
            g = n // (2 * s)
            t = self.twiddle[k].view(S, g, s, 2, 2)                 # (S, g, j, c, d)
            v = torch.einsum("...sgdj,sgjcd->...sgcj", v.reshape(*lead, S, g, 2, s), t)
            v = v.reshape(*lead, S, n)
        y = v.reshape(*lead, S * n)[..., :self.out_size]
        return y if self.bias is None else y + self.bias

    def extra_repr(self):
        return f"n={self.n}, strides={self.strides}, nstacks={self.nstacks}, out={self.out_size}"


def dct2_matrix(n):
    """Orthonormal DCT-II as an (out, in) matrix: c = D x, x = D^T c."""
    k = torch.arange(n, dtype=torch.float64)
    D = torch.cos(math.pi / n * (k[None, :] + 0.5) * k[:, None]) * math.sqrt(2.0 / n)
    D[0] /= math.sqrt(2.0)
    return D.float()


class _BN(nn.BatchNorm1d):
    """BatchNorm over the feature axis of (B, T, C) activations (folds at export)."""

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ResButterflyBlock(nn.Module):
    """x + alpha * B2(ReLU(BN(B1 x))), alpha a scalar initialised at 0 (ReZero).

    The identity path keeps the layer full-rank whatever the twiddles do (the
    published butterfly layers were near-singular); BN and alpha fold into B1 / B2.
    On the engine: B1 with BN folded + ReLU epilogue, then B2 with the residual as
    the epilogue's seed vector.
    """

    def __init__(self, H, lin_cfg, norm="batch"):
        super().__init__()
        self.b1 = make_linear(H, H, cfg=lin_cfg)
        self.bn = _BN(H) if norm == "batch" else nn.Identity()
        self.b2 = make_linear(H, H, cfg=lin_cfg)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return x + self.alpha * bfly(self.b2, torch.relu(self.bn(bfly(self.b1, x))))


class ButterflyGRU(nn.Module):
    """Multi-layer GRU with fused butterfly projections (W_ih, W_hh: H -> 3H, one
    stack per gate at power-of-two widths), the input projection hoisted out of the
    time loop. Same function as nsnet2.layers.StructuredGRU (fused), at ~half the
    per-step butterfly calls. Replaces gru_qat's Triton butterfly GRU, which races:
    its output differs run to run by up to ~0.9 at batch 64 (and sometimes at 2-8).
    PyTorch gate convention: n = tanh(x_n + r * (W_hn h + b_hn))."""

    def __init__(self, H_in, H, layers, nblocks=1):
        super().__init__()
        self.H = H
        self.x_proj = nn.ModuleList([make_linear(H_in if i == 0 else H, 3 * H, cfg={
            "kind": "butterfly", "nblocks": nblocks, "init": "randn"}) for i in range(layers)])
        self.h_proj = nn.ModuleList([make_linear(H, 3 * H, cfg={
            "kind": "butterfly", "nblocks": nblocks, "init": "ortho"}) for i in range(layers)])

    def forward(self, x):
        H = self.H
        for xp, hp in zip(self.x_proj, self.h_proj):
            gx = bfly(xp, x)
            h = x.new_zeros(x.shape[0], H)
            out = []
            for t in range(x.shape[1]):
                gh = hp(h)
                xr, xz, xn = gx[:, t].split(H, -1)
                hr, hz, hn = gh.split(H, -1)
                r = torch.sigmoid(xr + hr)
                z = torch.sigmoid(xz + hz)
                nt = torch.tanh(xn + r * hn)
                h = nt + z * (h - nt)
                out.append(h)
            x = torch.stack(out, dim=1)
        return x, None


class MinGRULayer(nn.Module):
    """minGRU (Feng et al. 2024): z = sigma(Wz x), h~ = Wh x, h = (1-z) h + z h~.

    No recurrent matrix: both projections are one butterfly (2 stacks) hoisted out
    of the time loop. On the engine that is one butterfly task (sigmoid LUT on the z
    stack, RAW on h~) plus exactly the Hadamard op the GRU already uses,
    h' = z (h~ - h) + h.
    """

    def __init__(self, H_in, H, lin_cfg):
        super().__init__()
        self.H = H
        self.proj = make_linear(H_in, 2 * H, cfg=lin_cfg)

    def forward(self, x):
        zl, ht = bfly(self.proj, x).split(self.H, dim=-1)
        z = torch.sigmoid(zl)
        h = x.new_zeros(x.shape[0], self.H)
        out = []
        for t in range(x.shape[1]):
            h = h + z[:, t] * (ht[:, t] - h)
            out.append(h)
        return torch.stack(out, dim=1)


class RecurrentStack(nn.Module):
    """Two minGRU or LRU layers with a ReLU after each (both cells are linear in
    their state, unlike the GRU's tanh, so the nonlinearity has to come from outside;
    on the engine a ReLU is a K=0 element-wise task)."""

    def __init__(self, kind, H, layers, lin_cfg, lru_r=(0.5, 0.99)):
        super().__init__()
        if kind == "mingru":
            self.layers = nn.ModuleList([MinGRULayer(H, H, lin_cfg) for _ in range(layers)])
        elif kind == "lru":
            from torch_structured import LRU
            self.layers = nn.ModuleList([
                LRU(H, H, 1, batch_first=True, kind="butterfly", r_min=lru_r[0], r_max=lru_r[1])
                for _ in range(layers)])
        else:
            raise ValueError(kind)
        self.kind = kind

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]
            x = torch.relu(x)
        return x, None


def _load_mean(h, n_bins=N_BINS):
    cfg = getattr(h, "input_norm", None)
    if not cfg:
        return None
    with open(_resolve_stats_path(cfg["path"])) as f:
        mean = json.load(f)["mean"]
    return torch.tensor(mean[:n_bins], dtype=torch.float32)


def _finish(noisy_mag, noisy_pha, mask256):
    """(B, T, 256) gain -> NSNet2's (mag, pha, com) with the Nyquist gain = bin 255's."""
    mask = torch.cat([mask256, mask256[..., -1:]], dim=-1).transpose(1, 2)
    mag = noisy_mag * mask
    com = torch.stack((mag * torch.cos(noisy_pha), mag * torch.sin(noisy_pha)), dim=-1)
    return mag, noisy_pha, com


# ---------------------------------------------------------------------------
# R-family: NSNet2 topology on butterflies (R, R2b, A, B, C, F)
# ---------------------------------------------------------------------------

class BflyNSNet(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        a = h.arch
        F, H, nb = N_BINS, a.get("hidden", 512), a.get("nblocks", 1)
        lin = {"kind": "butterfly", "nblocks": nb, "init": "randn"}
        self.cepstral = a.get("encoder", "butterfly") == "cepstral"
        mean = _load_mean(h)
        self.input_norm = mean is not None
        if self.input_norm:
            self.register_buffer("in_mean", mean)
        if self.cepstral:
            self.register_buffer("dct", dct2_matrix(F))
        self.fc_in = make_linear(2 * F if self.cepstral else F, H, cfg=lin)

        rnn = a.get("rnn", "gru")
        if rnn == "gru":
            self.rnn = ButterflyGRU(H, H, 2, nblocks=nb)
        else:
            self.rnn = RecurrentStack(rnn, H, 2, lin, tuple(a.get("lru_r", (0.5, 0.99))))

        fc = a.get("fc", {"kind": "plain", "n": 2})
        self.fc_kind = fc.get("kind", "plain")
        if self.fc_kind == "plain":
            self.fcs = nn.ModuleList([make_linear(H, H, cfg=lin) for _ in range(fc.get("n", 2))])
        elif self.fc_kind == "residual":
            self.fcs = nn.ModuleList([ResButterflyBlock(H, lin, fc.get("norm", "batch"))
                                      for _ in range(fc.get("n", 3))])
        else:
            raise ValueError(self.fc_kind)
        self.fc_out = make_linear(H, 2 * F if self.cepstral else F, cfg=lin)

    def forward(self, noisy_mag, noisy_pha):
        x = noisy_mag.transpose(1, 2)[..., :N_BINS]
        if self.input_norm:
            x = x - self.in_mean
        if self.cepstral:
            x = torch.cat([x, x @ self.dct.T], dim=-1)          # [spectrum ; cepstrum]
        h = torch.relu(bfly(self.fc_in, x))
        h, _ = self.rnn(h)
        for fc in self.fcs:
            h = torch.relu(bfly(fc, h)) if self.fc_kind == "plain" else fc(h)
        o = bfly(self.fc_out, h)
        if self.cepstral:
            s, q = o.split(N_BINS, dim=-1)
            o = s + q @ self.dct                                 # spectral + iDCT(cepstral) logits
        return _finish(noisy_mag, noisy_pha, torch.sigmoid(o))


# ---------------------------------------------------------------------------
# D / E: a frequency axis kept through the network
# ---------------------------------------------------------------------------

class StageGRU(nn.Module):
    """GRU layer on a (C channels x 256 bins) grid with stride-subset butterflies:
    the input projection (hoisted out of the time loop) carries the given strides,
    the recurrent one only the channel strides -- a per-bin recurrence over channels,
    like the per-band RNNs of FSPEN / DPCRN. Per-gate stacks, as on the engine."""

    def __init__(self, n, x_strides, h_strides):
        super().__init__()
        self.n = n
        self.x_proj = StageButterfly(n, x_strides, nstacks=3)
        self.h_proj = StageButterfly(n, h_strides, nstacks=3, init="ortho")

    def forward(self, x):
        n = self.n
        gx = self.x_proj(x)
        h = x.new_zeros(x.shape[0], n)
        out = []
        for t in range(x.shape[1]):
            gh = self.h_proj(h)
            xr, xz, xn = gx[:, t].split(n, -1)
            hr, hz, hn = gh.split(n, -1)
            r = torch.sigmoid(xr + hr)
            z = torch.sigmoid(xz + hz)
            nt = torch.tanh(xn + r * hn)
            h = nt + z * (h - nt)
            out.append(h)
        return torch.stack(out, dim=1)


class BflyGridNet(nn.Module):
    """Each frame is C channels x 256 bins, channel-major (index c*256 + f), so a
    butterfly stride < 256 mixes bins within a channel and a stride >= 256 mixes
    channels at the same bin. The frequency strides are spread over the layers --
    {1,2,4} then {8,16,32} then {64,128} -- so the receptive field along frequency
    grows with depth like a dilated-conv stack, but with frequency-specific weights;
    every layer also mixes channels. The decoder mirrors the encoder; with skips
    (E) each decoder level adds the encoder activation of the same resolution, a
    frequency U-Net. BN folds into the butterflies; the skip add is an epilogue seed.
    """

    ENC = ([1, 2, 4], [8, 16, 32], [64, 128])

    def __init__(self, h):
        super().__init__()
        self.h = h
        a = h.arch
        C = a.get("channels", 4)
        F = N_BINS
        n = C * F
        ch = [F * 2 ** i for i in range(int(math.log2(C)))]          # channel strides
        self.skips = bool(a.get("skips", False))
        # residual: each square level (enc1, enc2, dec2, dec1, dec0) becomes x + ReLU(BN(B x)),
        # an identity path at no extra butterfly cost (on the engine: the level's ReLU task,
        # then a K = 0 element-wise add of x).
        self.residual = bool(a.get("residual", False))
        mean = _load_mean(h)
        self.input_norm = mean is not None
        if self.input_norm:
            self.register_buffer("in_mean", mean)
        e0, e1, e2 = self.ENC
        self.enc0 = StageButterfly(F, e0, nstacks=C, in_size=F, out_size=n)
        self.enc1 = StageButterfly(n, e1 + ch)
        self.enc2 = StageButterfly(n, e2 + ch)
        self.bn_e = nn.ModuleList([_BN(n) for _ in range(3)])
        self.rnn = nn.ModuleList([StageGRU(n, ch + [1], ch), StageGRU(n, ch + [1], ch)])
        self.dec2 = StageButterfly(n, e2[::-1] + ch[::-1])
        self.dec1 = StageButterfly(n, e1[::-1] + ch[::-1])
        self.dec0 = StageButterfly(n, e0[::-1] + ch[::-1])
        self.bn_d = nn.ModuleList([_BN(n) for _ in range(3)])
        self.out = StageButterfly(n, ch[::-1], out_size=F)            # channel 0 = the gain

    def forward(self, noisy_mag, noisy_pha):
        x = noisy_mag.transpose(1, 2)[..., :N_BINS]
        if self.input_norm:
            x = x - self.in_mean
        def level(b, bn, v):
            y = torch.relu(bn(b(v)))
            return v + y if self.residual else y

        e0 = torch.relu(self.bn_e[0](self.enc0(x)))
        e1 = level(self.enc1, self.bn_e[1], e0)
        e2 = level(self.enc2, self.bn_e[2], e1)
        g = e2
        for layer in self.rnn:
            g = layer(g)
        d = g
        for dec, bn, e in ((self.dec2, self.bn_d[0], e2), (self.dec1, self.bn_d[1], e1),
                           (self.dec0, self.bn_d[2], e0)):
            d = level(dec, bn, d + e if self.skips else d)
        return _finish(noisy_mag, noisy_pha, torch.sigmoid(self.out(d)))


# ---------------------------------------------------------------------------
# Factory and engine cost
# ---------------------------------------------------------------------------

def build_generator(h):
    """NSNet2 unless the config carries an ``arch`` block."""
    arch = getattr(h, "arch", None) or (h.get("arch") if hasattr(h, "get") else None)
    if not arch:
        return NSNet2(h)
    kind = arch.get("type")
    if kind == "bflynet":
        return BflyNSNet(h)
    if kind == "bflygrid":
        return BflyGridNet(h)
    raise ValueError(f"unknown arch type {kind!r}")


def engine_pairs(model):
    """Butterfly pairs per frame = engine pair-issue cycles at one pair per cycle
    (§8: ~80 % of a frame's engine cycles). Every projection runs once per frame,
    recurrent ones included. The cepstral arm's two fixed 256-point DCTs run on the
    FFT pair words and are NOT counted here (~2 x 1.1K pairs)."""
    total = 0
    for m in model.modules():
        if _TSButterfly is not None and isinstance(m, _TSButterfly):
            total += m.nstacks * m.nblocks * m.log_n * (m.n // 2)
        elif isinstance(m, StageButterfly):
            total += m.nstacks * len(m.strides) * (m.n // 2)
    return total
