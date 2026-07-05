"""Params + MACs accounting for BASENet configurations.

MACs are reported per second of 16 kHz audio (160 STFT frames at hop 100),
the convention used by the speech-enhancement comparison tables the BASENet
paper draws from (Table 1: MANNER 2.9G, CMGAN 20.9G, MP-SENet 37.2G, ...).

Counts multiply-accumulates via forward hooks:
  * Conv2d           — out_elems * (in_ch/groups) * kernel
  * Linear           — out_elems * in_features
  * GRU              — seq_len * n_dir * 3 * hidden * (input + hidden)  (per layer)
  * FreqResample     — the constant-matrix einsum: out_f * in_f * C * N
  * CrossBandAttention — the two einsums (logits + context) on top of its convs

Usage:
    uv run python -m basenet.profile_macs                 # current defaults
    from basenet.profile_macs import profile; profile(cfg_dict)
"""

import torch
from torch import nn

from basenet.model import BASENet, CrossBandAttention, FreqResample, build_basenet

FRAMES_PER_SECOND = 160  # hop 100 @ 16 kHz


def profile(h=None, causal=True, n_frames=FRAMES_PER_SECOND, verbose=False):
    """Return (params, macs, per_module) for the config dict `h`.

    `per_module` maps top-level child name -> dict(params=..., macs=...).
    MACs are for a single utterance of `n_frames` STFT frames, batch 1.
    """
    model = build_basenet(h, causal=causal) if not isinstance(h, BASENet) else h
    model.eval()

    macs = {"total": 0}
    owner = {}  # module -> top-level child name
    for child_name, child in model.named_children():
        for m in child.modules():
            owner[m] = child_name

    def add(mod, n):
        macs["total"] += n
        name = owner.get(mod, "?")
        macs[name] = macs.get(name, 0) + n

    hooks = []

    def conv_hook(mod, inp, out):
        kf, kt = mod.conv.kernel_size if hasattr(mod, "conv") else mod.kernel_size
        add(mod, out.numel() * (mod.in_channels // mod.groups) * kf * kt)

    def linear_hook(mod, inp, out):
        add(mod, out.numel() * mod.in_features)

    def gru_hook(mod, inp, out):
        x = inp[0]
        seq = x.shape[1] if mod.batch_first else x.shape[0]
        batch = x.shape[0] if mod.batch_first else x.shape[1]
        n_dir = 2 if mod.bidirectional else 1
        total = 0
        in_size = mod.input_size
        for _ in range(mod.num_layers):
            total += seq * batch * n_dir * 3 * mod.hidden_size * (in_size + mod.hidden_size)
            in_size = mod.hidden_size * n_dir
        add(mod, total)

    def freq_resample_hook(mod, inp, out):
        # einsum("of,bcfn->bcon"): out_f * in_f MACs per (b, c, n)
        b, c, _, n = inp[0].shape
        add(mod, b * c * n * mod.w.shape[0] * mod.w.shape[1])

    def xband_hook(mod, inp, out):
        # the Q/K/V/O convs are Conv2d children (already hooked); count the einsums.
        band_feats = inp[0]
        total = 0
        n_bands = len(band_feats)
        for hb in band_feats:
            b, c, fb, n = hb.shape
            total += b * fb * n_bands * n * mod.d_k   # logits einsum
            total += b * fb * n_bands * n * c         # context einsum
        add(mod, total)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.GRU):
            hooks.append(m.register_forward_hook(gru_hook))
        elif isinstance(m, FreqResample):
            hooks.append(m.register_forward_hook(freq_resample_hook))
        elif isinstance(m, CrossBandAttention):
            hooks.append(m.register_forward_hook(xband_hook))

    fbins = model.n_features
    with torch.no_grad():
        model(torch.rand(1, fbins, n_frames), torch.rand(1, fbins, n_frames))
    for hk in hooks:
        hk.remove()

    params_per = {name: sum(p.numel() for p in child.parameters())
                  for name, child in model.named_children()}
    params = sum(params_per.values())
    per_module = {name: {"params": params_per.get(name, 0), "macs": macs.get(name, 0)}
                  for name in dict.fromkeys(list(params_per) + [k for k in macs if k != "total"])}

    if verbose:
        print(f"params: {params/1e6:.3f}M   MACs: {macs['total']/1e9:.3f}G / {n_frames} frames")
        for name, d in sorted(per_module.items(), key=lambda kv: -kv[1]["macs"]):
            print(f"  {name:15s} params {d['params']/1e3:8.1f}k   MACs {d['macs']/1e9:7.3f}G"
                  f"  ({100*d['macs']/max(macs['total'],1):4.1f}%)")
    return params, macs["total"], per_module


if __name__ == "__main__":
    for causal in (True, False):
        print(f"=== causal={causal} (paper: {'0.81M / 7.1G' if causal else '0.83M / 7.3G'}) ===")
        profile(causal=causal, verbose=True)
        print()
