"""Is NSNet2 / butterfly training bitwise reproducible? Run this twice, compare.

Runs N real training steps (generator + MetricGAN discriminator with batch PESQ,
the train.py losses, AdamW, the kernel side-table reset) from a fixed seed, then
saves every generator + discriminator weight and prints a hash of the data seen.
Two processes with the same --mode must give identical weights if training is
deterministic.

    python -m nsnet2.determinism_probe --config configs/ba_R.json --mode both --out a.pt
    python -m nsnet2.determinism_probe --config configs/ba_R.json --mode both --out b.pt
    python -m nsnet2.determinism_probe --compare a.pt b.pt

Modes: default (the current trainer), cudnn (cudnn.deterministic), ts
(torch_structured.set_deterministic: butterfly backward without atomics),
both, full (torch.use_deterministic_algorithms -- fails on the discriminator's
adaptive_max_pool2d backward). --data-generator gives the DataLoader its own
seeded generator + seed_worker, so the crops no longer depend on how much RNG
the model's init consumed.

Measured 2026-10-02 (RTX 4090, torch 2.14, torch-structured 1.3.0, ba_R, 12 steps,
batch 16): default / cudnn / ts each differ run to run (max |dw| ~2e-4); both is
bitwise identical. Data batches were byte-identical in every mode for one model,
but differ ACROSS architectures with the same seed unless --data-generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--mode", default="default", choices=("default", "cudnn", "ts", "both", "full"))
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--data-generator", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    if a.compare:
        import torch
        x, y = (torch.load(p) for p in a.compare)
        print(f"max |diff| {(x - y).abs().max().item():.3e}   bitwise equal: {torch.equal(x, y)}")
        return
    if a.mode == "full":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import torch.nn.functional as F
    import torch_structured
    from torch.utils.data import DataLoader
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table as kst
    from common.env import AttrDict
    from common.dataset import (Dataset, data_generator, load_voicebank_demand, mag_pha_istft,
                                mag_pha_stft, seed_worker)
    from common.discriminator import MetricDiscriminator, batch_pesq
    from nsnet2.bfly_arch import build_generator

    torch.backends.cudnn.benchmark = False
    if a.mode in ("cudnn", "both"):
        torch.backends.cudnn.deterministic = True
    if a.mode in ("ts", "both"):
        torch_structured.set_deterministic(True)
    if a.mode == "full":
        torch.use_deterministic_algorithms(True)

    h = AttrDict(json.load(open(a.config)))
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed(h.seed)
    dev = torch.device("cuda")
    gen = build_generator(h).to(dev)
    disc = MetricDiscriminator().to(dev)
    og = torch.optim.AdamW(gen.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    od = torch.optim.AdamW(disc.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    kw = dict(worker_init_fn=seed_worker, generator=data_generator(h.seed)) if a.data_generator else {}
    loader = DataLoader(Dataset(load_voicebank_demand()["train"], h.segment_size, h.sampling_rate,
                                split=True, shuffle=True, seed=h.seed),
                        num_workers=2, batch_size=a.batch, shuffle=False, drop_last=True, **kw)
    nf, hop, win, cf = h.n_fft, h.hop_size, h.win_size, h.compress_factor
    dh = hashlib.sha256()
    torch.cuda.reset_peak_memory_stats()
    t0 = None
    for step, (clean, noisy) in enumerate(loader):
        if step == 2:
            torch.cuda.synchronize()
            t0 = time.time()
        dh.update(clean.numpy().tobytes())
        dh.update(noisy.numpy().tobytes())
        clean, noisy = clean.to(dev), noisy.to(dev)
        ones = torch.ones(clean.shape[0], device=dev)
        cm, _, cc = mag_pha_stft(clean, nf, hop, win, cf)
        nm, npha, _ = mag_pha_stft(noisy, nf, hop, win, cf)
        mg, pg, cg = gen(nm, npha)
        ag = mag_pha_istft(mg, pg, nf, hop, win, cf)
        _, _, cgh = mag_pha_stft(ag, nf, hop, win, cf)
        bp = batch_pesq(list(clean.cpu().numpy()), list(ag.detach().cpu().numpy()))
        od.zero_grad()
        ld = F.mse_loss(ones, disc(cm, cm).flatten())
        if bp is not None:
            ld = ld + F.mse_loss(bp.to(dev), disc(cm, mg.detach()).flatten())
        ld.backward()
        od.step()
        og.zero_grad()
        lg = (0.9 * F.mse_loss(cm, mg) + 0.2 * F.mse_loss(cc, cg) + 0.2 * F.mse_loss(cg, cgh)
              + 0.05 * F.mse_loss(disc(cm, mg).flatten(), ones) + 0.2 * F.l1_loss(clean, ag[..., :clean.shape[-1]]))
        lg.backward()
        og.step()
        kst.reset_table()
        if step + 1 == a.steps:
            break
    torch.cuda.synchronize()
    dt = (time.time() - t0) / max(1, a.steps - 2)
    w = torch.cat([p.detach().flatten().cpu() for p in list(gen.parameters()) + list(disc.parameters())])
    if a.out:
        torch.save(w, a.out)
    print(f"mode={a.mode} data={dh.hexdigest()[:12]} step={dt * 1000:.0f}ms "
          f"peak={torch.cuda.max_memory_allocated() / 2 ** 20:.0f}MiB loss={lg.item():.6f}")


if __name__ == "__main__":
    sys.exit(main())
