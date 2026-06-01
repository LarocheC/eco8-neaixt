"""Benchmark the GRU swap on a realistic NSNet2 train step.

Compares cuDNN nn.GRU (kind=gru) against gru_qat.GRULayer (kind=triton)
at the configs/baseline.json shape: hidden=400, num_gru_layers=2,
T = (segment_size - n_fft)/hop_size + 1 with hop=256, segment=32000 -> ~124.

Usage:
    python -m nsnet2.bench_gru --batch 256 --T 124 --warmup 5 --iters 20
"""
from __future__ import annotations

import argparse
import time
import torch

from common.env import AttrDict
from nsnet2.model import NSNet2


def make_h(gru_kind: str, n_fft: int = 512, hidden: int = 400,
           num_layers: int = 2, **gru_kwargs) -> AttrDict:
    return AttrDict({
        "n_fft": n_fft,
        "hidden_dim": hidden,
        "fc_hidden_dim": 600,
        "num_gru_layers": num_layers,
        "linear": {"kind": "linear"},
        "gru": {"kind": gru_kind, **gru_kwargs},
    })


def time_train_step(model, mag, pha, n_iters: int, warmup: int) -> float:
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    target = torch.randn_like(mag)
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        out_mag, _, _ = model(mag, pha)
        loss = (out_mag - target).pow(2).mean()
        loss.backward()
        opt.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        opt.zero_grad(set_to_none=True)
        out_mag, _, _ = model(mag, pha)
        loss = (out_mag - target).pow(2).mean()
        loss.backward()
        opt.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / n_iters * 1000.0  # ms / iter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--T", type=int, default=124)
    p.add_argument("--n_fft", type=int, default=512)
    p.add_argument("--hidden", type=int, default=400)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--variants", nargs="+",
                   default=["gru", "triton"],
                   help="any of: gru, triton, triton_compile, triton_monarch4, "
                        "triton_monarch8, triton_butterfly, py_butterfly, "
                        "py_butterfly_2blocks, py_monarch4, py_monarch8")
    args = p.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_freq = args.n_fft // 2 + 1
    mag = torch.randn(args.batch, n_freq, args.T, device=device)
    pha = torch.randn(args.batch, n_freq, args.T, device=device)

    print(f"shape: B={args.batch} T={args.T} H={args.hidden} layers={args.num_layers}")
    print(f"warmup={args.warmup} iters={args.iters}\n")

    results = {}
    for variant in args.variants:
        if variant == "gru":
            h = make_h("gru", args.n_fft, args.hidden, args.num_layers)
        elif variant == "triton":
            h = make_h("triton", args.n_fft, args.hidden, args.num_layers)
        elif variant == "triton_compile":
            h = make_h("triton", args.n_fft, args.hidden, args.num_layers, compile_step=True)
        elif variant == "triton_monarch4":
            h = make_h("triton_monarch", args.n_fft, args.hidden, args.num_layers, nblocks=4)
        elif variant == "triton_monarch8":
            h = make_h("triton_monarch", args.n_fft, args.hidden, args.num_layers, nblocks=8)
        elif variant == "triton_butterfly":
            h = make_h("triton_butterfly", args.n_fft, args.hidden, args.num_layers)
        elif variant == "py_butterfly":
            h = make_h("butterfly", args.n_fft, args.hidden, args.num_layers)
        elif variant == "py_butterfly_2blocks":
            h = make_h("butterfly", args.n_fft, args.hidden, args.num_layers, nblocks=2)
        elif variant == "py_monarch4":
            h = make_h("monarch", args.n_fft, args.hidden, args.num_layers, nblocks=4)
        elif variant == "py_monarch8":
            h = make_h("monarch", args.n_fft, args.hidden, args.num_layers, nblocks=8)
        else:
            print(f"!! unknown variant {variant!r}, skipping")
            continue

        torch.manual_seed(0)
        try:
            model = NSNet2(h).to(device)
        except Exception as e:
            print(f"{variant:>20s}: BUILD FAILED: {e}")
            continue
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        try:
            ms = time_train_step(model, mag, pha, args.iters, args.warmup)
        except Exception as e:
            print(f"{variant:>20s}: RUN FAILED: {e}")
            continue
        results[variant] = ms
        print(f"{variant:>20s}: {ms:7.2f} ms/iter   ({n_params:.2f}M params)")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "gru" in results:
        base = results["gru"]
        print(f"\nrelative to cuDNN nn.GRU baseline ({base:.2f} ms):")
        for v, ms in results.items():
            speedup = base / ms
            print(f"  {v:>20s}: {speedup:5.2f}x  ({ms:.2f} ms)")


if __name__ == "__main__":
    main()
