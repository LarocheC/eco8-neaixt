"""Pre-screen sparsity patterns on a trained checkpoint, without fine-tuning.

For each candidate pattern this reports, per matrix, the fraction of weight
*energy* the mask keeps:

    retained = ||W . M||_F / ||W||_F

It is a cheap proxy, not a quality metric — it says nothing about what
fine-tuning can recover. Its use is ordering: a pattern that throws away much
more energy than another at the same nominal sparsity is the worse starting
point, and matrices with a low retained fraction are the ones to watch (or
exclude) when the fine-tune does not converge.

    python -m nsnet2.sparsity_probe --config cp_nsnet2/config.json \
        --checkpoint cp_nsnet2/g_best
"""

from __future__ import annotations

import argparse
import json

import torch

from common.env import AttrDict
from common.utils import load_checkpoint
from nsnet2.export_sparse import collect_matrices
from nsnet2.model import NSNet2
from nsnet2.sparsity import build_mask, parse_pattern

DEFAULT_PATTERNS = ("2:4", "4:8", "1x4:80", "unstructured:50", "unstructured:80")


def probe(model: torch.nn.Module, patterns, *, axis: str = "in",
          tail: str = "keep", scope: str = "matrix") -> dict:
    out = {}
    mats = collect_matrices(model)
    dense_norm = torch.cat([w.flatten() for _, w, _ in mats]).norm()
    for pattern in patterns:
        rows, kept_flat = [], []
        for name, w, _ in mats:
            mask = build_mask(w, pattern, axis=axis, tail=tail, scope=scope)
            kept = w * mask
            kept_flat.append(kept.flatten())
            total = w.norm().item()
            rows.append({
                "name": name,
                "shape": list(w.shape),
                "sparsity": 1.0 - mask.mean().item(),
                "retained_energy": kept.norm().item() / total if total else 1.0,
            })
        kept_all = torch.cat(kept_flat)
        out[pattern] = {
            "nominal_sparsity": parse_pattern(pattern)["sparsity"],
            "achieved_sparsity": 1.0 - (kept_all != 0).float().mean().item(),
            "retained_energy": (kept_all.norm() / dense_norm).item(),
            "matrices": rows,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--patterns", nargs="*", default=list(DEFAULT_PATTERNS))
    ap.add_argument("--axis", default="in", choices=("in", "out"))
    ap.add_argument("--scope", default="matrix", choices=("matrix", "row"))
    ap.add_argument("--json", default="", help="Optional path to dump the raw report.")
    a = ap.parse_args()

    with open(a.config) as f:
        h = AttrDict(json.load(f))
    model = NSNet2(h)
    model.load_state_dict(load_checkpoint(a.checkpoint, torch.device("cpu"))["generator"])
    model.eval()

    report = probe(model, a.patterns, axis=a.axis, scope=a.scope)

    names = [r["name"] for r in report[a.patterns[0]]["matrices"]]
    width = max(len(n) for n in names) + 2
    print(f"retained weight energy  ||W.M||_F / ||W||_F   (checkpoint: {a.checkpoint})")
    print(f"grouping axis = {a.axis}, block scope = {a.scope}\n")
    hdr = f"{'matrix':<{width}}" + "".join(f"{p:>18}" for p in a.patterns)
    print(hdr)
    print("-" * len(hdr))
    for i, name in enumerate(names):
        line = f"{name:<{width}}"
        for p in a.patterns:
            line += f"{report[p]['matrices'][i]['retained_energy']:>18.3f}"
        print(line)
    print("-" * len(hdr))
    line = f"{'ALL':<{width}}"
    for p in a.patterns:
        line += f"{report[p]['retained_energy']:>18.3f}"
    print(line)
    line = f"{'achieved sparsity':<{width}}"
    for p in a.patterns:
        line += f"{report[p]['achieved_sparsity']:>18.3f}"
    print(line)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
