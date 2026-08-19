"""Build a self-contained hand-off bundle of sparse NSNet2 checkpoints.

Produces one directory per pattern (weights + masks + manifest + golden
vectors), a README describing the layout and conventions, and a standalone
``verify.py`` the recipient can run with nothing but numpy. Everything is
tarred at the end.

    python -m nsnet2.package_handoff \\
        --arm dense=cp_ov_dense_control --arm 2:4=cp_ov_2to4 \\
        --out nsnet2_sparse_handoff

Each ``--arm NAME=CHECKPOINT_DIR`` reads ``<dir>/config.json`` and
``<dir>/g_best``. The pattern is taken from the config, so an arm is exported
under whatever mask it was actually trained with.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile

VERIFY_SCRIPT = '''"""Check a hand-off bundle with nothing but numpy: python verify.py [dir]"""
import json
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "."
manifest = json.load(open(f"{root}/manifest.json"))
npz = np.load(f"{root}/weights.npz")

print(f"{manifest['model']}  pattern={manifest['pattern']}  "
      f"axis={manifest['group_axis']}  N_inference={manifest['N_inference']}")
print(f"{'matrix':<20}{'M':>6}{'K':>6}{'sparsity':>10}{'pattern':>10}{'ref':>10}")
print("-" * 62)

bad = 0
for m in manifest["matrices"]:
    name, M, K = m["name"], m["M"], m["K"]
    w = npz[f"{name}.weight"]
    mask = npz[f"{name}.mask"]

    assert w.shape == (M, K), f"{name}: shape {w.shape} != {(M, K)}"
    assert np.all(w[mask == 0] == 0), f"{name}: nonzero weight under a zero mask"
    assert int((mask != 0).sum()) == m["nonzero"], f"{name}: nonzero count"

    # Structural check: for an N:M pattern every full group of M columns along
    # K must hold at most N nonzeros. This is the contract a generated kernel
    # relies on, so verify it against the shipped array, not the manifest.
    ok = "n/a"
    pat = m["pattern"]
    if pat.startswith("1x"):
        # Block pattern: every full block of B columns along K must be kept or
        # dropped whole. Partial blocks would break a block-packed kernel.
        block = int(pat[2:].split(":")[0])
        nb = K // block
        if nb:
            counts = (w[:, :nb * block] != 0).reshape(M, nb, block).sum(-1)
            viol = int(((counts != 0) & (counts != block)).sum())
            bad += viol
            ok = "OK" if viol == 0 else f"{viol} BAD"
    elif ":" in pat and not pat.startswith("unstructured"):
        n, g = (int(v) for v in pat.split(":"))
        ng = K // g
        if ng:
            counts = (w[:, :ng * g] != 0).reshape(M, ng, g).sum(-1)
            viol = int((counts > n).sum())
            bad += viol
            ok = "OK" if viol == 0 else f"{viol} BAD"

    ref = "-"
    if f"{name}.ref_x" in npz.files:
        y = w @ npz[f"{name}.ref_x"]
        if f"{name}.bias" in npz.files:
            y = y + npz[f"{name}.bias"]
        err = np.max(np.abs(y - npz[f"{name}.ref_y"]))
        ref = f"{err:.2e}"
        if err > 1e-3:
            bad += 1
            ref += " BAD"

    print(f"{name:<20}{M:>6}{K:>6}{m['sparsity']:>10.4f}{ok:>10}{ref:>10}")

print("-" * 62)
print("FAILED" if bad else "all checks passed")
sys.exit(1 if bad else 0)
'''

README = """# NSNet2 sparse checkpoints — hand-off bundle

Speech-enhancement model (NSNet2: input FC, two GRU layers, three output FCs)
trained under fixed semi-structured sparsity masks, for evaluating sparse-dense
MatMul packing and code generation at batch 1.

Source code, training recipe and export tooling:
<https://github.com/LarocheC/eco8-neaixt>, branch `sparse-masks-rowfusion`.
See `SPARSE_MATMUL_COLLAB.md` there for the full method and results.

## What is in here

{arm_table}

One directory per pattern, each containing:

* `weights.npz` — per matrix: `<name>.weight` (float32, dense array **with
  explicit zeros**), `<name>.mask` (uint8, 1 = kept), `<name>.bias` (float32),
  and golden vectors `<name>.ref_x` / `<name>.ref_y`.
* `manifest.json` — shapes, pattern, grouping axis, achieved sparsity, ragged
  tail counts, and the value of N at inference vs training.

Plus `verify.py` at the top level, which needs only numpy:

```bash
python verify.py 2_4         # checks shapes, mask/weight agreement,
                             # N:M group conformance and the golden vectors
```

## Conventions

**Layout.** Every weight is row-major `(M, K)` and used as `y = W · x + b` with
`x` of shape `(K, N)`.

**N = 1 at deployment.** The model runs one 16 ms frame at a time, so each of
these is a matrix-vector product. During training N is 256 · T.

**Grouping runs along K.** For an N:M pattern, the groups of M are contiguous
*within a row* — i.e. along the input dimension, contiguous in memory for a
row-major `(M, K)` array. This matches the NVIDIA 2:4 convention. If your packer
wants the groups along the output dimension instead, say so and I will retrain;
the masking code supports both.

**Ragged tail.** `fc_in` has K = 257, which is 64 groups of 4 plus one leftover
column. That column is left dense, so `fc_in` measures 49.8% sparse rather than
exactly 50%. `manifest.json` reports `tail_elements` per matrix. Padding K to
260 instead is easy if that suits the kernel better.

**GRU gate packing.** `gru.weight_ih_l*` and `gru.weight_hh_l*` are `(3H, K)`:
PyTorch stacks the three GRU gates (r, z, n) along the **output** dimension, so
each gate is a contiguous block of rows and a group of 4 along K never straddles
a gate boundary. Each gate submatrix independently satisfies the pattern, so a
`1200 x 400` may be packed as one matrix or as three `400 x 400` with identical
results.

## Where the work is

The four GRU matrices are 69% of the weights and execute once per frame, so they
dominate the runtime. `gru.weight_hh_l0` and `gru.weight_hh_l1` sit inside the
recurrence and cannot be batched over time even in principle — they are the
strictest N=1 case in the model.

| matrix              |    M |   K |  params |
| ------------------- | ---: | --: | ------: |
| `fc_in`             |  400 | 257 | 102,800 |
| `gru.weight_ih_l0`  | 1200 | 400 | 480,000 |
| `gru.weight_hh_l0`  | 1200 | 400 | 480,000 |
| `gru.weight_ih_l1`  | 1200 | 400 | 480,000 |
| `gru.weight_hh_l1`  | 1200 | 400 | 480,000 |
| `fc1`               |  600 | 400 | 240,000 |
| `fc2`               |  600 | 600 | 360,000 |
| `fc_out`            |  257 | 600 | 154,200 |

## Quality

PESQ on the full 824-utterance VoiceBank-DEMAND test set. All arms were
fine-tuned from the same dense baseline on an identical schedule, so the mask is
the only variable.

{pesq_table}

The spread across all arms is ~0.012 PESQ while the run-to-run variation within
a single arm is ~0.010, so these are statistically indistinguishable — no
pattern here costs measurable quality. Pick whichever packs best.

Two caveats. All arms including the dense control sit ~0.07 below our published
200-epoch baseline (2.845), because these were shortened fine-tunes with a
freshly initialised discriminator; the comparison between arms is unaffected
since they all paid the same penalty. And these are FP32 — int8 behaviour under
this much sparsity is not yet characterised.

## Loading

```python
import json
import numpy as np

manifest = json.load(open("2_4/manifest.json"))
npz = np.load("2_4/weights.npz")

W = npz["gru.weight_hh_l0.weight"]      # (1200, 400) float32, explicit zeros
b = npz["gru.weight_hh_l0.bias"]        # (1200,)
m = npz["gru.weight_hh_l0.mask"]        # (1200, 400) uint8, 1 = kept

x = npz["gru.weight_hh_l0.ref_x"]       # (400,) float32
assert np.allclose(W @ x + b, npz["gru.weight_hh_l0.ref_y"], atol=1e-4)
```

## Questions

Happy to regenerate in a different format, a different grouping direction, a
different sparsity level, or with the mask expressed as indices rather than a
dense array — whatever your packer prefers.
"""


def parse_arm(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--arm expects NAME=CHECKPOINT_DIR, got {spec!r}")
    name, path = spec.split("=", 1)
    return name, path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", type=parse_arm, required=True,
                    metavar="NAME=CHECKPOINT_DIR")
    ap.add_argument("--out", default="nsnet2_sparse_handoff")
    ap.add_argument("--pesq", default="",
                    help="Optional JSON mapping arm name -> PESQ, for the README.")
    ap.add_argument("--no-tar", action="store_true")
    a = ap.parse_args()

    pesq = json.loads(a.pesq) if a.pesq else {}

    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out)

    rows = []
    for name, ckpt_dir in a.arm:
        # "2:4" -> "2_4", "1x4:80" -> "1x4_80", "unstructured:80" -> "unstructured_80"
        slug = name.replace(":", "_")
        dest = os.path.join(a.out, slug)
        cfg = os.path.join(ckpt_dir, "config.json")
        ckpt = os.path.join(ckpt_dir, "g_best")
        for f in (cfg, ckpt):
            if not os.path.isfile(f):
                raise SystemExit(f"missing {f}")
        print(f"exporting {name} from {ckpt_dir} -> {dest}")
        subprocess.run([sys.executable, "-m", "nsnet2.export_sparse",
                        "--config", cfg, "--checkpoint", ckpt, "--out", dest,
                        "--compress", "--reference"], check=True,
                       stdout=subprocess.DEVNULL)
        manifest = json.load(open(os.path.join(dest, "manifest.json")))
        total = sum(m["M"] * m["K"] for m in manifest["matrices"])
        nz = sum(m["nonzero"] for m in manifest["matrices"])
        rows.append({
            "dir": slug,
            "pattern": manifest["pattern"],
            "sparsity": 1 - nz / total,
            "pesq": pesq.get(name),
        })

    arm_table = ("| directory | pattern | sparsity |\n"
                 "| --------- | ------- | -------: |\n")
    for r in rows:
        arm_table += f"| `{r['dir']}` | {r['pattern']} | {100 * r['sparsity']:.1f}% |\n"

    pesq_table = ("| pattern | sparsity | PESQ |\n"
                  "| ------- | -------: | ---: |\n")
    for r in rows:
        val = f"{r['pesq']:.3f}" if r["pesq"] is not None else "—"
        pesq_table += f"| {r['pattern']} | {100 * r['sparsity']:.1f}% | {val} |\n"

    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write(README.format(arm_table=arm_table, pesq_table=pesq_table))
    with open(os.path.join(a.out, "verify.py"), "w") as f:
        f.write(VERIFY_SCRIPT)

    if not a.no_tar:
        tar_path = f"{a.out}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(a.out, arcname=os.path.basename(a.out))
        size = os.path.getsize(tar_path) / 1e6
        print(f"\nwrote {tar_path} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
