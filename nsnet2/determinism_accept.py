"""Acceptance tests for bitwise-reproducible training (common/determinism.py).

Unlike ``nsnet2.determinism_probe``, which re-implements a training step in
order to bisect *where* the nondeterminism lives, this harness drives the real
``nsnet2.train`` in a subprocess and compares the checkpoints it writes. That
is the point: the resume path is trainer code, so testing a copy of it would
prove nothing.

    # two fresh runs must agree bitwise
    python -m nsnet2.determinism_accept twin   --config configs/ba_R.json --steps 50

    # stopping at step k and resuming must equal never having stopped
    python -m nsnet2.determinism_accept resume --config configs/ba_R.json --steps 50 --cut 20

    # same seed, different architecture -> same batches (needs data_generator)
    python -m nsnet2.determinism_accept batches --config configs/ba_R.json \
                                                --config-b configs/ba_D_grid.json

    # step time and peak GPU memory, flags on vs off
    python -m nsnet2.determinism_accept bench  --config configs/ba_R.json --batch 256

``--batch`` overrides the config's batch size (the configs ship batch 256, which
needs a 24 GB-class card). ``--flags-off`` runs the same comparison with the
reproducibility flags disabled, which is how you demonstrate that they are what
is doing the work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAGS = ("deterministic", "data_generator", "exact_resume")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def write_config(src, dst, flags_on=True, batch=None, extra=None):
    """Copy a training config with the reproducibility flags forced on/off."""
    with open(src) as f:
        cfg = json.load(f)
    for k in FLAGS:
        cfg[k] = bool(flags_on)
    if batch:
        cfg["batch_size"] = int(batch)
    cfg.update(extra or {})
    with open(dst, "w") as f:
        json.dump(cfg, f, indent=4)
    return dst


def run_trainer(config, ckpt_dir, max_steps, ckpt_every, env=None):
    """Run nsnet2.train to an exact step count. Returns the captured stdout."""
    cmd = [sys.executable, "-m", "nsnet2.train",
           "--config", config,
           "--checkpoint_path", ckpt_dir,
           "--max_steps", str(max_steps),
           "--checkpoint_interval", str(ckpt_every),
           "--summary_interval", str(10 ** 9),
           "--validation_interval", str(10 ** 9),   # keep the test to training only
           "--stdout_interval", "10",
           "--training_epochs", "10000"]
    e = dict(os.environ, PYTHONHASHSEED="0", **(env or {}))
    p = subprocess.run(cmd, cwd=REPO, env=e, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-4000:] + "\n" + p.stderr[-4000:] + "\n")
        raise SystemExit("trainer failed (rc={}) for {}".format(p.returncode, config))
    return p.stdout


def compare_ckpt(path_a, path_b, label):
    """Bitwise-compare two checkpoints. Returns (ok, worst_key, max_abs_diff)."""
    import torch
    a = torch.load(path_a, map_location="cpu")
    b = torch.load(path_b, map_location="cpu")

    def tensors(obj, prefix=""):
        if isinstance(obj, torch.Tensor):
            yield prefix, obj
        elif isinstance(obj, dict):
            for k, v in obj.items():
                yield from tensors(v, "{}/{}".format(prefix, k))

    ta, tb = dict(tensors(a)), dict(tensors(b))
    missing = set(ta) ^ set(tb)
    if missing:
        print("  {}: KEY MISMATCH {}".format(label, sorted(missing)[:5]))
        return False, None, None
    worst, worst_k, bad = 0.0, None, 0
    for k, va in ta.items():
        vb = tb[k]
        if torch.equal(va, vb):
            continue
        bad += 1
        if va.is_floating_point():
            d = (va.double() - vb.double()).abs().max().item()
            if d > worst:
                worst, worst_k = d, k
        elif worst_k is None:
            worst_k = k
    ok = bad == 0
    print("  {}: {} ({} tensors, {} differing{})".format(
        label, "BITWISE EQUAL" if ok else "DIFFER", len(ta), bad,
        "" if ok else ", max |d|={:.3e} at {}".format(worst, worst_k)))
    return ok, worst_k, worst


def final_ckpts(ckpt_dir, step):
    return (os.path.join(ckpt_dir, "g_{:08d}".format(step)),
            os.path.join(ckpt_dir, "do_{:08d}".format(step)))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def cmd_twin(a):
    """Two independent fresh runs to the same step must agree bitwise."""
    work = tempfile.mkdtemp(prefix="det_twin_", dir=a.workdir)
    cfg = write_config(a.config, os.path.join(work, "cfg.json"),
                       flags_on=not a.flags_off, batch=a.batch,
                       extra={"train_subset": a.subset} if a.subset else None)
    last = (a.steps // a.ckpt_every) * a.ckpt_every
    print("twin: {} x2, {} steps, batch {}, flags {}".format(
        os.path.basename(a.config), a.steps, a.batch or "(config)",
        "OFF" if a.flags_off else "ON"))
    for run in ("A", "B"):
        run_trainer(cfg, os.path.join(work, run), a.steps + 1, a.ckpt_every)
    ga, da = final_ckpts(os.path.join(work, "A"), last)
    gb, db = final_ckpts(os.path.join(work, "B"), last)
    ok = compare_ckpt(ga, gb, "generator  @ step {}".format(last))[0]
    ok &= compare_ckpt(da, db, "disc+optim @ step {}".format(last))[0]
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    return ok


def cmd_resume(a):
    """Stopping at --cut and resuming must equal the uninterrupted run."""
    work = tempfile.mkdtemp(prefix="det_resume_", dir=a.workdir)
    cfg = write_config(a.config, os.path.join(work, "cfg.json"),
                       flags_on=not a.flags_off, batch=a.batch,
                       extra={"train_subset": a.subset} if a.subset else None)
    last = (a.steps // a.ckpt_every) * a.ckpt_every
    cut = (a.cut // a.ckpt_every) * a.ckpt_every
    assert 0 < cut < last, "--cut must land on a checkpoint strictly inside the run"
    print("resume: {} straight-through vs cut at {} , compare at step {}".format(
        os.path.basename(a.config), cut, last))

    straight = os.path.join(work, "straight")
    run_trainer(cfg, straight, a.steps + 1, a.ckpt_every)

    cut_dir = os.path.join(work, "cut")
    run_trainer(cfg, cut_dir, cut + 1, a.ckpt_every)        # stops just after g_cut
    # drop anything written past the cut so the resume really starts there
    for f in sorted(os.listdir(cut_dir)):
        if (f.startswith("g_") or f.startswith("do_")) and f[-8:].isdigit() \
                and int(f[-8:]) > cut:
            os.remove(os.path.join(cut_dir, f))
    run_trainer(cfg, cut_dir, a.steps + 1, a.ckpt_every)    # resume -> same end step

    ga, da = final_ckpts(straight, last)
    gb, db = final_ckpts(cut_dir, last)
    ok = compare_ckpt(ga, gb, "generator  @ step {}".format(last))[0]
    ok &= compare_ckpt(da, db, "disc+optim @ step {}".format(last))[0]
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    return ok


def cmd_batches(a):
    """Same seed + different architecture must see the same crops.

    Reproduces train.py's exact start-up order (seed, build generator, build
    discriminator, then build the loader) because that order is the bug: model
    init consumes a different amount of global RNG per architecture, and
    without a loader generator the worker base seed is drawn from what is left.
    """
    import torch
    from common.env import AttrDict
    from common.dataset import Dataset, data_generator, load_voicebank_demand, seed_worker
    from common.discriminator import MetricDiscriminator
    from nsnet2.bfly_arch import build_generator
    from torch.utils.data import DataLoader

    def digest(config, use_gen):
        h = AttrDict(json.load(open(config)))
        if a.batch:
            h["batch_size"] = int(a.batch)
        torch.manual_seed(h.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(h.seed)
        build_generator(h)            # consumes RNG -- differs per architecture
        MetricDiscriminator()
        kw = dict(worker_init_fn=seed_worker, generator=data_generator(h.seed)) if use_gen else {}
        ds = Dataset(load_voicebank_demand(cache_dir=a.hf_cache_dir)["train"],
                     h.segment_size, h.sampling_rate, split=True, shuffle=True, seed=h.seed)
        dl = DataLoader(ds, num_workers=h.num_workers, shuffle=False,
                        batch_size=h.batch_size, drop_last=True, **kw)
        dh = hashlib.sha256()
        for n, (clean, noisy) in enumerate(dl):
            dh.update(clean.numpy().tobytes())
            dh.update(noisy.numpy().tobytes())
            if n + 1 == a.nbatch:
                break
        return dh.hexdigest()[:16]

    ok = True
    for use_gen in (False, True):
        da = digest(a.config, use_gen)
        db = digest(a.config_b, use_gen)
        same = da == db
        print("  data_generator={:<5} {} {}  {}  -> {}".format(
            str(use_gen), da, db,
            os.path.basename(a.config) + " vs " + os.path.basename(a.config_b),
            "SAME" if same else "DIFFER"))
        if use_gen:
            ok = same            # only the generator-on case is required to match
    return ok


def cmd_bench(a):
    """Step time and peak GPU memory, flags on vs off, via determinism_probe."""
    rows = []
    for mode in ("default", "both"):
        cmd = [sys.executable, "-m", "nsnet2.determinism_probe",
               "--config", a.config, "--mode", mode,
               "--steps", str(a.steps), "--batch", str(a.batch or 256)]
        if mode == "both":
            cmd.append("--data-generator")
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("mode=")), None)
        rows.append((mode, line or "FAILED: " + (p.stderr.strip().splitlines() or [""])[-1][:160]))
    for mode, line in rows:
        print("  flags {:<3} {}".format("off" if mode == "default" else "on", line))
    return all(l.startswith("mode=") for _, l in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", choices=("twin", "resume", "batches", "bench"))
    ap.add_argument("--config", default="configs/ba_R.json")
    ap.add_argument("--config-b", default="configs/ba_D_grid.json",
                    help="second architecture for the 'batches' test")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cut", type=int, default=20, help="resume: step to stop at")
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--nbatch", type=int, default=4, help="batches test: how many to hash")
    ap.add_argument("--flags-off", action="store_true",
                    help="run with the reproducibility flags disabled (control)")
    ap.add_argument("--subset", type=int, default=0,
                    help="truncate the training set to N utterances, so a short "
                         "run still crosses epoch boundaries")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the temp run directories")
    ap.add_argument("--hf-cache-dir", default=None)
    a = ap.parse_args()
    ok = {"twin": cmd_twin, "resume": cmd_resume,
          "batches": cmd_batches, "bench": cmd_bench}[a.test](a)
    print("{}: {}".format(a.test.upper(), "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
