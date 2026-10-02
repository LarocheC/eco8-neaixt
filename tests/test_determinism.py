"""Bitwise-reproducible training helpers (common/determinism.py).

These are the fast, dataset-free guards. The end-to-end acceptance tests --
two runs of the real trainer agreeing bitwise, and resume-from-step-k equalling
the uninterrupted run -- live in ``nsnet2/determinism_accept.py``, because they
need a GPU and VoiceBank-DEMAND.
"""

import random

import numpy as np
import pytest
import torch

from common.determinism import (apply_determinism, rng_state, seed_everything,
                                set_rng_state)

CUDA = torch.cuda.is_available()


def _draw(gen):
    """One sample from every stream a training step can touch."""
    out = [torch.randn(4), random.random(), float(np.random.rand()),
           int(torch.empty((), dtype=torch.int64).random_(generator=gen))]
    if CUDA:
        out.append(torch.randn(4, device="cuda"))
    return out


def _same(a, b):
    return all(torch.equal(x, y) if torch.is_tensor(x) else x == y
               for x, y in zip(a, b))


def test_rng_roundtrip_covers_every_stream():
    gen = torch.Generator()
    gen.manual_seed(11)
    snap = rng_state(gen)
    first = _draw(gen)
    set_rng_state(snap, gen)
    assert _same(first, _draw(gen))


def test_snapshot_survives_weights_only_load(tmp_path):
    """The do_ checkpoint must still load under torch.load's default.

    torch>=2.6 defaults ``weights_only=True`` and ``common.utils.load_checkpoint``
    relies on that default. A raw ``np.random.get_state()`` tuple carries an
    ndarray whose unpickler global is not allowlisted, so storing one would make
    every checkpoint fail to load -- i.e. break resume outright. Regression
    guard for exactly that.
    """
    gen = torch.Generator()
    gen.manual_seed(5)
    path = tmp_path / "do_test"
    torch.save({"rng": rng_state(gen), "steps": 7}, path)

    loaded = torch.load(path, map_location="cuda" if CUDA else "cpu")
    first = _draw(gen)
    set_rng_state(loaded["rng"], gen)
    # map_location moves the saved ByteTensors onto the GPU; the restore has to
    # bring them back or every setter below raises.
    gen.manual_seed(5)
    set_rng_state(loaded["rng"], gen)
    assert _same(first, _draw(gen))


def test_seed_everything_is_reproducible():
    seed_everything(1234)
    a = (random.random(), float(np.random.rand()))
    seed_everything(1234)
    assert a == (random.random(), float(np.random.rand()))


def test_apply_determinism_off_is_a_noop():
    assert apply_determinism(False) == []


def test_apply_determinism_on_reports_what_it_set():
    on = apply_determinism(True)
    assert "cudnn.deterministic" in on
    assert torch.backends.cudnn.deterministic is True


def test_set_rng_state_tolerates_absent_and_partial_state():
    set_rng_state(None)
    set_rng_state({})
    set_rng_state({"python": random.getstate()})      # no torch/numpy keys


@pytest.mark.skipif(not CUDA, reason="needs a GPU")
def test_cuda_rng_is_captured():
    """The discriminator's nn.Dropout(0.3) advances the CUDA RNG every step, so
    a snapshot that skipped it would silently desynchronise a resume."""
    snap = rng_state()
    assert "torch_cuda" in snap
    first = torch.randn(8, device="cuda")
    set_rng_state(snap)
    assert torch.equal(first, torch.randn(8, device="cuda"))


# ---------------------------------------------------------------------------
# The resume contract, without a GPU or VoiceBank-DEMAND.
#
# train.py resumes mid-epoch by restoring the epoch-start snapshot, re-creating
# the DataLoader iterator, skipping the batches already trained on, and only
# then restoring the checkpointed snapshot. These exercise that exact sequence
# against a toy dataset that draws from the global ``random`` the way
# Dataset.__getitem__ crops do -- including num_workers=0, where the cropping
# happens in the main process instead of in a seeded worker.
# ---------------------------------------------------------------------------

from torch.utils.data import DataLoader  # noqa: E402

from common.dataset import data_generator, seed_worker  # noqa: E402


class _CropToy(torch.utils.data.Dataset):
    """Stands in for Dataset: the sample depends on the RNG, not just the index."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.tensor([float(idx), random.random()])


def _loader(gen, workers, batch=2, n=12):
    return DataLoader(_CropToy(n), batch_size=batch, shuffle=False, drop_last=True,
                      num_workers=workers, worker_init_fn=seed_worker, generator=gen)


def _run(workers, epochs=3, cut=(1, 2)):
    """Iterate `epochs` epochs, snapshotting as train.py does.

    Returns (all batches seen, the epoch-start snapshot of the cut epoch, the
    snapshot taken right after the cut batch, and the batches from the cut
    onwards -- i.e. what a correct resume has to reproduce).
    """
    cut_epoch, cut_batch = cut
    gen = data_generator(99)
    seed_everything(7)
    seen, epoch_snap, mid_snap, tail = [], None, None, []
    for epoch in range(epochs):
        snap = rng_state(gen)
        if epoch == cut_epoch:
            epoch_snap = snap
        for i, b in enumerate(_loader(gen, workers)):
            seen.append(b)
            if epoch > cut_epoch or (epoch == cut_epoch and i > cut_batch):
                tail.append(b)
            if epoch == cut_epoch and i == cut_batch:
                mid_snap = rng_state(gen)
    return seen, epoch_snap, mid_snap, tail


def _resume(workers, epoch_snap, mid_snap, epochs=3, cut=(1, 2)):
    """Replay from the cut the way train.py does, and return the batches."""
    cut_epoch, cut_batch = cut
    gen = data_generator(99)
    out, pending = [], mid_snap
    for epoch in range(cut_epoch, epochs):
        skip = 0
        if epoch == cut_epoch:
            skip = cut_batch + 1
            set_rng_state(epoch_snap, gen)
        # Mirrors train.py: replay the skipped batches by driving the
        # iterator, then restore BEFORE the next batch is drawn.
        it = iter(_loader(gen, workers))
        for _ in range(skip):
            try:
                next(it)
            except StopIteration:
                break
        if pending is not None:
            set_rng_state(pending, gen)
            pending = None
        out.extend(it)
    return out


@pytest.mark.parametrize("workers", [0, 2])
def test_resume_replays_the_same_batches(workers):
    """Skip-then-restore must reproduce every batch from the cut to the end,
    across the epoch boundary that follows it."""
    _, epoch_snap, mid_snap, tail = _run(workers)
    replayed = _resume(workers, epoch_snap, mid_snap)
    assert len(replayed) == len(tail) and len(tail) > 0
    for a, b in zip(tail, replayed):
        assert torch.equal(a, b)


@pytest.mark.parametrize("workers", [0, 2])
def test_seeded_loader_is_reproducible_run_to_run(workers):
    a = _run(workers)[0]
    b = _run(workers)[0]
    assert all(torch.equal(x, y) for x, y in zip(a, b))
