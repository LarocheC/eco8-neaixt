"""Bitwise-reproducible training: deterministic kernels and RNG capture/restore.

Two independent things are needed to make a run reproduce itself exactly, and
they are controlled by separate config flags because they have different costs
and different blast radii (see ``nsnet2/train.py``):

``deterministic``
    Deterministic *kernels*. Two sources of run-to-run drift were measured with
    ``python -m nsnet2.determinism_probe`` (2026-10-02, RTX 4090, torch 2.14,
    torch-structured 1.3.0, 12 steps, batch 16): the MetricDiscriminator's conv
    backward picks nondeterministic cuDNN algorithms, and torch_structured's
    Triton butterfly backward accumulates ``d_twiddle`` with ``tl.atomic_add``,
    whose float summation order varies per launch. Either one alone still
    differs run to run (max |dw| ~2e-4); both together are bitwise identical.
    ``torch.use_deterministic_algorithms(True)`` is NOT usable here: the
    discriminator's ``adaptive_max_pool2d`` backward has no deterministic CUDA
    implementation.

``exact_resume``
    Resuming mid-epoch must land on the same weights as never having stopped.
    That needs more than the optimizer and LR schedule (both of which were
    already verified to restore exactly, and a crash+resume still cost the
    butterfly control ~0.07 PESQ): it needs every RNG stream and the position
    within the epoch. The discriminator contains ``nn.Dropout(0.3)`` and trains
    in ``.train()`` mode, so the CUDA RNG advances on *every* step -- restoring
    it is not optional.

Two snapshots per checkpoint are stored, because the DataLoader draws its
worker base seed when the *iterator* is created, once per epoch, before any
batch is seen:

``epoch_start_rng``   state as of the top of the epoch, before the iterator
                      exists. Restoring it and re-creating the iterator makes
                      the workers replay the epoch's crops exactly.
``rng``               state at the moment the checkpoint was written, i.e.
                      after the batch that was just trained on. Restored once
                      the skipped batches have been consumed.
"""

import random

import numpy as np
import torch


def apply_determinism(enabled):
    """Switch on deterministic kernels. Returns the list of what was enabled.

    A no-op when ``enabled`` is false, so the default path is bit-for-bit the
    behaviour of runs started before this module existed.
    """
    if not enabled:
        return []
    enabled_now = []
    torch.backends.cudnn.deterministic = True
    enabled_now.append('cudnn.deterministic')
    try:
        import torch_structured
        torch_structured.set_deterministic(True)
        enabled_now.append('torch_structured.set_deterministic')
    except (ImportError, AttributeError):
        # No butterfly layers installed: the cuDNN half is still the whole fix.
        pass
    return enabled_now


def seed_everything(seed):
    """Seed the streams train.py never seeded: the main process's python and
    numpy RNGs.

    ``torch.manual_seed`` / ``torch.cuda.manual_seed`` were already called, and
    ``Dataset.__init__`` shuffles with its own ``random.Random(seed)``, but the
    *global* ``random`` and ``numpy`` generators in the main process were left
    at whatever entropy the interpreter started with. Nothing numerically
    important reads them today (crops happen in workers, which ``seed_worker``
    seeds), so this changes no result -- but it makes the captured RNG snapshot
    reproducible, and it closes the hole for ``num_workers=0``, where
    ``Dataset.__getitem__`` crops from the main process's ``random``.
    """
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))


def _pack_numpy(state):
    """numpy's legacy MT19937 state as weights_only-safe primitives.

    ``torch.load`` defaults to ``weights_only=True`` from torch 2.6, and
    ``common.utils.load_checkpoint`` relies on that default. A raw
    ``np.random.get_state()`` tuple carries an ndarray, whose unpickler global
    is not allowlisted -- storing it would make every do_ checkpoint fail to
    load, i.e. break resume outright. Keys go in as an int64 tensor instead.
    """
    name, keys, pos, has_gauss, cached = state
    return {'name': str(name),
            'keys': torch.from_numpy(np.asarray(keys, dtype=np.int64)),
            'pos': int(pos), 'has_gauss': int(has_gauss), 'cached': float(cached)}


def _unpack_numpy(packed):
    keys = np.asarray(packed['keys'].cpu().numpy(), dtype=np.uint32)
    return (packed['name'], keys, packed['pos'], packed['has_gauss'], packed['cached'])


def rng_state(data_gen=None):
    """Snapshot every RNG stream a training step can consume.

    ``data_gen`` is the DataLoader's ``generator=``, which seeds the workers;
    it is kept separate from the global torch CPU RNG precisely so that model
    init cannot shift the data stream.
    """
    state = {
        'torch_cpu': torch.get_rng_state(),
        'python': random.getstate(),
        'numpy': _pack_numpy(np.random.get_state(legacy=True)),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    if data_gen is not None:
        state['data_gen'] = data_gen.get_state()
    return state


def set_rng_state(state, data_gen=None):
    """Restore a snapshot from ``rng_state``. Tolerates missing keys.

    ``load_checkpoint`` calls ``torch.load(..., map_location=device)``, which
    moves the saved ByteTensors onto the GPU; every RNG setter below wants them
    back on the CPU, hence the explicit ``.cpu()``.
    """
    if not state:
        return
    if 'torch_cpu' in state:
        torch.set_rng_state(state['torch_cpu'].cpu())
    if 'python' in state:
        # torch.save round-trips the tuple's inner list as a list; random.setstate
        # requires a tuple of (int, tuple-of-ints, int|None).
        version, keys, gauss = state['python']
        random.setstate((version, tuple(keys), gauss))
    if 'numpy' in state:
        np.random.set_state(_unpack_numpy(state['numpy']))
    if 'torch_cuda' in state and torch.cuda.is_available():
        saved = [s.cpu() for s in state['torch_cuda']]
        if len(saved) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(saved)
        else:
            # Resuming onto a different GPU count: seed device 0 and leave the
            # rest, rather than crashing on a shape mismatch.
            torch.cuda.set_rng_state(saved[0])
    if data_gen is not None and 'data_gen' in state:
        data_gen.set_state(state['data_gen'].cpu())
