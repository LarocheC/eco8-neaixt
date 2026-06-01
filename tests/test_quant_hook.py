"""tests/test_quant_hook.py - Phase 6 in-training quant-eval hook gate (TRN-01..05).

Seven pytest functions cover TRN-01..05 unit-level + the CONCERNS.md alignment
grep-gate. Tests build a synthetic mini generator + tiny HF DatasetDict + scratch
ckpt_dir, then drive quant_hook.run_quant_eval directly with a synthetic
SummaryWriter pointed at tmp_path/'logs'.

DET-02 sidestep preserved: zero deserialization-side load sites in this file.
The synthetic fixture uses torch.save (writing only). Actual checkpoint reading
happens INSIDE quant.quantize_checkpoint via NSNet2Streaming.from_checkpoint
(Phase 2 D-09 owner).

Test files are self-contained per the convention established by tests/test_quant.py
and tests/test_inference_onnx.py - copy fixture helpers verbatim, do NOT cross-import.

Test list (locked names per VALIDATION.md per-task map):
  1. test_run_quant_eval_smoke              - VALIDATION 06-01-01 / TRN-01
  2. test_disabled_short_circuit            - VALIDATION 06-01-02 / TRN-01
  3. test_tb_scalar_names_exact             - VALIDATION 06-01-03 / TRN-02
  4. test_stdout_breakdown_cycle1_only      - VALIDATION 06-01-04 / TRN-03
  5. test_calib_indices_deterministic       - VALIDATION 06-01-05 / TRN-04
  6. test_no_quant_imports_when_disabled    - VALIDATION 06-01-06 / TRN-05
  7. test_no_parallel_best_int8_pesq        - TRN-04 / CONCERNS.md alignment grep-gate

See .planning/phases/06-in-training-quant-eval-hook/06-CONTEXT.md (D-01..D-04) and
06-VALIDATION.md (test name lock).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from datasets import Dataset, DatasetDict

from common.env import AttrDict
from nsnet2.model import NSNet2
from nsnet2.export_onnx import export_streaming
from nsnet2.quant import quantize_checkpoint
import nsnet2.quant as quant_mod


# Repo root resolved once so subprocess-spawn tests don't depend on pytest invocation cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------
# Helper builders ported VERBATIM from tests/test_inference_onnx.py (lines 64-110).
# Self-contained per the locked convention -- do NOT cross-import sibling test files.
# ----------------------------------------------------------------------------

# Tiny dimensions: ~50 ONNX nodes, completes integration smoke in ~2s.
_H_TINY_BASE = {
    "n_fft": 64,
    "hop_size": 16,
    "win_size": 64,
    "compress_factor": 0.3,
    "sampling_rate": 16000,
    "seed": 1234,
    "hidden_dim": 16,
    "fc_hidden_dim": 16,
    "num_gru_layers": 2,
    "linear": {"kind": "linear"},
    "gru": {"kind": "gru"},
}


def _make_split(audios, prefix, sr=16000):
    """Build a Dataset.from_dict-compatible payload mirroring HF VBD schema."""
    return {
        "id":    [f"{prefix}_{i}" for i in range(len(audios))],
        "clean": [{"path": f"{prefix}_{i}_c", "array": a, "sampling_rate": sr} for i, a in enumerate(audios)],
        "noisy": [{"path": f"{prefix}_{i}_n", "array": a, "sampling_rate": sr} for i, a in enumerate(audios)],
    }


def _make_dataset_dict(train_audios, test_audios) -> DatasetDict:
    return DatasetDict({
        "train": Dataset.from_dict(_make_split(train_audios, "tr")),
        "test":  Dataset.from_dict(_make_split(test_audios,  "te")),
    })


def _random_audio(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n_samples).astype(np.float32)


def _redirect_hf_cache(monkeypatch, tmp_path: Path):
    """Point datasets_config.HF_DATASETS_CACHE at tmp_path so vbd-hashes/ writes are isolated per test.

    Mirrors tests/test_inference_onnx.py:100-110. Required for any test that constructs the
    VBDCalibrationReader (i.e., goes through quantize_checkpoint).
    """
    from datasets import config as datasets_config
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", str(tmp_path))
    # calibration.py imports `datasets_config` directly; patch that reference too.
    import nsnet2.calibration as calibration_mod
    monkeypatch.setattr(calibration_mod.datasets_config, "HF_DATASETS_CACHE", str(tmp_path))


# ----------------------------------------------------------------------------
# Phase 6 NEW fixture: in-memory (generator, h, hf, sw, ckpt_dir) for direct
# hook invocation. Adapts tests/test_inference_onnx.py:_make_synthetic_cp_dir_with_int8
# + adds in-memory generator + SummaryWriter (the hook will write its state_dict
# during run_quant_eval; we do NOT pre-write it here).
# ----------------------------------------------------------------------------

def _make_synthetic_train_state(tmp_path, monkeypatch):
    """Build (generator, h, hf, sw, ckpt_dir) for direct hook invocation.

    Pipeline (mirrors _make_synthetic_cp_dir_with_int8 + adds in-memory generator + sw):
      1. Build tiny NSNet2(h) with normal_(0, 0.5) bias init (real activation ranges).
      2. Synthesize tiny HF DatasetDict (5 train + 2 test utts).
      3. Patch quant.load_voicebank_demand -> synthetic ds (offline).
      4. Build sw = SummaryWriter(tmp_path / "logs").
      5. ckpt_dir = tmp_path / "cp_synthetic" (the hook will create .quant_cache/ inside).

    Returns (generator, h, hf, sw, ckpt_dir).
    """
    _redirect_hf_cache(monkeypatch, tmp_path)
    torch.manual_seed(0)
    h_dict = dict(_H_TINY_BASE)
    h_dict["num_gpus"] = 0
    h = AttrDict(h_dict)
    # Wrap h.quant as nested AttrDict (env.py AttrDict is flat; the hook's defensive guard
    # also re-wraps -- but the test fixture mimics the post-rewrap shape for clarity).
    h.quant = AttrDict({"enabled": True, "n_calib_utts": 3})

    base = NSNet2(h)
    # Mirror tests/test_quant.py:122-127: normal_(0, 0.5) on every GRU bias tensor.
    for k in range(base.gru.num_layers):
        for name in (f"bias_ih_l{k}", f"bias_hh_l{k}"):
            torch.nn.init.normal_(getattr(base.gru, name), 0.0, 0.5)
    base.eval()

    train_audios = [_random_audio(2048, seed=i) for i in range(5)]
    test_audios  = [_random_audio(2048, seed=100 + i) for i in range(2)]
    hf = _make_dataset_dict(train_audios, test_audios)

    # Patch the HF dataset loader the hook reaches indirectly via quantize_checkpoint
    # (which calls quant.load_voicebank_demand). Lambda accepts cache_dir kwarg.
    monkeypatch.setattr(quant_mod, "load_voicebank_demand", lambda cache_dir=None: hf)

    from torch.utils.tensorboard import SummaryWriter
    sw = SummaryWriter(str(tmp_path / "logs"))

    ckpt_dir = tmp_path / "cp_synthetic"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return base, h, hf, sw, ckpt_dir


# ----------------------------------------------------------------------------
# Test 1: run_quant_eval smoke -- return-dict shape + non-NaN values (TRN-01).
# ----------------------------------------------------------------------------

def test_run_quant_eval_smoke(tmp_path, monkeypatch):
    """VALIDATION 06-01-01 / TRN-01: run_quant_eval returns dict with expected keys, non-NaN values."""
    generator, h, hf, sw, ckpt_dir = _make_synthetic_train_state(tmp_path, monkeypatch)
    from nsnet2.quant_hook import run_quant_eval
    result = run_quant_eval(
        generator, h, hf, sw, steps=1, ckpt_dir=ckpt_dir,
        fp32_pesq=2.5, log_breakdown=False, num_gpus=0,
    )
    expected_keys = {"int8_pesq", "fp32_pesq", "delta", "t_export", "t_calibrate", "t_ort_validate"}
    assert set(result.keys()) == expected_keys, result.keys()
    assert result["fp32_pesq"] == 2.5, result["fp32_pesq"]
    assert result["t_export"] > 0 and np.isfinite(result["t_export"]), result["t_export"]
    assert result["t_calibrate"] > 0 and np.isfinite(result["t_calibrate"]), result["t_calibrate"]
    assert result["t_ort_validate"] > 0 and np.isfinite(result["t_ort_validate"]), result["t_ort_validate"]
    # int8_pesq may be NaN if all 2 synthetic utts fail PESQ on random noise -- that's OK,
    # the contract is "key present + finite if any utterance succeeded".
    # Scratch artifacts produced by the hook.
    assert (ckpt_dir / ".quant_cache" / "g_best_fp32.onnx").exists()
    assert (ckpt_dir / ".quant_cache" / "g_train.onnx").exists()


# ----------------------------------------------------------------------------
# Test 2: disabled short-circuit -- defensive .get() gate semantics (TRN-01).
# ----------------------------------------------------------------------------

def test_disabled_short_circuit():
    """VALIDATION 06-01-02 / TRN-01: the lazy-import gate in train.py uses defensive .get()
    so missing OR explicitly-false h.quant.enabled BOTH short-circuit before the import.

    Hand-evaluates the same expression on AttrDict shapes Phase 6 must support:
      - h with NO quant key            -> False (backward-compat with pre-Phase-6 cp_baseline)
      - h.quant = {"enabled": False}   -> False
      - h.quant = {"enabled": True}    -> True
    """
    train_text = (REPO_ROOT / "nsnet2" / "train.py").read_text()
    assert 'h.get("quant", {}).get("enabled", False)' in train_text, (
        "train.py missing the defensive lazy-import gate substring"
    )

    h_missing = AttrDict({"seed": 0})
    h_false   = AttrDict({"seed": 0, "quant": {"enabled": False, "n_calib_utts": 200}})
    h_true    = AttrDict({"seed": 0, "quant": {"enabled": True, "n_calib_utts": 200}})

    assert h_missing.get("quant", {}).get("enabled", False) is False
    assert h_false.get("quant", {}).get("enabled", False) is False
    assert h_true.get("quant", {}).get("enabled", False) is True


# ----------------------------------------------------------------------------
# Test 3: TB scalar names exact (TRN-02 verbatim names).
# ----------------------------------------------------------------------------

def test_tb_scalar_names_exact(tmp_path, monkeypatch):
    """VALIDATION 06-01-03 / TRN-02: TB scalars exactly named 'Validation/PESQ Score (int8)'
    and 'Validation/PESQ Delta (FP32-int8)'.

    Drives run_quant_eval against the synthetic mini state, flushes+closes the SummaryWriter,
    then reads the resulting events file via tensorboard.backend.event_processing.event_accumulator
    (NEW pattern Phase 6 introduces -- no prior test in this repo reads TB events back from disk).
    """
    generator, h, hf, sw, ckpt_dir = _make_synthetic_train_state(tmp_path, monkeypatch)
    from nsnet2.quant_hook import run_quant_eval
    run_quant_eval(generator, h, hf, sw, steps=1, ckpt_dir=ckpt_dir,
                   fp32_pesq=2.5, log_breakdown=False, num_gpus=0)
    sw.flush()
    sw.close()

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(tmp_path / "logs"))
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    assert "Validation/PESQ Score (int8)" in tags, tags
    assert "Validation/PESQ Delta (FP32-int8)" in tags, tags


# ----------------------------------------------------------------------------
# Test 4: stdout breakdown cycle-1-only (TRN-03).
# ----------------------------------------------------------------------------

def test_stdout_breakdown_cycle1_only(tmp_path, monkeypatch, capsys):
    """VALIDATION 06-01-04 / TRN-03: cycle-1 stdout = verbatim
    'Quant eval: export={t1}s, calibrate={t2}s, ort_validate={t3}s'; later cycles silent.

    Exercises the log_breakdown= keyword-only flag: True -> breakdown fires; False -> silent.
    """
    generator, h, hf, sw, ckpt_dir = _make_synthetic_train_state(tmp_path, monkeypatch)
    from nsnet2.quant_hook import run_quant_eval

    # Cycle 1: log_breakdown=True -> stdout breakdown fires.
    run_quant_eval(generator, h, hf, sw, steps=1, ckpt_dir=ckpt_dir,
                   fp32_pesq=2.5, log_breakdown=True, num_gpus=0)
    captured = capsys.readouterr()
    assert "Quant eval: export=" in captured.out, captured.out
    assert "calibrate="          in captured.out, captured.out
    assert "ort_validate="       in captured.out, captured.out

    # Cycle 2: log_breakdown=False -> NO breakdown line (later cycles silent).
    run_quant_eval(generator, h, hf, sw, steps=2, ckpt_dir=ckpt_dir,
                   fp32_pesq=2.5, log_breakdown=False, num_gpus=0)
    captured2 = capsys.readouterr()
    assert "Quant eval: export=" not in captured2.out, captured2.out


# ----------------------------------------------------------------------------
# Test 5: calibration indices deterministic across cycles (TRN-04).
# ----------------------------------------------------------------------------

def test_calib_indices_deterministic(tmp_path, monkeypatch):
    """VALIDATION 06-01-05 / TRN-04: two consecutive run_quant_eval calls produce same calibration_hash
    (read from int8 ONNX metadata). Verifies cache-once-via-deterministic-RNG semantic --
    VBDCalibrationReader's rng.sample(h.seed) inside calibration.py:79-80 is the load-bearing
    determinism source; the hook's serialized config.json pins h.calibration.seed = h.seed
    AND h.calibration.num_utterances = h.quant.n_calib_utts so indices are byte-identical.
    """
    generator, h, hf, sw, ckpt_dir = _make_synthetic_train_state(tmp_path, monkeypatch)
    from nsnet2.quant_hook import run_quant_eval

    run_quant_eval(generator, h, hf, sw, steps=1, ckpt_dir=ckpt_dir,
                   fp32_pesq=2.5, num_gpus=0)
    int8_path = ckpt_dir / ".quant_cache" / "g_train.onnx"
    model = onnx.load(str(int8_path))
    hash1 = next(p.value for p in model.metadata_props if p.key == "calibration_hash")

    run_quant_eval(generator, h, hf, sw, steps=2, ckpt_dir=ckpt_dir,
                   fp32_pesq=2.5, num_gpus=0)
    model = onnx.load(str(int8_path))
    hash2 = next(p.value for p in model.metadata_props if p.key == "calibration_hash")

    assert hash1 == hash2, f"calibration_hash drifted across cycles: {hash1} != {hash2}"


# ----------------------------------------------------------------------------
# Test 6: no quant imports when disabled (TRN-05 lazy-import gate).
# ----------------------------------------------------------------------------

def test_no_quant_imports_when_disabled(tmp_path):
    """VALIDATION 06-01-06 / TRN-05: train.py module-load imports nothing from quant.py /
    quant_hook / onnxruntime.quantization. The lazy-import boundary IS the
    `if h.get("quant", {}).get("enabled", False):` block.

    Spawns a Python subprocess that imports `train` via importlib.util.spec_from_file_location
    (so __name__ != '__main__' and the if __name__ == '__main__': main() guard does not fire),
    then audits sys.modules for forbidden entries:
      - 'quant_hook'                          (the hook module itself)
      - 'quant'                               (the heavy quant module loaded transitively via the hook)
      - any module starting with 'onnxruntime.quantization' (ORT static-PTQ subpackage)
    """
    audit = (
        "import sys; "
        "import importlib.util; "
        "spec = importlib.util.spec_from_file_location('train', 'nsnet2/train.py'); "
        "mod = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(mod); "
        "forbidden = [m for m in list(sys.modules) "
        "             if m == 'nsnet2.quant_hook' or m == 'nsnet2.quant' or "
        "                m.startswith('onnxruntime.quantization')]; "
        "print('FORBIDDEN_LOADED:', forbidden); "
        "assert not forbidden, forbidden"
    )
    result = subprocess.run(
        [sys.executable, "-c", audit],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "FORBIDDEN_LOADED: []" in result.stdout, result.stdout


# ----------------------------------------------------------------------------
# Test 7 (defense-in-depth): no parallel best_int8_pesq variable (TRN-04 / CONCERNS.md).
# ----------------------------------------------------------------------------

def test_no_parallel_best_int8_pesq():
    """TRN-04 / CONCERNS.md alignment: hook does NOT add a parallel best_int8_pesq variable
    mirroring the existing best_pesq-resets-on-resume bug.

    Cheap grep gate (file read + 2 substring checks); fires at every test run as a
    regression guard against future edits that hoist a parallel-best mechanism.
    """
    train_text       = (REPO_ROOT / "nsnet2" / "train.py").read_text()
    quant_hook_text  = (REPO_ROOT / "nsnet2" / "quant_hook.py").read_text()
    assert "best_int8_pesq" not in train_text, "train.py introduced best_int8_pesq -- TRN-04 forbids"
    assert "best_int8_pesq" not in quant_hook_text, "quant_hook.py introduced best_int8_pesq -- TRN-04 forbids"
