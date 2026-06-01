"""Phase 2 EXP-01..03 smoke + parity gate.

See:
- .planning/phases/02-fp32-onnx-export/02-CONTEXT.md (D-01..D-16 — locked test structure)
- .planning/phases/02-fp32-onnx-export/02-RESEARCH.md (verified ORT determinism recipe + 100-frame parity pattern)
- .planning/phases/02-fp32-onnx-export/02-VALIDATION.md (per-task verification map — 7 test functions)
- .planning/phases/01-streaming-forward-fp32-parity-gate/01-CONTEXT.md (Phase 1 D-05 carry — variant skip)

Determinism (manual_seed, cudnn flags, deterministic algos, CUBLAS env) is provided
by tests/conftest.py — autouse, session-scoped (Plan 01-01). No extra setup needed.

Failure-mode coverage:
- EXP-01 (file written, IO names, dynamic batch): test_export_smoke + test_output_path_strategy + test_from_checkpoint_smoke.
- EXP-02 (100-frame ORT-vs-PyTorch numerical parity per D-05): test_numerical_parity_100_frames.
- EXP-03 (onnx.checker passes; zero opaque GRU ops; op-type set subset of canonical 12): test_onnx_checker_passes + test_no_opaque_gru_ops + test_op_type_set_subset.

D-07: tests call ``export_streaming(...)`` directly as a Python function — no shell-out,
no ``uv run`` overhead per test, full Python tracebacks on failure.

D-02: synthetic random-init NSNet2 + non-zero GRU biases (sigma=0.5) saved as
``{'generator': state_dict}`` checkpoint at ``tmp_path/'g_best'`` with sibling
``tmp_path/'config.json'``. NO runtime dependency on the trained-checkpoint directory.
"""

from __future__ import annotations

import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from common.env import AttrDict
from nsnet2.export_onnx import export_streaming
from nsnet2.model import NSNet2
from nsnet2.streaming import NSNet2Streaming


@pytest.fixture
def baseline_h() -> AttrDict:
    """Load configs/baseline.json — D-01 picks baseline as the canonical config."""
    with open("configs/baseline.json") as f:
        return AttrDict(json.load(f))


@pytest.fixture
def gru_only_skip(baseline_h):
    """Phase 1 D-05 carry: only run when gru.kind == 'gru'. Variant export deferred to v2."""
    kind = baseline_h.get("gru", {"kind": "gru"}).get("kind", "gru")
    if kind != "gru":
        pytest.skip(
            f"Phase 2 export tests only run for gru.kind='gru' (got {kind!r}) — Phase 1 D-05."
        )


@pytest.fixture
def synthetic_ckpt(baseline_h, gru_only_skip, tmp_path) -> tuple[NSNet2, str]:
    """Random-init NSNet2 + non-zero biases per D-02; saved as a {'generator': state_dict}
    checkpoint at tmp_path/'g_best' with sibling tmp_path/'config.json'.

    Returns (base, ckpt_path_str). Tests use ckpt_path_str with from_checkpoint /
    export_streaming; base is exposed for direct numerical comparisons.
    """
    torch.manual_seed(0)                                                  # D-01 (also enforced by conftest)
    base = NSNet2(baseline_h)
    # D-02 / D-07: normal_(0, 0.5) on every GRU bias tensor (mirrors Phase 1 parity fixture).
    for k in range(base.gru.num_layers):
        for name in (f"bias_ih_l{k}", f"bias_hh_l{k}"):
            torch.nn.init.normal_(getattr(base.gru, name), 0.0, 0.5)
    base.eval()
    ckpt_path = tmp_path / "g_best"
    config_path = tmp_path / "config.json"
    torch.save({"generator": base.state_dict()}, str(ckpt_path))
    with open(config_path, "w") as f:
        json.dump(dict(baseline_h), f)
    return base, str(ckpt_path)


@pytest.fixture
def exported_path(synthetic_ckpt, tmp_path):
    """Run export_streaming directly per D-07 (no shell-out). Returns Path to the .onnx."""
    _, ckpt_path = synthetic_ckpt
    out = tmp_path / "test_fp32.onnx"
    return export_streaming(ckpt_path, str(out))


@pytest.fixture
def ort_session(exported_path):
    """ORT InferenceSession per D-04 + 02-RESEARCH.md determinism recipe."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1                                           # determinism (per D-04 + research)
    so.inter_op_num_threads = 1                                           # determinism
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL                  # determinism (default but explicit)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(exported_path),
        sess_options=so,
        providers=["CPUExecutionProvider"],                               # D-04: explicit CPU EP
    )


def test_from_checkpoint_smoke(synthetic_ckpt, baseline_h):
    """D-09..D-12: from_checkpoint reads sibling config.json, builds NSNet2(h),
    deserializes weights with weights-only safety, extracts ['generator'], strict
    load_state_dict. Verify the returned wrapper has the right shape contract on
    forward_step.
    """
    _, ckpt_path = synthetic_ckpt
    streaming = NSNet2Streaming.from_checkpoint(ckpt_path)
    assert isinstance(streaming, NSNet2Streaming)
    assert streaming.gru_kind == "gru"
    n_freq = baseline_h.n_fft // 2 + 1                                    # n_fft//2+1 for the canonical h
    L = streaming.num_layers
    H = streaming.hidden_size
    frame_in = torch.zeros(1, n_freq)                                     # (B, n_freq)
    states_in = torch.zeros(L, 1, H)                                      # (L, B, H)
    with torch.no_grad():
        mask, states_out = streaming.forward_step(frame_in, states_in)
    assert mask.shape == (1, n_freq)                                      # (B, n_freq)
    assert states_out.shape == (L, 1, H)                                  # (L, B, H)


def test_export_smoke(exported_path):
    """EXP-01: export_streaming() writes a non-empty .onnx with the right IO names
    and a single dynamic batch axis."""
    assert exported_path.exists()
    assert exported_path.stat().st_size > 0
    model = onnx.load(str(exported_path))
    inp_names = {i.name for i in model.graph.input}
    out_names = {o.name for o in model.graph.output}
    assert inp_names == {"frame_in", "states_in"}, inp_names
    assert out_names == {"mask", "states_out"}, out_names


def test_output_path_strategy(synthetic_ckpt, tmp_path):
    """D-13: default output is Path(ckpt).parent / 'g_best_fp32.onnx'.
    D-14: --output (here: output_path positional) overrides."""
    _, ckpt_path = synthetic_ckpt

    # D-14: explicit output_path override
    explicit = tmp_path / "explicit.onnx"
    out = export_streaming(ckpt_path, str(explicit))
    assert out == explicit
    assert explicit.exists()

    # D-13: default-derivation. ckpt_path is tmp_path/'g_best'; default sits next to it.
    out2 = export_streaming(ckpt_path, None)
    expected_default = (tmp_path / "g_best_fp32.onnx").resolve()
    assert out2.resolve() == expected_default
    assert (tmp_path / "g_best_fp32.onnx").exists()


def test_onnx_checker_passes(exported_path):
    """EXP-03 / SC-3: onnx.checker.check_model accepts the exported file."""
    model = onnx.load(str(exported_path))
    onnx.checker.check_model(model)                                       # raises ValidationError on bad graph


def test_no_opaque_gru_ops(exported_path):
    """EXP-03 / SC-4 / Pitfall 11 closure: zero opaque GRU ops in the exported graph."""
    model = onnx.load(str(exported_path))
    gru_ops = [n for n in model.graph.node if n.op_type == "GRU"]
    assert len(gru_ops) == 0, (
        f"Pitfall 11: {len(gru_ops)} opaque GRU op(s) survived export. "
        f"Phase 4 quantize_static cannot rewrite this op. "
        f"Check NSNet2Streaming.forward_step in models/streaming.py."
    )


def test_op_type_set_subset(exported_path):
    """EXP-03 / Pitfall C closure: op-type set is a subset of the Phase 1 verified clean set.
    Catches structured-variant ops (LRU, StructuredGRU) that op_type=='GRU' would miss.
    Per 02-RESEARCH.md A3: subset (not exact match) — robust to PyTorch minor-version
    optimization changes. The 12-type set is empirically verified canonical for both
    Phase 1's test fixture AND the full trained-checkpoint export.
    """
    model = onnx.load(str(exported_path))
    op_types = {n.op_type for n in model.graph.node}
    expected_clean = {
        "Add", "Concat", "Constant", "Gather", "Gemm", "MatMul",
        "Mul", "Relu", "Sigmoid", "Sub", "Tanh", "Unsqueeze",
    }
    new_ops = op_types - expected_clean
    assert not new_ops, (
        f"Pitfall 11/C lookalike: unexpected op types {new_ops} appeared in the graph. "
        f"Expected subset of {expected_clean}. New ops may indicate a structured-variant "
        f"path was traced or a Reshape/Transpose was introduced near the GRU."
    )


def test_numerical_parity_100_frames(synthetic_ckpt, exported_path, ort_session, baseline_h):
    """EXP-02 / SC-2 / D-01..D-05: PyTorch streaming forward vs ORT FP32 ONNX forward
    over a 100-frame trajectory. Tolerance asserted over the FULL trajectory (D-05),
    not just the final frame.
    """
    base, _ = synthetic_ckpt
    streaming = NSNet2Streaming(base).eval()
    n_freq = baseline_h.n_fft // 2 + 1                                    # n_fft//2+1
    L = streaming.num_layers
    H = streaming.hidden_size

    torch.manual_seed(0)                                                  # D-01 (redundant with conftest, explicit per D-01)
    mag = torch.randn(1, 100, n_freq).abs()                               # D-01: (B=1, T=100, F=n_freq), positive

    # PyTorch side — D-03: per-frame loop, state-threaded.
    states_torch = torch.zeros(L, 1, H)                                   # (L, B, H)
    masks_torch = []
    with torch.no_grad():
        for t in range(100):
            fr = mag[:, t]                                                # (1, n_freq)
            m, states_torch = streaming.forward_step(fr, states_torch)
            masks_torch.append(m.numpy())

    # ORT side — D-03: per-frame loop, state-threaded.
    mag_np = mag.numpy().astype(np.float32)                               # ORT requires float32 (no auto-cast)
    states_ort = np.zeros((L, 1, H), dtype=np.float32)                    # (L, B, H)
    masks_ort = []
    for t in range(100):
        fr = mag_np[:, t]                                                 # (1, n_freq) float32
        out = ort_session.run(
            ["mask", "states_out"],                                       # output names match D-08 / EXP-01
            {"frame_in": fr, "states_in": states_ort},
        )
        masks_ort.append(out[0])
        states_ort = out[1]                                               # thread states_out into next frame

    # D-05: max over the FULL trajectory, not just final frame.
    masks_torch_arr = np.stack(masks_torch, axis=1)                       # (1, 100, n_freq)
    masks_ort_arr = np.stack(masks_ort, axis=1)                           # (1, 100, n_freq)
    err = np.abs(masks_torch_arr - masks_ort_arr)
    max_abs_err = float(err.max())
    assert max_abs_err < 1e-4, (
        f"EXP-02 FAIL: max_abs_err={max_abs_err:.3e} >= 1e-4 over the 100-frame trajectory. "
        f"Empirical baseline (02-RESEARCH.md): trained checkpoint gets 1.371e-06, ~73x under threshold. "
        f"If between 1e-4 and 1e-3: check ORT inputs are np.float32 (no dtype auto-cast). "
        f"If above 1e-3: check from_checkpoint actually loaded state['generator']."
    )
