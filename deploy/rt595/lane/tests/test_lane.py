"""Tests for the `lane` auditor.

Two layers:

* **Synthetic** -- every gate is exercised against a hand-built ctx dict with no
  .tflite, no TFLM tree and no toolchain.  These always run.
* **Golden** -- pinned measurements against the real TFLM tree, the real
  resolver .cpp files and the five real artifacts.  These skip cleanly when the
  inputs are not on this box, but when they do run they are exact, so a silent
  regression in the resolver-header parse fails CI instead of quietly shrinking
  the supported-op set.

Run with::

    ~/.venvs/rt595-export/bin/python -m pytest deploy/rt595/lane/tests/test_lane.py -q
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

from lane.gates import (
    WARN,
    ALL_GATE_SPECS,
    G_SCALE_FINITE,
    Gate,
    GateResult,
    any_hard_fail,
    any_hard_skip,
    parse_io_layout,
    parse_resolver_cpp,
    run_gates,
)
from lane.gates.graph import check_accel_coverage, check_opset
from lane.gates.quant import (
    check_quant_struct,
    check_scale_finite,
    check_scale_floor,
    check_scale_positive,
)
from lane.gates.resolver import (
    check_resolver_arity,
    check_resolver_exact,
    check_resolver_lean,
    check_resolver_firmware,
)
from lane.gates.runtime import check_load_isolated
from lane.gates.structure import (
    check_io_layout,
    check_io_symmetry,
    check_single_subgraph,
)
from lane.report import (
    EXIT_BLOCKED,
    EXIT_INCOMPLETE,
    EXIT_OK,
    AuditReport,
    exit_code,
    render_json,
    render_text,
)
from lane.runtime import accel_map
from lane.runtime import tflm_introspect as ti
from lane.runtime.flatbuffer import ModelInfo, TensorInfo, load_model
from lane.runtime.litert_host import load_isolated

N_GATES = 14

# --------------------------------------------------------------------------
# GOLDEN NUMBERS -- measured on this tree, 2026-07-28.
# --------------------------------------------------------------------------
# micro_mutable_op_resolver.h has 115 `TfLiteStatus Add` lines: 96 builtin
# registrations + 17 custom registrations + the 2 machinery methods (AddBuiltin,
# AddCustom).  If any of these three numbers move, the parse changed and the
# whole opset audit is suspect.
GOLDEN_BUILTIN_METHODS = 96
GOLDEN_CUSTOM_METHODS = 17
GOLDEN_TOTAL_METHODS = GOLDEN_BUILTIN_METHODS + GOLDEN_CUSTOM_METHODS  # 113
GOLDEN_MACHINERY_METHODS = 2

# The calibrator floor seen on every ConvFSENet/NSNet2 state-feedback input.
GOLDEN_FLOOR_SCALE = 3.9215683651783184e-12

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEPLOY_ROOT = os.path.dirname(os.path.dirname(_HERE))
HOST_OUT = os.path.join(_DEPLOY_ROOT, "host_out")
RESOLVER_DIR = os.path.join(_DEPLOY_ROOT, "iss", "resolvers")
FIRMWARE_RESOLVER = os.path.join(_DEPLOY_ROOT, "app", "model_ops_micro.cpp")


def _tflm_tree():
    from lane.__main__ import default_tflm_tree

    return default_tflm_tree()


def _lib():
    from lane.__main__ import default_lib

    return default_lib()


TFLM_TREE = _tflm_tree()
needs_tree = pytest.mark.skipif(TFLM_TREE is None, reason="no tflite-micro checkout found")
needs_models = pytest.mark.skipif(
    not os.path.isdir(HOST_OUT), reason="deploy/rt595/host_out missing"
)


def _model(name):
    path = os.path.join(HOST_OUT, name)
    if not os.path.isfile(path):
        pytest.skip("missing artifact %s" % name)
    return load_model(path)


# --------------------------------------------------------------------------
# synthetic fixtures
# --------------------------------------------------------------------------


def mk_tensor(
    index,
    name="t",
    dtype="INT8",
    shape=(1, 4),
    scales=(0.5,),
    zps=(0,),
    qdim=0,
):
    return TensorInfo(
        index=index,
        name=name,
        dtype=dtype,
        shape=tuple(shape),
        scales=tuple(scales),
        zero_points=tuple(zps),
        is_quantized=bool(scales),
        quantized_dimension=qdim,
    )


def mk_model(
    tensors=None, ops=None, inputs=(0,), outputs=(1,), versions=None, n_subgraphs=1
):
    tensors = tensors or [mk_tensor(0, "in"), mk_tensor(1, "out")]
    ops = ops if ops is not None else {"CONV_2D": 2, "ADD": 1}
    return ModelInfo(
        path="/synthetic/model.tflite",
        n_subgraphs=n_subgraphs,
        op_histogram=dict(ops),
        tensors=tuple(tensors),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        op_versions=dict(versions or {k: 1 for k in ops}),
    )


# ==========================================================================
# G-SCALE-FINITE
# ==========================================================================


def test_scale_finite_passes_on_clean_model():
    r = check_scale_finite({"model": mk_model()})
    assert r.passed and not r.skipped
    assert r.gate is G_SCALE_FINITE
    assert r.evidence["n_nonfinite_tensors"] == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_scale_finite_catches_each_kind_of_nonfinite(bad):
    m = mk_model(tensors=[mk_tensor(0, "in"), mk_tensor(1, "poisoned", scales=(0.5, bad))])
    r = check_scale_finite({"model": m})
    assert not r.passed
    assert r.evidence["n_nonfinite_tensors"] == 1
    assert r.evidence["tensors"][0]["name"] == "poisoned"
    assert "non-finite" in r.detail


def test_scale_finite_is_hard_fail_severity():
    assert G_SCALE_FINITE.severity == "hard-fail"
    m = mk_model(tensors=[mk_tensor(0, "x", scales=(float("nan"),))])
    assert any_hard_fail([check_scale_finite({"model": m})])


def test_scale_finite_skips_without_model():
    r = check_scale_finite({})
    assert r.skipped and r.status == "SKIP"
    assert not r.is_hard_fail  # a skip must never block


def test_float_model_with_no_quantization_is_clean():
    m = mk_model(tensors=[mk_tensor(0, "f", dtype="FLOAT32", scales=())])
    r = check_scale_finite({"model": m})
    assert r.passed and r.evidence["n_quantized_tensors"] == 0


# ==========================================================================
# G-SCALE-FLOOR
# ==========================================================================


def test_scale_floor_flags_inputs_separately():
    tensors = [
        mk_tensor(0, "audio_in", scales=(0.01,)),
        mk_tensor(1, "state_in", scales=(GOLDEN_FLOOR_SCALE,)),
        mk_tensor(2, "some_intermediate", scales=(GOLDEN_FLOOR_SCALE,)),
    ]
    m = mk_model(tensors=tensors, inputs=(0, 1), outputs=(2,))
    r = check_scale_floor({"model": m})
    assert not r.passed  # warn severity, but still a finding
    assert r.gate.severity == "warn"
    assert r.evidence["n_floored"] == 2
    assert r.evidence["n_floored_inputs"] == 1
    assert r.evidence["floored_inputs"][0]["name"] == "state_in"
    assert r.evidence["floored_inputs"][0]["input_position"] == 1
    # The prose must name the population the ratio divides -- see
    # test_floored_input_detail_does_not_read_as_a_fraction_of_the_tensor_count.
    assert "1 of the model's 2 INPUTS" in r.detail


def test_scale_floor_warn_never_blocks_the_build():
    m = mk_model(tensors=[mk_tensor(0, "s", scales=(1e-13,))], inputs=(0,), outputs=(0,))
    assert not any_hard_fail([check_scale_floor({"model": m})])


def test_scale_floor_says_benign_when_no_input_is_floored():
    tensors = [mk_tensor(0, "in", scales=(0.02,)), mk_tensor(1, "mid", scales=(1e-12,))]
    m = mk_model(tensors=tensors, inputs=(0,), outputs=(1,))
    r = check_scale_floor({"model": m})
    assert not r.passed
    assert r.evidence["n_floored_inputs"] == 0
    assert "likely benign" in r.detail


def test_scale_floor_threshold_is_configurable():
    m = mk_model(tensors=[mk_tensor(0, "x", scales=(1e-9,))], inputs=(0,), outputs=(0,))
    assert check_scale_floor({"model": m}).passed
    assert not check_scale_floor({"model": m, "floor_threshold": 1e-8}).passed


def test_scale_floor_ignores_nonfinite_scales_when_finding_the_minimum():
    # A NaN must not be mistaken for a tiny scale -- that is G-SCALE-FINITE's job.
    m = mk_model(tensors=[mk_tensor(0, "x", scales=(float("nan"), 0.5))], inputs=(0,))
    r = check_scale_floor({"model": m})
    assert r.passed


def test_per_axis_scales_use_the_smallest_channel():
    m = mk_model(
        tensors=[mk_tensor(0, "w", scales=(0.4, 0.2, 1e-13))], inputs=(), outputs=()
    )
    r = check_scale_floor({"model": m})
    assert not r.passed and r.evidence["n_floored"] == 1


# ==========================================================================
# G-OPSET
# ==========================================================================


def test_opset_passes_when_every_op_is_supported():
    ctx = {
        "model": mk_model(ops={"CONV_2D": 3, "ADD": 1}),
        "supported_ops": {"CONV_2D", "ADD", "MUL"},
        "tflm_tree": "/some/tree",
    }
    r = check_opset(ctx)
    assert r.passed and "/some/tree" in r.detail


def test_opset_names_the_tree_it_consulted_on_failure():
    ctx = {
        "model": mk_model(ops={"POW": 1, "REDUCE_ALL": 1, "ADD": 4}),
        "supported_ops": {"ADD"},
        "tflm_tree": "/opt/tflite-micro",
    }
    r = check_opset(ctx)
    assert not r.passed
    assert "/opt/tflite-micro" in r.detail  # contract: must name the tree
    assert r.evidence["missing"] == ["POW", "REDUCE_ALL"]
    assert r.evidence["missing_counts"] == {"POW": 1, "REDUCE_ALL": 1}


def test_opset_skips_rather_than_failing_without_a_tree():
    r = check_opset({"model": mk_model(), "tflm_tree_error": "no tree here"})
    assert r.skipped and not r.is_hard_fail
    assert "no tree here" in r.detail


# ==========================================================================
# G-ACCEL-COVERAGE
# ==========================================================================


def test_accel_coverage_is_weighted_by_node_count():
    ctx = {
        "model": mk_model(ops={"CONV_2D": 90, "GATHER_ND": 10}),
        "accelerated": {"CONV_2D"},
        "accel_status": {"ok": True},
    }
    r = check_accel_coverage(ctx)
    assert r.evidence["fraction"] == pytest.approx(0.90)
    assert r.passed
    assert r.evidence["reference_c"] == {"GATHER_ND": 10}


def test_accel_coverage_warns_below_threshold():
    ctx = {
        "model": mk_model(ops={"CONV_2D": 1, "GATHER_ND": 99}),
        "accelerated": {"CONV_2D"},
        "accel_status": {"ok": True},
    }
    r = check_accel_coverage(ctx)
    assert not r.passed and not r.is_hard_fail
    assert r.evidence["biggest_gaps"][0] == ("GATHER_ND", 99)


def test_accel_coverage_skips_when_the_map_is_unavailable():
    ctx = {"model": mk_model(), "accelerated": set(), "accel_status": {"ok": False, "reason": "no xt-nm"}}
    r = check_accel_coverage(ctx)
    assert r.skipped and "no xt-nm" in r.detail


# ==========================================================================
# G-RESOLVER-EXACT
# ==========================================================================


def test_resolver_exact_passes_on_a_perfect_match():
    ctx = {
        "model": mk_model(ops={"CONV_2D": 2, "ADD": 1}),
        "resolver_path": "/x/ops.cpp",
        "resolver_ops": {"CONV_2D", "ADD"},
        "resolver_arity": 2,
        "resolver_n_registrations": 2,
    }
    assert check_resolver_exact(ctx).passed


def test_resolver_exact_reports_missing_direction():
    ctx = {
        "model": mk_model(ops={"FULLY_CONNECTED": 16, "ADD": 12}),
        "resolver_path": "/x/model_ops_micro.cpp",
        "resolver_ops": {"ADD"},
        "resolver_arity": 1,
        "resolver_n_registrations": 1,
    }
    r = check_resolver_exact(ctx)
    assert not r.passed and r.is_hard_fail
    assert r.evidence["missing"] == ["FULLY_CONNECTED"]
    assert r.evidence["unused"] == []
    assert "MISSING" in r.detail


def test_resolver_unused_direction_warns_but_does_not_block():
    """A dead registration wastes a slot; it does not stop the model booting.

    Regression guard for the headline false positive: folding both directions
    into one hard-fail made the only healthy artifact in the tree come back
    BLOCKED against its own resolver over a single dead AddTanh.
    """
    ctx = {
        "model": mk_model(ops={"ADD": 1}),
        "resolver_path": "/x/ops.cpp",
        "resolver_ops": {"ADD", "TANH"},
        "resolver_arity": 2,
        "resolver_n_registrations": 2,
    }
    exact = check_resolver_exact(ctx)
    assert exact.passed, "a surplus registration must not be a boot blocker"

    lean = check_resolver_lean(ctx)
    assert not lean.passed and not lean.is_hard_fail
    assert lean.gate.severity == WARN
    assert lean.evidence["unused"] == ["TANH"]
    assert "TANH" in lean.detail


def test_resolver_lean_passes_when_nothing_is_dead():
    ctx = {
        "model": mk_model(ops={"ADD": 1, "CONV_2D": 4}),
        "resolver_path": "/x/ops.cpp",
        "resolver_ops": {"ADD", "CONV_2D"},
        "resolver_arity": 2,
        "resolver_n_registrations": 2,
    }
    assert check_resolver_lean(ctx).passed


def test_resolver_exact_blocks_on_missing_even_when_also_unused():
    """The missing direction still hard-fails, regardless of surplus entries."""
    ctx = {
        "model": mk_model(ops={"ADD": 1, "GATHER_ND": 6}),
        "resolver_path": "/x/ops.cpp",
        "resolver_ops": {"ADD", "TANH"},
        "resolver_arity": 2,
        "resolver_n_registrations": 2,
    }
    r = check_resolver_exact(ctx)
    assert not r.passed and r.is_hard_fail
    assert r.evidence["missing"] == ["GATHER_ND"]
    assert "MISSING" in r.detail
    # ...and the dead entry is still reported, just by the warn-level gate.
    assert check_resolver_lean(ctx).evidence["unused"] == ["TANH"]


def test_resolver_arity_too_small_is_called_out_as_dropped_registrations():
    ctx = {
        "model": mk_model(ops={"ADD": 1, "MUL": 1}),
        "resolver_path": "/x/ops.cpp",
        "resolver_ops": {"ADD", "MUL"},
        "resolver_arity": 1,
        "resolver_arity_source": "literal",
        "resolver_n_registrations": 2,
    }
    r = check_resolver_arity(ctx)
    assert not r.passed and r.is_hard_fail
    assert "silently return kTfLiteError" in r.detail


def test_resolver_arity_oversized_is_called_out_as_wasted_slots():
    ctx = {
        "model": mk_model(ops={"ADD": 1}),
        "resolver_path": "/x/ops.cpp",
        "resolver_ops": {"ADD"},
        "resolver_arity": 9,
        "resolver_arity_source": "literal",
        "resolver_n_registrations": 1,
    }
    r = check_resolver_arity(ctx)
    assert not r.passed and "unused slot" in r.detail


def test_resolver_exact_skips_when_no_resolver_given():
    r = check_resolver_exact({"model": mk_model()})
    assert r.skipped and not r.is_hard_fail


# ==========================================================================
# G-LOAD-ISOLATED
# ==========================================================================


def test_load_isolated_gate_passes_on_clean_load():
    m = mk_model(inputs=(0,), outputs=(1,))
    ctx = {
        "model": m,
        "load_result": {"ok": True, "returncode": 0, "signal": None, "stderr": "",
                        "n_inputs": 1, "n_outputs": 1},
    }
    assert check_load_isolated(ctx).passed


def test_load_isolated_gate_fails_on_sigsegv():
    ctx = {
        "model": mk_model(),
        "load_result": {"ok": False, "returncode": -11, "signal": "SIGSEGV",
                        "stderr": "", "n_inputs": None, "n_outputs": None},
    }
    r = check_load_isolated(ctx)
    assert not r.passed and r.is_hard_fail
    assert "SIGSEGV" in r.detail and "not catchable in-process" in r.detail


def test_load_isolated_gate_flags_io_disagreement():
    ctx = {
        "model": mk_model(inputs=(0,), outputs=(1,)),
        "load_result": {"ok": True, "returncode": 0, "signal": None, "stderr": "",
                        "n_inputs": 7, "n_outputs": 1},
    }
    r = check_load_isolated(ctx)
    assert not r.passed and r.evidence["io_mismatch"]


def test_load_isolated_gate_skips_when_not_run():
    assert check_load_isolated({"model": mk_model()}).skipped


# ==========================================================================
# harness
# ==========================================================================


def test_run_gates_returns_one_result_per_gate_with_an_empty_ctx():
    results = run_gates({})
    assert len(results) == N_GATES
    assert {r.gate.id for r in results} == {
        "G-SINGLE-SUBGRAPH", "G-IO-SYMMETRY", "G-IO-LAYOUT",
        "G-SCALE-FINITE", "G-SCALE-POSITIVE", "G-QUANT-STRUCT", "G-SCALE-FLOOR",
        "G-OPSET", "G-ACCEL-COVERAGE",
        "G-RESOLVER-EXACT", "G-RESOLVER-LEAN", "G-RESOLVER-ARITY",
        "G-RESOLVER-FIRMWARE", "G-LOAD-ISOLATED",
    }
    assert all(r.skipped for r in results)
    # A skip is still not a hard *fail* -- but it is also not health, and the
    # report layer must say so.  See the INCOMPLETE tests below.
    assert not any_hard_fail(results)
    assert any_hard_skip(results)


def test_run_gates_converts_a_raising_gate_into_a_failure():
    def exploding(ctx):
        raise RuntimeError("boom")

    exploding.gate = Gate(id="G-BOOM", title="t", severity="hard-fail")
    (r,) = run_gates({}, gates=[exploding])
    assert not r.passed and "boom" in r.detail


def test_gate_ids_and_severities_are_the_documented_ones():
    """The severity split is part of the contract: it decides the exit code."""
    assert {g.id: g.severity for g in ALL_GATE_SPECS} == {
        "G-SINGLE-SUBGRAPH": "hard-fail",
        "G-IO-SYMMETRY": "hard-fail",
        "G-IO-LAYOUT": "warn",
        "G-SCALE-FINITE": "hard-fail",
        "G-SCALE-POSITIVE": "hard-fail",
        "G-QUANT-STRUCT": "hard-fail",
        "G-SCALE-FLOOR": "warn",
        "G-OPSET": "hard-fail",
        "G-ACCEL-COVERAGE": "warn",
        "G-RESOLVER-EXACT": "hard-fail",
        "G-RESOLVER-LEAN": "warn",
        "G-RESOLVER-ARITY": "hard-fail",
        "G-RESOLVER-FIRMWARE": "hard-fail",
        "G-LOAD-ISOLATED": "hard-fail",
    }
    assert len(ALL_GATE_SPECS) == N_GATES


def test_gate_dataclasses_are_frozen():
    g = Gate(id="G-X", title="t", severity="warn")
    with pytest.raises(Exception):
        g.id = "other"  # type: ignore[misc]


def test_report_renders_text_and_json():
    rep = AuditReport(
        model_path="/tmp/x.tflite",
        results=run_gates({"model": mk_model()}),
        notes=["a note"],
        summary={"ops": "3 nodes"},
    )
    text = render_text([rep])
    assert "x.tflite" in text and "a note" in text
    blob = json.loads(render_json([rep]))
    assert blob["n_models"] == 1
    assert len(blob["reports"][0]["gates"]) == N_GATES


def test_report_json_survives_sets_and_tuples_in_evidence():
    g = Gate(id="G-X", title="t", severity="warn")
    r = GateResult(gate=g, passed=True, detail="d", evidence={"s": {"b", "a"}, "t": (1, 2)})
    rep = AuditReport(model_path="/tmp/x.tflite", results=[r])
    blob = json.loads(render_json([rep]))
    assert blob["reports"][0]["gates"][0]["evidence"]["s"] == ["a", "b"]


# ==========================================================================
# GOLDEN: the resolver-header parse
# ==========================================================================


@needs_tree
def test_golden_resolver_method_counts():
    """Pinned counts. A drop here means the brace matcher regressed."""
    builtin, custom = ti.counts(TFLM_TREE)
    assert builtin == GOLDEN_BUILTIN_METHODS
    assert custom == GOLDEN_CUSTOM_METHODS
    assert len(ti.resolver_methods(TFLM_TREE)) == GOLDEN_TOTAL_METHODS
    assert len(ti.supported_builtins(TFLM_TREE)) == GOLDEN_BUILTIN_METHODS


@needs_tree
def test_golden_counts_reconcile_with_the_raw_header():
    """Every `TfLiteStatus Add` line is accounted for -- nothing silently dropped."""
    header = os.path.join(TFLM_TREE, ti.RESOLVER_HEADER)
    with open(header, encoding="utf-8") as fh:
        raw_lines = sum(1 for ln in fh if "TfLiteStatus Add" in ln)
    assert raw_lines == GOLDEN_TOTAL_METHODS + GOLDEN_MACHINERY_METHODS == 115


@needs_tree
def test_golden_multiline_methods_are_not_lost():
    """The exact regression the naive line-regex causes.

    These methods take a `const TFLMRegistration&` default argument, so their
    `AddBuiltin(...)` call is on a different line from `TfLiteStatus AddX(`.
    """
    methods = ti.resolver_methods(TFLM_TREE)
    assert methods["AddFullyConnected"] == "FULLY_CONNECTED"
    assert methods["AddTransposeConv"] == "TRANSPOSE_CONV"
    assert methods["AddConv2D"] == "CONV_2D"
    assert methods["AddDepthwiseConv2D"] == "DEPTHWISE_CONV_2D"

    header = os.path.join(TFLM_TREE, ti.RESOLVER_HEADER)
    with open(header, encoding="utf-8") as fh:
        text = fh.read()
    naive = set(
        re.findall(
            r"TfLiteStatus\s+Add\w+\([^)]*\)\s*\{\s*return\s+AddBuiltin\("
            r"BuiltinOperator_(\w+)",
            text,
        )
    )
    # The naive parse really does lose them; ours really does keep them.
    assert "FULLY_CONNECTED" not in naive
    assert "TRANSPOSE_CONV" not in naive
    assert len(naive) < GOLDEN_BUILTIN_METHODS
    assert {"FULLY_CONNECTED", "TRANSPOSE_CONV"} <= ti.supported_builtins(TFLM_TREE)


@needs_tree
def test_golden_addethosu_dynamic_custom_name_is_resolved():
    """AddEthosU registers via GetString_ETHOSU(), not a string literal."""
    assert ti.resolver_methods(TFLM_TREE)["AddEthosU"] == "CUSTOM:ethos-u"


@needs_tree
@pytest.mark.parametrize(
    "op", ["FULLY_CONNECTED", "TRANSPOSE_CONV", "GATHER_ND", "SELECT_V2",
           "DEQUANTIZE", "STRIDED_SLICE", "REDUCE_MAX", "CONV_2D"]
)
def test_golden_ops_tflm_does_support(op):
    assert op in ti.supported_builtins(TFLM_TREE)


@needs_tree
@pytest.mark.parametrize("op", ["POW", "REDUCE_ALL", "REDUCE_ANY"])
def test_golden_ops_tflm_does_not_support(op):
    """reduce.cc is REDUCE_MAX/MEAN/SUM only, and there is no POW kernel."""
    assert op not in ti.supported_builtins(TFLM_TREE)


@needs_tree
def test_golden_custom_signal_ops_present():
    custom = ti.supported_custom(TFLM_TREE)
    assert {"CUSTOM:SignalRfft", "CUSTOM:SignalFilterBank",
            "CUSTOM:SignalOverlapAdd"} <= custom


# ==========================================================================
# the C++ scanner itself
# ==========================================================================


def test_blank_comments_preserves_strings_and_line_numbers():
    src = 'a; // {{{\nb("http://x{"); /* }\n} */ c;\n'
    out = ti._blank_comments(src)
    assert out.count("\n") == src.count("\n")
    assert '"http://x{"' in out
    assert "{{{" not in out


def test_brace_matcher_ignores_braces_in_strings_and_comments():
    src = ti._blank_comments('f() { g("}"); /* } */ h(); }')
    start = src.index("{")
    end = ti._match_delims(src, start, "{", "}")
    assert src[end - 1] == "}" and end == len(src)


def test_brace_matcher_raises_on_unbalanced_input():
    with pytest.raises(ValueError):
        ti._match_delims("{ a b c", 0, "{", "}")


# ==========================================================================
# resolver .cpp parsing
# ==========================================================================


SYNTHETIC_CPP = """
// AddNotReal();  <- a comment, must be ignored
#include "x.h"
extern "C" tflite::MicroOpResolver &MODEL_GetOpsResolver(void)
{
    static tflite::MicroMutableOpResolver<3> s_resolver;
    s_resolver.AddConv2D();
    s_resolver.AddFullyConnected();
    s_resolver.AddCustom("SignalRfft", tflite::tflm_signal::Register_RFFT());
    return s_resolver;
}
"""


def test_parse_resolver_cpp_synthetic(tmp_path):
    p = tmp_path / "ops.cpp"
    p.write_text(SYNTHETIC_CPP)
    parsed = parse_resolver_cpp(str(p))
    assert parsed["arity"] == 3
    assert parsed["n_registrations"] == 3
    assert set(parsed["ops"]) == {"CONV_2D", "FULLY_CONNECTED", "CUSTOM:SignalRfft"}
    assert parsed["unmapped"] == []
    assert "AddNotReal" not in parsed["methods"]


def test_parse_resolver_cpp_detects_duplicates(tmp_path):
    p = tmp_path / "dup.cpp"
    p.write_text(
        "static tflite::MicroMutableOpResolver<2> r;\n"
        "r.AddAdd();\n"
        "r.AddAdd();\n"
    )
    parsed = parse_resolver_cpp(str(p))
    assert parsed["duplicates"] == ["ADD"]


def test_parse_resolver_cpp_works_without_a_tflm_tree(tmp_path):
    """The schema-enum fallback maps Add* names with no source tree at all."""
    p = tmp_path / "ops.cpp"
    p.write_text(SYNTHETIC_CPP)
    parsed = parse_resolver_cpp(str(p), method_map=None)
    assert parsed["used_fallback_map"] is True
    assert {"CONV_2D", "FULLY_CONNECTED"} <= set(parsed["ops"])


@pytest.mark.parametrize(
    "cpp,arity,n,ops",
    [
        ("lisennet_ops.cpp", 14, 14, {"ADD", "CONCATENATION", "CONV_2D",
                                      "DEPTHWISE_CONV_2D", "LOGISTIC", "MUL", "PAD",
                                      "QUANTIZE", "RESHAPE", "SLICE", "SUB", "TANH",
                                      "TRANSPOSE", "TRANSPOSE_CONV"}),
        ("convfsenet_ops.cpp", 10, 10, {"ADD", "CONCATENATION", "CONV_2D",
                                        "DEPTHWISE_CONV_2D", "GATHER_ND", "LOGISTIC",
                                        "QUANTIZE", "RESHAPE", "SLICE", "TRANSPOSE"}),
        ("nsnet2_ops.cpp", 10, 10, {"ADD", "CONCATENATION", "FULLY_CONNECTED",
                                    "LOGISTIC", "MUL", "QUANTIZE", "RESHAPE", "SLICE",
                                    "SUB", "TANH"}),
    ],
)
def test_golden_real_resolvers_parse(cpp, arity, n, ops):
    path = os.path.join(RESOLVER_DIR, cpp)
    if not os.path.isfile(path):
        pytest.skip("missing %s" % path)
    method_map = ti.resolver_methods(TFLM_TREE) if TFLM_TREE else None
    parsed = parse_resolver_cpp(path, method_map)
    assert parsed["arity"] == arity
    assert parsed["n_registrations"] == n
    assert set(parsed["ops"]) == ops
    assert parsed["unmapped"] == [] and parsed["duplicates"] == []


# ==========================================================================
# GOLDEN: the five artifacts
# ==========================================================================


@needs_models
def test_golden_lisennet_is_the_healthy_one():
    m = _model("relu6deep_streaming_int8.tflite")
    assert (len(m.inputs), len(m.outputs)) == (26, 26)  # 1 audio + 25 states
    assert m.n_ops == 320
    assert "TANH" not in m.op_histogram  # the AddTanh registration is dead code
    assert check_scale_finite({"model": m}).passed
    assert check_scale_floor({"model": m}).passed  # 0/26 floored


@needs_models
def test_golden_convfsenet_win_has_182_nonfinite_scale_tensors():
    m = _model("convfsenet_win_streaming.tflite")
    r = check_scale_finite({"model": m})
    assert not r.passed
    assert r.evidence["n_nonfinite_tensors"] == 182
    assert "POW" in m.op_histogram  # the power-law compression that caused it


@needs_models
def test_golden_convfsenet_win_fp32_sibling_is_clean():
    """Proves the damage is done by quantization, not by the graph."""
    path = os.path.join(HOST_OUT, "convfsenet_win_streaming.fp32.tflite")
    if not os.path.isfile(path):
        pytest.skip("no fp32 control")
    assert check_scale_finite({"model": load_model(path)}).passed


@needs_models
@pytest.mark.parametrize(
    "name,n_in,n_floored_in",
    [
        ("convfsenet_sliced_streaming.tflite", 10, 9),
        ("convfsenet_nocompress_streaming.tflite", 10, 9),
        ("nsnet2_plain_streaming.tflite", 2, 1),
        ("relu6deep_streaming_int8.tflite", 26, 0),
    ],
)
def test_golden_state_input_floor(name, n_in, n_floored_in):
    m = _model(name)
    assert len(m.inputs) == n_in
    r = check_scale_floor({"model": m})
    assert r.evidence["n_floored_inputs"] == n_floored_in
    if n_floored_in:
        assert r.evidence["floored_inputs"][0]["min_scale"] == GOLDEN_FLOOR_SCALE


@needs_models
@needs_tree
@pytest.mark.parametrize(
    "name,missing",
    [
        ("convfsenet_win_streaming.tflite", {"POW", "REDUCE_ALL"}),
        ("convfsenet_nocompress_streaming.tflite", {"REDUCE_ALL"}),
        ("convfsenet_sliced_streaming.tflite", set()),
        ("nsnet2_plain_streaming.tflite", set()),
        ("relu6deep_streaming_int8.tflite", set()),
    ],
)
def test_golden_opset_verdicts(name, missing):
    m = _model(name)
    ctx = {"model": m, "supported_ops": ti.supported_ops(TFLM_TREE), "tflm_tree": TFLM_TREE}
    r = check_opset(ctx)
    assert set(r.evidence["missing"]) == missing
    assert r.passed == (not missing)


@needs_models
@needs_tree
def test_golden_firmware_resolver_cannot_run_nsnet2():
    """The committed firmware resolver has no AddFullyConnected."""
    if not os.path.isfile(FIRMWARE_RESOLVER):
        pytest.skip("no firmware resolver")
    parsed = parse_resolver_cpp(FIRMWARE_RESOLVER, ti.resolver_methods(TFLM_TREE))
    m = _model("nsnet2_plain_streaming.tflite")
    r = check_resolver_exact(
        {
            "model": m,
            "resolver_path": FIRMWARE_RESOLVER,
            "resolver_ops": set(parsed["ops"]),
            "resolver_arity": parsed["arity"],
            "resolver_n_registrations": parsed["n_registrations"],
            "tflm_tree": TFLM_TREE,
        }
    )
    assert not r.passed
    assert r.evidence["missing"] == ["FULLY_CONNECTED"]
    assert "AddFullyConnected" in r.detail


@needs_models
@needs_tree
def test_golden_firmware_resolver_has_a_dead_tanh_for_lisennet():
    if not os.path.isfile(FIRMWARE_RESOLVER):
        pytest.skip("no firmware resolver")
    parsed = parse_resolver_cpp(FIRMWARE_RESOLVER, ti.resolver_methods(TFLM_TREE))
    m = _model("relu6deep_streaming_int8.tflite")
    r = check_resolver_exact(
        {
            "model": m,
            "resolver_path": FIRMWARE_RESOLVER,
            "resolver_ops": set(parsed["ops"]),
            "resolver_arity": parsed["arity"],
            "resolver_n_registrations": parsed["n_registrations"],
            "tflm_tree": TFLM_TREE,
        }
    )
    assert r.evidence["missing"] == [] and r.evidence["unused"] == ["TANH"]


# ==========================================================================
# subprocess isolation
# ==========================================================================


@needs_models
def test_golden_isolated_load_survives_the_segfaulting_model():
    """The whole point of the subprocess: this must return, not die."""
    path = os.path.join(HOST_OUT, "convfsenet_win_streaming.tflite")
    if not os.path.isfile(path):
        pytest.skip("missing artifact")
    res = load_isolated(path, timeout=180)
    assert res["ok"] is False
    assert res["returncode"] < 0
    assert res["signal"] == "SIGSEGV"
    # ...and we are still here to assert it.


@needs_models
def test_golden_isolated_load_succeeds_on_the_healthy_model():
    path = os.path.join(HOST_OUT, "relu6deep_streaming_int8.tflite")
    if not os.path.isfile(path):
        pytest.skip("missing artifact")
    res = load_isolated(path, timeout=180)
    assert res["ok"] and res["returncode"] == 0 and res["signal"] is None
    assert (res["n_inputs"], res["n_outputs"]) == (26, 26)


def test_isolated_load_on_a_nonexistent_file_is_reported_not_raised():
    res = load_isolated("/nope/nope.tflite")
    assert res["ok"] is False and "no such file" in res["stderr"]


def test_isolated_load_on_garbage_is_a_clean_failure(tmp_path):
    p = tmp_path / "junk.tflite"
    p.write_bytes(b"not a flatbuffer at all")
    res = load_isolated(str(p), timeout=120)
    assert res["ok"] is False


# ==========================================================================
# accel map degradation
# ==========================================================================


def test_accel_map_returns_empty_and_explains_when_the_lib_is_missing():
    got = accel_map.accelerated_builtins("/nope/libtflm.a", TFLM_TREE or "/nope", None)
    assert got == set()
    assert accel_map.LAST_STATUS["ok"] is False
    assert "not found" in accel_map.LAST_STATUS["reason"]


def test_accel_map_returns_empty_and_explains_when_the_tree_is_missing(tmp_path):
    fake_lib = tmp_path / "libtflm_rt500.a"
    fake_lib.write_bytes(b"!<arch>\n")
    got = accel_map.accelerated_builtins(str(fake_lib), str(tmp_path / "no-tree"), None)
    assert got == set()
    assert accel_map.LAST_STATUS["ok"] is False


def test_accel_map_never_raises_on_a_bogus_nm(monkeypatch, tmp_path):
    fake_lib = tmp_path / "libtflm_rt500.a"
    fake_lib.write_bytes(b"!<arch>\n")
    monkeypatch.setattr(accel_map, "find_nm", lambda: None)
    assert accel_map.accelerated_builtins(str(fake_lib), TFLM_TREE or ".", None) == set()
    assert "xt-nm" in accel_map.LAST_STATUS["reason"]


@needs_tree
def test_accel_map_cache_round_trips_and_invalidates(tmp_path):
    lib = _lib()
    if lib is None or accel_map.find_nm() is None:
        pytest.skip("no built lib / no xt-nm on this box")
    cache = tmp_path / "accel.json"
    first = accel_map.accelerated_builtins(lib, TFLM_TREE, str(cache))
    assert accel_map.LAST_STATUS.get("cached") is False
    second = accel_map.accelerated_builtins(lib, TFLM_TREE, str(cache))
    assert accel_map.LAST_STATUS.get("cached") is True
    assert first == second and first

    # A different TFLM tree must invalidate: the tree is what maps op -> kernel TU.
    blob = json.loads(cache.read_text())
    blob["tflm_tree"] = "/some/other/tree"
    cache.write_text(json.dumps(blob))
    accel_map.accelerated_builtins(lib, TFLM_TREE, str(cache))
    assert accel_map.LAST_STATUS.get("cached") is False


@needs_tree
def test_golden_accel_map_when_the_toolchain_is_present():
    lib = _lib()
    if lib is None or accel_map.find_nm() is None:
        pytest.skip("no built lib / no xt-nm on this box")
    acc = accel_map.accelerated_builtins(lib, TFLM_TREE, None)
    assert accel_map.LAST_STATUS["ok"] is True
    # The kernels whose accel lives in a differently-named TU: these are the
    # ones a "does <op>.o reference xa_nn_*" check gets wrong.
    assert {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"} <= acc
    # ...and the guard against over-crediting by filename prefix.
    assert "TRANSPOSE" in acc and "TRANSPOSE_CONV" in acc
    # Genuinely unaccelerated (pure memcpy / control-flow kernels).
    assert "RESHAPE" not in acc and "GATHER_ND" not in acc


def test_accel_map_prefix_guard_does_not_credit_a_sibling_registrar():
    hits = accel_map._candidate_objects(
        "transpose", {"transpose_conv.o"}, {"transpose_conv"}, set()
    )
    assert hits == []
    hits = accel_map._candidate_objects(
        "fully_connected", {"fully_connected_int8.o"}, {"transpose_conv"}, set()
    )
    assert hits == ["fully_connected_int8.o"]


# ==========================================================================
# flatbuffer parser edge cases
# ==========================================================================


def test_load_model_rejects_a_non_tflite_file(tmp_path):
    p = tmp_path / "x.tflite"
    p.write_bytes(b"\x00" * 64)
    with pytest.raises(ValueError):
        load_model(str(p))


def test_load_model_rejects_a_truncated_file(tmp_path):
    p = tmp_path / "x.tflite"
    p.write_bytes(b"ab")
    with pytest.raises(ValueError):
        load_model(str(p))


def test_load_model_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        load_model("/nope/nope.tflite")


@needs_models
def test_load_model_agrees_with_the_interpreter_on_io_counts():
    """Cross-check the hand-rolled parse against the reference implementation."""
    name = "relu6deep_streaming_int8.tflite"
    path = os.path.join(HOST_OUT, name)
    if not os.path.isfile(path):
        pytest.skip("missing artifact")
    m = load_model(path)
    res = load_isolated(path, timeout=180)
    if not res["ok"]:
        pytest.skip("interpreter could not load the control model")
    assert res["n_inputs"] == len(m.inputs)
    assert res["n_outputs"] == len(m.outputs)


@needs_models
def test_op_versions_are_captured():
    m = _model("relu6deep_streaming_int8.tflite")
    assert set(m.op_versions) == set(m.op_histogram)
    assert all(v >= 1 for v in m.op_versions.values())


# ==========================================================================
# CLI
# ==========================================================================


@needs_models
def test_cli_all_exits_1_and_writes_json(tmp_path, capsys):
    from lane.__main__ import main

    out = tmp_path / "report.json"
    rc = main(["audit", "--all", "--no-load", "--json", str(out)])
    captured = capsys.readouterr().out
    assert rc == 1  # at least one artifact hard-fails
    assert "SUMMARY" in captured
    blob = json.loads(out.read_text())
    assert blob["exit_code"] == 1
    assert blob["n_models"] >= 1
    assert "convfsenet_win_streaming.tflite" in blob["blocked"]


@needs_models
def test_cli_single_model_with_explicit_resolver(capsys):
    """`--no-load` leaves a hard-fail gate unrun, so the run is INCOMPLETE.

    This used to assert exit 0 / PASS-WITH-WARNINGS.  It was encoding the bug:
    G-LOAD-ISOLATED is a deploy-blocking gate, `--no-load` means nobody ran it,
    and a run that skipped a deploy-blocking gate is not a green run.
    """
    from lane.__main__ import main

    model = os.path.join(HOST_OUT, "convfsenet_sliced_streaming.tflite")
    resolver = os.path.join(RESOLVER_DIR, "convfsenet_ops.cpp")
    firmware = os.path.join(RESOLVER_DIR, "convfsenet_ops.cpp")
    if not (os.path.isfile(model) and os.path.isfile(resolver)):
        pytest.skip("missing inputs")
    argv = ["audit", model, "--resolver", resolver,
            "--firmware-resolver", firmware, "--no-load"]
    rc = main(argv)
    out = capsys.readouterr().out
    assert rc == EXIT_INCOMPLETE
    assert "G-RESOLVER-EXACT" in out and "INCOMPLETE" in out
    assert "G-LOAD-ISOLATED" in out

    # ...and the escape hatch changes only the number, never the word.
    rc = main(argv + ["--allow-incomplete"])
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    assert "verdict: INCOMPLETE" in out


@needs_models
def test_cli_infers_the_right_resolver_per_family(capsys):
    from lane.__main__ import infer_family

    for name, expect in [
        ("relu6deep_streaming_int8.tflite", "lisennet_ops.cpp"),
        ("convfsenet_sliced_streaming.tflite", "convfsenet_ops.cpp"),
        ("nsnet2_plain_streaming.tflite", "nsnet2_ops.cpp"),
    ]:
        path = os.path.join(HOST_OUT, name)
        if not os.path.isfile(path):
            continue
        family, resolver, why = infer_family(load_model(path), path)
        assert resolver and os.path.basename(resolver) == expect, why


def test_cli_says_so_when_it_cannot_infer_a_resolver():
    from lane.__main__ import infer_family

    m = mk_model(inputs=(0, 1, 2), outputs=(3,))
    family, resolver, why = infer_family(m, "/tmp/mystery_model.tflite")
    assert family is None and resolver is None
    assert "cannot infer" in why and "--resolver" in why


def test_cli_rejects_all_plus_a_positional(capsys):
    from lane.__main__ import main

    assert main(["audit", "--all", "x.tflite"]) == 2


def test_cli_rejects_no_target(capsys):
    from lane.__main__ import main

    assert main(["audit"]) == 2


def test_cli_reports_a_missing_model_cleanly(capsys):
    from lane.__main__ import main

    assert main(["audit", "/nope/nope.tflite"]) == 2
    assert "no such model" in capsys.readouterr().err


# ==========================================================================
# hygiene
# ==========================================================================


def test_lane_never_imports_torch():
    """Contract: the auditor is flatbuffer-only and must stay torch-free."""
    import lane
    import lane.__main__  # noqa: F401
    import lane.gates
    import lane.report
    import lane.runtime.accel_map
    import lane.runtime.flatbuffer
    import lane.runtime.litert_host
    import lane.runtime.tflm_introspect

    assert "torch" not in sys.modules

    root = os.path.dirname(os.path.abspath(lane.__file__))
    for dirpath, _dirs, files in os.walk(root):
        if "__pycache__" in dirpath or dirpath.endswith("tests"):
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            with open(os.path.join(dirpath, fn), encoding="utf-8") as fh:
                text = fh.read()
            assert not re.search(r"^\s*import\s+torch\b", text, re.M), fn
            assert not re.search(r"^\s*from\s+torch\b", text, re.M), fn


# ==========================================================================
# REPORTED-TEXT / EVIDENCE CONSISTENCY
#
# Both of these were live defects: the human-readable detail line disagreed
# with the machine-readable evidence it was summarising, which is exactly the
# failure mode an auditor cannot afford.
# ==========================================================================


def test_accel_detail_accounts_for_every_portable_c_node():
    """The 'portable C:' list must reconcile with n_accelerated_ops.

    Regression: the list was truncated to the top 5 with no remainder marker,
    so convfsenet_win (6 portable-C ops) rendered 79 of its 80 portable nodes
    and the text silently contradicted the count.
    """
    ops = {
        "CONV_2D": 20, "RESHAPE": 59, "GATHER_ND": 9, "SELECT_V2": 9,
        "DEQUANTIZE": 1, "POW": 1, "REDUCE_ALL": 1,
    }
    r = check_accel_coverage(
        {"model": mk_model(ops=ops), "accelerated": {"CONV_2D"},
         "accel_status": {"ok": True}, "accel_min_fraction": 0.0}
    )
    total = sum(ops.values())
    assert r.evidence["n_ops"] == total
    assert r.evidence["n_accelerated_ops"] == 20
    n_portable = total - 20
    assert sum(r.evidence["reference_c"].values()) == n_portable

    # Every portable node is accounted for in the prose: the ops named
    # explicitly plus the remainder the "(+N more op(s), M node(s))" tail
    # declares must add back up to n_portable.
    named = sum(
        int(m.group(2))
        for m in re.finditer(r"([A-Z_0-9]+) x(\d+)", r.detail.split("portable C:")[1])
    )
    tail = re.search(r"\(\+(\d+) more op\(s\), (\d+) node\(s\)\)", r.detail)
    remainder = int(tail.group(2)) if tail else 0
    assert named + remainder == n_portable
    n_named_ops = len(re.findall(r"[A-Z_0-9]+ x\d+", r.detail.split("portable C:")[1]))
    n_more = int(tail.group(1)) if tail else 0
    assert n_named_ops + n_more == len(r.evidence["reference_c"])


def test_accel_detail_has_no_remainder_tail_when_nothing_is_hidden():
    r = check_accel_coverage(
        {"model": mk_model(ops={"CONV_2D": 10, "RESHAPE": 4}),
         "accelerated": {"CONV_2D"}, "accel_status": {"ok": True},
         "accel_min_fraction": 0.0}
    )
    assert "portable C: RESHAPE x4" in r.detail
    assert "more op(s)" not in r.detail


def test_floored_input_detail_does_not_read_as_a_fraction_of_the_tensor_count():
    """`9/10 of them` after `110 tensor(s)` reads as 9/10ths of the 110.

    The floored-input denominator is the model's INPUT count, not the count of
    collapsed tensors, so the prose has to name the population it divides.
    """
    floor = GOLDEN_FLOOR_SCALE
    tensors = [
        mk_tensor(0, "state_a", scales=(floor,)),
        mk_tensor(1, "state_b", scales=(floor,)),
        mk_tensor(2, "spectrum", scales=(0.011,)),
        # a floored intermediate, so len(floored) != len(floored_inputs)
        mk_tensor(3, "mid", scales=(floor,)),
        mk_tensor(4, "mid2", scales=(floor,)),
    ]
    m = mk_model(tensors=tensors, inputs=(0, 1, 2), outputs=(2,),
                 ops={"CONV_2D": 1})
    r = check_scale_floor({"model": m})
    assert not r.passed
    assert r.evidence["n_floored"] == 4
    assert r.evidence["n_floored_inputs"] == 2
    assert r.evidence["n_inputs"] == 3
    # The bare "2/3 of them" phrasing must not come back.
    assert "2/3 of them" not in r.detail
    assert "of the model's 3 INPUTS" in r.detail
    # And the two populations must both still be legible.
    assert "4 tensor(s) have a collapsed range" in r.detail


@needs_models
def test_golden_convfsenet_win_portable_c_nodes_reconcile():
    """The real artifact that exposed the truncation bug."""
    m = _model("convfsenet_win_streaming.tflite")
    accel = {"ADD", "CONCATENATION", "CONV_2D", "DEPTHWISE_CONV_2D", "LOGISTIC",
             "QUANTIZE", "SLICE", "TRANSPOSE"}
    r = check_accel_coverage(
        {"model": m, "accelerated": accel, "accel_status": {"ok": True},
         "accel_min_fraction": 0.0}
    )
    assert r.evidence["n_ops"] == 261
    assert r.evidence["n_accelerated_ops"] == 181
    assert sum(r.evidence["reference_c"].values()) == 80
    assert len(r.evidence["reference_c"]) == 6
    # 6 portable ops, 5 named inline -> the remainder must be declared.
    assert "(+1 more op(s), 1 node(s))" in r.detail


# ==========================================================================
# REGRESSION: a gate that could not run must never read as health
#
# The original defect: `lane/gates/__init__.py` encoded SKIP honestly
# (`passed=True` + an evidence flag) and the reporting layer counted it with
# the passes.  `AuditReport.verdict` looked only at hard_fails and warns, so
# "we checked nothing" and "we checked everything and it's fine" both printed
# CLEAR; render_json emitted `blocked: []` / `exit_code: 0`; and two ordinary
# CLI typos were enough to get there.
# ==========================================================================


def _skipping_gate(gate_id, severity):
    from lane.gates import skip as _skip

    g = Gate(id=gate_id, title="t", severity=severity)

    def fn(ctx):
        return _skip(g, "tooling missing on this box")

    fn.gate = g
    return fn


def _passing_gate(gate_id, severity="hard-fail"):
    g = Gate(id=gate_id, title="t", severity=severity)

    def fn(ctx):
        return GateResult(gate=g, passed=True, detail="fine", evidence={})

    fn.gate = g
    return fn


def _failing_gate(gate_id, severity="hard-fail"):
    g = Gate(id=gate_id, title="t", severity=severity)

    def fn(ctx):
        return GateResult(gate=g, passed=False, detail="broken", evidence={})

    fn.gate = g
    return fn


def test_a_skipped_hard_gate_is_never_reported_as_CLEAR():
    """The headline fix: a hard-fail gate that could not run => INCOMPLETE."""
    results = run_gates(
        {}, gates=[_passing_gate("G-OK"), _skipping_gate("G-GONE", "hard-fail")]
    )
    rep = AuditReport(model_path="/tmp/x.tflite", results=results)
    assert rep.verdict == "INCOMPLETE"
    assert rep.complete is False
    assert [r.gate.id for r in rep.hard_skips] == ["G-GONE"]
    # ...and it is still not a hard *fail*: the two states stay distinct.
    assert not rep.hard_fails


def test_clear_requires_every_hard_gate_to_have_actually_run():
    rep = AuditReport(
        model_path="/tmp/x.tflite",
        results=run_gates({}, gates=[_passing_gate("G-A"), _passing_gate("G-B")]),
    )
    assert rep.verdict == "CLEAR" and rep.complete is True


def test_a_skipped_warn_gate_does_not_make_the_audit_incomplete():
    """Only *deploy-blocking* gates can make a run incomplete."""
    rep = AuditReport(
        model_path="/tmp/x.tflite",
        results=run_gates(
            {}, gates=[_passing_gate("G-A"), _skipping_gate("G-W", "warn")]
        ),
    )
    assert rep.verdict == "CLEAR" and rep.complete is True


def test_a_real_hard_fail_still_outranks_an_incomplete():
    rep = AuditReport(
        model_path="/tmp/x.tflite",
        results=run_gates(
            {}, gates=[_failing_gate("G-BAD"), _skipping_gate("G-GONE", "hard-fail")]
        ),
    )
    assert rep.verdict == "BLOCKED"
    assert exit_code([rep]) == EXIT_BLOCKED


def test_exit_code_is_3_for_incomplete_and_0_only_when_complete():
    incomplete = AuditReport(
        model_path="/tmp/a.tflite",
        results=run_gates({}, gates=[_skipping_gate("G-GONE", "hard-fail")]),
    )
    clean = AuditReport(
        model_path="/tmp/b.tflite", results=run_gates({}, gates=[_passing_gate("G-A")])
    )
    assert exit_code([incomplete]) == EXIT_INCOMPLETE
    assert exit_code([incomplete], allow_incomplete=True) == EXIT_OK
    assert exit_code([clean]) == EXIT_OK
    # One incomplete model in a sweep is enough to stop the sweep being green.
    assert exit_code([clean, incomplete]) == EXIT_INCOMPLETE


def test_render_json_surfaces_the_gates_that_could_not_run():
    """A CI job reads `blocked` and `exit_code`; both used to say 'all good'."""
    rep = AuditReport(
        model_path="/tmp/x.tflite",
        results=run_gates(
            {},
            gates=[
                _passing_gate("G-OK"),
                _skipping_gate("G-OPSET-ISH", "hard-fail"),
                _skipping_gate("G-WARNY", "warn"),
            ],
        ),
    )
    blob = json.loads(render_json([rep]))
    assert blob["blocked"] == []
    assert blob["incomplete"] == ["x.tflite"]
    assert blob["complete"] is False
    assert blob["exit_code"] == EXIT_INCOMPLETE
    assert blob["n_hard_skips"] == 1
    assert blob["skipped_hard_gates"] == ["G-OPSET-ISH"]
    assert blob["reports"][0]["skipped_hard_gates"][0]["id"] == "G-OPSET-ISH"


def test_render_text_names_the_gates_that_were_not_checked():
    rep = AuditReport(
        model_path="/tmp/x.tflite",
        results=run_gates({}, gates=[_skipping_gate("G-GONE", "hard-fail")]),
    )
    text = render_text([rep], width=200)
    assert "NOT CHECKED" in text and "G-GONE" in text
    assert "verdict: INCOMPLETE" in text
    assert "verdict: CLEAR" not in text


@needs_models
def test_cli_rejects_a_mistyped_resolver_path_instead_of_skipping(capsys):
    """`--resolver nsnet2_opps.cpp` used to SKIP the gate and exit 0."""
    from lane.__main__ import main

    model = os.path.join(HOST_OUT, "nsnet2_plain_streaming.tflite")
    if not os.path.isfile(model):
        pytest.skip("missing artifact")
    rc = main(["audit", model, "--resolver",
               os.path.join(RESOLVER_DIR, "nsnet2_opps.cpp"), "--no-load"])
    assert rc == 2
    assert "not a readable file" in capsys.readouterr().err


@needs_models
def test_cli_rejects_a_resolver_that_is_a_directory(capsys):
    from lane.__main__ import main

    model = os.path.join(HOST_OUT, "nsnet2_plain_streaming.tflite")
    if not os.path.isfile(model):
        pytest.skip("missing artifact")
    rc = main(["audit", model, "--resolver", RESOLVER_DIR, "--no-load"])
    assert rc == 2
    assert "not a readable file" in capsys.readouterr().err


@needs_models
def test_cli_rejects_an_unreadable_resolver(tmp_path, capsys):
    from lane.__main__ import main

    model = os.path.join(HOST_OUT, "nsnet2_plain_streaming.tflite")
    if not os.path.isfile(model):
        pytest.skip("missing artifact")
    blocked = tmp_path / "ops.cpp"
    blocked.write_text("static tflite::MicroMutableOpResolver<1> r;\nr.AddAdd();\n")
    os.chmod(str(blocked), 0o000)
    if os.access(str(blocked), os.R_OK):  # running as root
        pytest.skip("cannot make a file unreadable here")
    try:
        rc = main(["audit", model, "--resolver", str(blocked), "--no-load"])
    finally:
        os.chmod(str(blocked), 0o600)
    assert rc == 2
    assert "not readable" in capsys.readouterr().err


@needs_models
def test_cli_rejects_a_tflm_tree_that_is_not_a_tflm_tree(tmp_path, capsys):
    from lane.__main__ import main

    model = os.path.join(HOST_OUT, "nsnet2_plain_streaming.tflite")
    if not os.path.isfile(model):
        pytest.skip("missing artifact")
    rc = main(["audit", model, "--tflm-tree", str(tmp_path), "--no-load"])
    assert rc == 2
    assert "not a tflite-micro checkout" in capsys.readouterr().err


# ==========================================================================
# REGRESSION: G-SCALE-POSITIVE (negative and zero scales)
# ==========================================================================


def test_negative_scale_is_a_hard_fail_not_a_pass():
    """-0.0103 is finite and abs()-large, so both old quant gates saw nothing."""
    m = mk_model(tensors=[mk_tensor(0, "x", scales=(-0.010341591,))], inputs=(0,))
    assert check_scale_finite({"model": m}).passed  # still finite
    assert check_scale_floor({"model": m}).passed  # still not small
    r = check_scale_positive({"model": m})
    assert not r.passed and r.is_hard_fail
    assert r.evidence["n_negative_tensors"] == 1
    assert "NEGATIVE" in r.detail and "no sign guard" in r.detail


def test_negative_scale_on_one_per_axis_channel_is_caught():
    """The per-channel path in kernel_util.cc has no sign guard at all."""
    t = mk_tensor(0, "filt", shape=(6, 1, 1, 3), qdim=0,
                  scales=(0.01, -0.01, 0.02, 0.03, 0.04, 0.05),
                  zps=(0, 0, 0, 0, 0, 0))
    r = check_scale_positive({"model": mk_model(tensors=[t], inputs=(), outputs=())})
    assert not r.passed
    assert r.evidence["negative"][0]["bad_channels"] == [1]
    assert r.evidence["negative"][0]["per_axis"] is True


def test_zero_scale_is_a_hard_fail_and_not_called_likely_benign():
    """A zero scale is a divide-by-zero, not a collapsed range."""
    m = mk_model(tensors=[mk_tensor(0, "in", scales=(0.02,)),
                          mk_tensor(1, "conv_out", scales=(0.0,))],
                 inputs=(0,), outputs=(1,))
    r = check_scale_positive({"model": m})
    assert not r.passed and r.is_hard_fail
    assert r.evidence["n_zero_tensors"] == 1
    assert "ZERO" in r.detail and "divide-by-zero" in r.detail

    # ...and the floor gate must no longer claim it, nor excuse it.
    floor = check_scale_floor({"model": m})
    assert floor.passed
    assert "likely benign" not in floor.detail


def test_scale_floor_is_strictly_greater_than_zero():
    """`0 < s <= threshold`: zero belongs to G-SCALE-POSITIVE, not here."""
    m = mk_model(
        tensors=[mk_tensor(0, "zero", scales=(0.0,)),
                 mk_tensor(1, "tiny", scales=(GOLDEN_FLOOR_SCALE,))],
        inputs=(0,), outputs=(1,),
    )
    r = check_scale_floor({"model": m})
    assert not r.passed
    assert r.evidence["n_floored"] == 1
    assert r.evidence["floored_tensors"][0]["name"] == "tiny"
    assert 0.0 not in r.evidence["distinct_floor_values"]


def test_negative_tiny_scale_is_not_laundered_into_a_floor_warning():
    """abs() made -1e-13 look like an ordinary calibrator floor."""
    m = mk_model(tensors=[mk_tensor(0, "x", scales=(-1e-13,))], inputs=(0,))
    assert check_scale_floor({"model": m}).passed
    assert not check_scale_positive({"model": m}).passed


def test_min_positive_scale_and_has_nonpositive_scale_read_the_raw_tuple():
    t = mk_tensor(0, "t", scales=(-0.5, 0.25, float("nan")))
    assert t.has_nonpositive_scale() is True
    assert t.min_abs_scale() == 0.25  # unchanged: the floor gate's old view
    assert t.min_positive_scale() == 0.25
    assert t.nonpositive_scales() == (-0.5,)
    assert mk_tensor(1, "u", scales=(0.5,)).has_nonpositive_scale() is False


# ==========================================================================
# REGRESSION: G-QUANT-STRUCT (structural validity of the quant block)
# ==========================================================================


def test_short_per_axis_scale_vector_is_a_hard_fail():
    """12 channels, 9 scales: every per-channel kernel walks off the end."""
    t = mk_tensor(0, "w", shape=(12, 1, 1, 3), qdim=0,
                  scales=tuple(0.01 * (i + 1) for i in range(9)),
                  zps=tuple(0 for _ in range(9)))
    r = check_quant_struct({"model": mk_model(tensors=[t], inputs=(), outputs=())})
    assert not r.passed and r.is_hard_fail
    assert r.evidence["n_axis_length_mismatch"] == 1
    assert r.evidence["axis_length_mismatch"][0]["axis_len"] == 12
    assert r.evidence["axis_length_mismatch"][0]["n_scales"] == 9


def test_per_axis_vector_matching_a_non_zero_quantized_dimension_is_fine():
    t = mk_tensor(0, "w", shape=(1, 1, 3, 4), qdim=3,
                  scales=(0.1, 0.2, 0.3, 0.4), zps=(0, 0, 0, 0))
    assert check_quant_struct({"model": mk_model(tensors=[t], inputs=(), outputs=())}).passed


def test_zero_point_outside_the_dtype_range_is_a_hard_fail():
    t = mk_tensor(0, "x", dtype="INT8", scales=(0.01,), zps=(4242,))
    r = check_quant_struct({"model": mk_model(tensors=[t], inputs=(0,), outputs=(0,))})
    assert not r.passed and r.is_hard_fail
    assert r.evidence["zero_point_out_of_range"][0]["offenders"] == [4242]
    assert r.evidence["zero_point_out_of_range"][0]["allowed"] == [-128, 127]


def test_scale_and_zero_point_vector_lengths_must_agree():
    t = mk_tensor(0, "w", shape=(3,), qdim=0, scales=(0.1, 0.2, 0.3), zps=(0, 0))
    r = check_quant_struct({"model": mk_model(tensors=[t], inputs=(), outputs=())})
    assert not r.passed
    assert r.evidence["n_scale_zeropoint_mismatch"] == 1


def test_quant_struct_passes_on_an_ordinary_model():
    r = check_quant_struct({"model": mk_model()})
    assert r.passed and not r.skipped


# ==========================================================================
# REGRESSION: G-IO-SYMMETRY and G-SINGLE-SUBGRAPH
# ==========================================================================


def test_io_asymmetry_is_a_hard_fail():
    """A streaming export that loses one state output audited fully clean."""
    m = mk_model(inputs=(0, 1, 2), outputs=(0, 1))
    r = check_io_symmetry({"model": m})
    assert not r.passed and r.is_hard_fail
    assert "3 input(s) but 2 output(s)" in r.detail
    assert "MODEL_STATE_MAP" in r.detail


def test_io_symmetry_passes_and_reports_the_state_port_count():
    r = check_io_symmetry({"model": mk_model(inputs=(0, 1), outputs=(0, 1))})
    assert r.passed and "1 state port(s)" in r.detail


def test_io_symmetry_uses_the_family_io_hint_it_already_had():
    """`_FAMILIES` carried 26/10/2 and used it only to guess a resolver."""
    r = check_io_symmetry(
        {"model": mk_model(inputs=(0, 1), outputs=(0, 1)),
         "family": "LiSenNet", "family_expected_io": 26}
    )
    assert r.passed  # symmetric, so not a blocker...
    assert "expected to have 26 IO" in r.detail  # ...but no longer silent


def test_multi_subgraph_model_is_blocked_not_merely_noted():
    r = check_single_subgraph({"model": mk_model(n_subgraphs=2)})
    assert not r.passed and r.is_hard_fail
    assert "SUBGRAPH 0" in r.detail
    assert "quantization" in r.detail and "accel coverage" in r.detail


def test_single_subgraph_model_passes():
    assert check_single_subgraph({"model": mk_model()}).passed


@pytest.mark.parametrize(
    "fn,needle",
    [
        (check_scale_finite, "in subgraph 0"),
        (check_scale_positive, "in subgraph 0"),
        (check_quant_struct, "in subgraph 0"),
    ],
)
def test_quant_gate_pass_text_is_scoped_to_the_subgraph_it_read(fn, needle):
    """No whole-model claims from a subgraph-0-only read."""
    r = fn({"model": mk_model()})
    assert r.passed and needle in r.detail


def test_opset_pass_text_is_scoped_to_the_subgraph_it_read():
    r = check_opset(
        {"model": mk_model(ops={"ADD": 1}), "supported_ops": {"ADD"},
         "tflm_tree": "/t"}
    )
    assert r.passed and "in subgraph 0" in r.detail


# ==========================================================================
# REGRESSION: G-IO-LAYOUT (the generated firmware IO contract)
# ==========================================================================


IO_LAYOUT_H = """\
/* AUTO-GENERATED by deploy/rt595/host/gen_io_layout.py - DO NOT EDIT. */
/* source: /somewhere/lisennet_conv_hardened_streaming_fp32.tflite */
#ifndef MODEL_IO_LAYOUT_H
#define MODEL_IO_LAYOUT_H
#define MODEL_N_IO            (18)
#define MODEL_N_STATES        (17)
#define MODEL_FEATURE_IN_POS  (5)
#define MODEL_FEATURE_IS_INT8 (0)
#define MODEL_MASK_OUT_POS    (10)
#endif
"""


def test_parse_io_layout_reads_the_generated_header(tmp_path):
    p = tmp_path / "model_io_layout.h"
    p.write_text(IO_LAYOUT_H)
    parsed = parse_io_layout(str(p))
    assert parsed["defines"]["MODEL_N_IO"] == 18
    assert parsed["defines"]["MODEL_N_STATES"] == 17
    assert parsed["source"].endswith("lisennet_conv_hardened_streaming_fp32.tflite")


def test_io_layout_gate_flags_a_stale_header(tmp_path):
    """The committed header says 18 IO fp32; the blessed artifact is 26 int8."""
    p = tmp_path / "model_io_layout.h"
    p.write_text(IO_LAYOUT_H)
    tensors = [mk_tensor(i, "p%d" % i) for i in range(26)]
    m = mk_model(tensors=tensors, inputs=tuple(range(26)), outputs=tuple(range(26)))
    r = check_io_layout({"model": m, "io_layout": parse_io_layout(str(p))})
    assert not r.passed and not r.is_hard_fail  # warn: the header names one model
    assert "MODEL_N_IO is 18 but the model has 26 input(s)" in r.detail
    assert "MODEL_FEATURE_IS_INT8 is 0" in r.detail
    assert "lisennet_conv_hardened_streaming_fp32.tflite" in r.detail


def test_io_layout_gate_passes_when_the_header_matches(tmp_path):
    p = tmp_path / "model_io_layout.h"
    p.write_text(
        IO_LAYOUT_H.replace("(18)", "(2)").replace("(17)", "(1)")
        .replace("MODEL_FEATURE_IN_POS  (5)", "MODEL_FEATURE_IN_POS  (0)")
        .replace("MODEL_FEATURE_IS_INT8 (0)", "MODEL_FEATURE_IS_INT8 (1)")
        .replace("MODEL_MASK_OUT_POS    (10)", "MODEL_MASK_OUT_POS    (1)")
    )
    m = mk_model(tensors=[mk_tensor(0, "feat"), mk_tensor(1, "state")],
                 inputs=(0, 1), outputs=(0, 1))
    assert check_io_layout({"model": m, "io_layout": parse_io_layout(str(p))}).passed


def test_io_layout_gate_skips_visibly_when_no_header_is_available():
    r = check_io_layout({"model": mk_model()})
    assert r.skipped and not r.is_hard_skip  # warn severity: not a blocker


# ==========================================================================
# REGRESSION: resolver reachability, the preprocessor, and symbolic arity
# ==========================================================================


CLEAN_CPP = """\
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

extern "C" tflite::MicroOpResolver &MODEL_GetOpsResolver(void)
{
    static tflite::MicroMutableOpResolver<3> s_resolver;
    s_resolver.AddConv2D();
    s_resolver.AddSlice();
    s_resolver.AddAdd();
    return s_resolver;
}
"""


def _parse(tmp_path, text, name="ops.cpp"):
    p = tmp_path / name
    p.write_text(text)
    return parse_resolver_cpp(str(p))


def test_add_call_inside_if_zero_is_not_counted(tmp_path):
    text = CLEAN_CPP.replace(
        "    s_resolver.AddSlice();", "#if 0\n    s_resolver.AddSlice();\n#endif"
    )
    parsed = _parse(tmp_path, text)
    assert "SLICE" not in parsed["ops"]
    assert parsed["n_registrations"] == 2
    assert parsed["n_if_zero_blocks_stripped"] == 1


def test_add_call_behind_an_undefined_ifdef_is_reported_not_counted(tmp_path):
    text = CLEAN_CPP.replace(
        "    s_resolver.AddSlice();",
        "#ifdef CONFIG_SE_ENABLE_SLICE\n    s_resolver.AddSlice();\n#endif",
    )
    parsed = _parse(tmp_path, text)
    assert "SLICE" not in parsed["ops"]
    assert [c["op"] for c in parsed["conditional"]] == ["SLICE"]
    assert "CONFIG_SE_ENABLE_SLICE" in parsed["conditional"][0]["guard"]


def test_conditional_registration_makes_the_gate_fail_and_say_why(tmp_path):
    text = CLEAN_CPP.replace(
        "    s_resolver.AddSlice();",
        "#ifdef CONFIG_SE_ENABLE_SLICE\n    s_resolver.AddSlice();\n#endif",
    )
    parsed = _parse(tmp_path, text)
    ctx = {
        "model": mk_model(ops={"CONV_2D": 1, "SLICE": 39, "ADD": 1}),
        "resolver_path": parsed["path"],
        "resolver_ops": set(parsed["ops"]),
        "resolver_n_registrations": parsed["n_registrations"],
        "resolver_conditional": parsed["conditional"],
        "resolver_unreachable": parsed["unreachable"],
    }
    r = check_resolver_exact(ctx)
    assert not r.passed and r.is_hard_fail
    assert r.evidence["missing"] == ["SLICE"]
    assert "CONFIG_SE_ENABLE_SLICE" in r.detail
    assert "never emits that registration" in r.detail
    # The headline false claim must be gone.
    assert "registers exactly" not in r.detail


def test_add_call_in_an_orphaned_helper_is_not_counted(tmp_path):
    """A registration nothing calls is not a registration."""
    text = CLEAN_CPP.replace("    s_resolver.AddSlice();\n", "").replace(
        'extern "C" tflite::MicroOpResolver &MODEL_GetOpsResolver(void)',
        "static void install_extra(tflite::MicroOpResolver &base)\n"
        "{ auto &r = static_cast<tflite::MicroMutableOpResolver<3> &>(base);"
        " r.AddSlice(); }\n\n"
        'extern "C" tflite::MicroOpResolver &MODEL_GetOpsResolver(void)',
    )
    parsed = _parse(tmp_path, text)
    assert "SLICE" not in parsed["ops"]
    assert [c["op"] for c in parsed["unreachable"]] == ["SLICE"]
    # The declaration anchor must be the real object, not the static_cast.
    assert parsed["receiver"] == "s_resolver"
    assert parsed["body_found"] is True


def test_unreachable_registration_makes_the_gate_fail_and_say_why(tmp_path):
    text = CLEAN_CPP.replace("    s_resolver.AddSlice();\n", "").replace(
        'extern "C" tflite::MicroOpResolver &MODEL_GetOpsResolver(void)',
        "static void install_extra(tflite::MicroOpResolver &base)\n"
        "{ auto &r = static_cast<tflite::MicroMutableOpResolver<3> &>(base);"
        " r.AddSlice(); }\n\n"
        'extern "C" tflite::MicroOpResolver &MODEL_GetOpsResolver(void)',
    )
    parsed = _parse(tmp_path, text)
    r = check_resolver_exact(
        {
            "model": mk_model(ops={"CONV_2D": 1, "SLICE": 39, "ADD": 1}),
            "resolver_path": parsed["path"],
            "resolver_ops": set(parsed["ops"]),
            "resolver_n_registrations": parsed["n_registrations"],
            "resolver_unreachable": parsed["unreachable"],
        }
    )
    assert not r.passed
    assert r.evidence["missing"] == ["SLICE"]
    assert "outside the body of the function" in r.detail


def test_file_scope_resolver_still_counts_every_call(tmp_path):
    """No enclosing function => nothing to be outside of; do not invent a find."""
    parsed = _parse(
        tmp_path,
        "static tflite::MicroMutableOpResolver<2> r;\nr.AddAdd();\nr.AddMul();\n",
    )
    assert parsed["body_found"] is False
    assert parsed["unreachable"] == []
    assert set(parsed["ops"]) == {"ADD", "MUL"}


def test_include_guard_is_not_mistaken_for_a_conditional_registration(tmp_path):
    text = "#ifndef OPS_H\n#define OPS_H\n" + CLEAN_CPP + "#endif\n"
    parsed = _parse(tmp_path, text)
    assert parsed["conditional"] == []
    assert set(parsed["ops"]) == {"CONV_2D", "SLICE", "ADD"}


def test_symbolic_arity_is_resolved_from_the_same_translation_unit(tmp_path):
    """`constexpr int kNumOps = N;` is the style TFLM's own examples use."""
    text = CLEAN_CPP.replace(
        "    static tflite::MicroMutableOpResolver<3> s_resolver;",
        "    constexpr int kNumOps = 3;\n"
        "    static tflite::MicroMutableOpResolver<kNumOps> s_resolver;",
    )
    parsed = _parse(tmp_path, text)
    assert parsed["arity"] == 3
    assert parsed["arity_source"] == "resolved-constant"
    assert parsed["arity_expr"] == "kNumOps"


def test_symbolic_arity_from_a_define_is_resolved(tmp_path):
    text = "#define K_NUM_OPS 3\n" + CLEAN_CPP.replace(
        "MicroMutableOpResolver<3>", "MicroMutableOpResolver<K_NUM_OPS>"
    )
    parsed = _parse(tmp_path, text)
    assert parsed["arity"] == 3 and parsed["arity_source"] == "resolved-constant"


def test_a_wrong_symbolic_arity_is_caught_like_a_literal_one(tmp_path):
    """The exact adversarial case: kNumOps = 3, three more Add* calls follow."""
    text = CLEAN_CPP.replace(
        "    static tflite::MicroMutableOpResolver<3> s_resolver;",
        "    constexpr int kNumOps = 1;\n"
        "    static tflite::MicroMutableOpResolver<kNumOps> s_resolver;",
    )
    parsed = _parse(tmp_path, text)
    r = check_resolver_arity(
        {
            "resolver_path": parsed["path"],
            "resolver_ops": set(parsed["ops"]),
            "resolver_arity": parsed["arity"],
            "resolver_arity_expr": parsed["arity_expr"],
            "resolver_arity_source": parsed["arity_source"],
            "resolver_n_registrations": parsed["n_registrations"],
        }
    )
    assert not r.passed and r.is_hard_fail
    assert "ARITY too small" in r.detail


def test_unresolvable_arity_skips_visibly_and_never_claims_a_match(tmp_path):
    """`arity <?> matches` was a positive claim about a check that did not run."""
    text = CLEAN_CPP.replace(
        "MicroMutableOpResolver<3>", "MicroMutableOpResolver<kOpsFromAnotherHeader>"
    )
    parsed = _parse(tmp_path, text)
    assert parsed["arity"] is None and parsed["arity_source"] == "unresolved"
    r = check_resolver_arity(
        {
            "resolver_path": parsed["path"],
            "resolver_ops": set(parsed["ops"]),
            "resolver_arity": parsed["arity"],
            "resolver_arity_expr": parsed["arity_expr"],
            "resolver_arity_source": parsed["arity_source"],
            "resolver_n_registrations": parsed["n_registrations"],
        }
    )
    assert r.skipped and r.status == "SKIP"
    assert r.is_hard_skip  # => INCOMPLETE, not a pass
    assert "NOT checked" in r.detail
    assert "matches" not in r.detail


def test_resolver_exact_pass_text_makes_no_arity_claim():
    """Arity moved to its own gate; the set-equality gate must not imply it."""
    r = check_resolver_exact(
        {
            "model": mk_model(ops={"ADD": 1}),
            "resolver_path": "/x/ops.cpp",
            "resolver_ops": {"ADD"},
            "resolver_n_registrations": 1,
        }
    )
    assert r.passed
    assert "arity" not in r.detail.lower()


def test_conditional_arity_is_reported_as_a_range_not_a_false_alarm(tmp_path):
    text = CLEAN_CPP.replace(
        "    s_resolver.AddSlice();",
        "#ifdef CONFIG_SE_ENABLE_SLICE\n    s_resolver.AddSlice();\n#endif",
    )
    parsed = _parse(tmp_path, text)
    r = check_resolver_arity(
        {
            "resolver_path": parsed["path"],
            "resolver_ops": set(parsed["ops"]),
            "resolver_arity": parsed["arity"],
            "resolver_arity_expr": parsed["arity_expr"],
            "resolver_arity_source": parsed["arity_source"],
            "resolver_n_registrations": parsed["n_registrations"],
            "resolver_conditional": parsed["conditional"],
        }
    )
    assert r.passed  # 3 fits 2..3
    assert "somewhere in 2..3" in r.detail


# ==========================================================================
# REGRESSION: the sweep must audit the resolver that actually ships
# ==========================================================================


@needs_models
@needs_tree
def test_firmware_resolver_gate_catches_the_nsnet2_boot_blocker():
    """`--all` printed RESOLVER-EXACT=PASS for a model the firmware cannot boot."""
    if not os.path.isfile(FIRMWARE_RESOLVER):
        pytest.skip("no firmware resolver")
    parsed = parse_resolver_cpp(FIRMWARE_RESOLVER, ti.resolver_methods(TFLM_TREE))
    r = check_resolver_firmware(
        {
            "model": _model("nsnet2_plain_streaming.tflite"),
            "firmware_resolver_path": FIRMWARE_RESOLVER,
            "firmware_resolver_ops": set(parsed["ops"]),
            "firmware_resolver_n_registrations": parsed["n_registrations"],
            "tflm_tree": TFLM_TREE,
        }
    )
    assert not r.passed and r.is_hard_fail
    assert r.evidence["missing"] == ["FULLY_CONNECTED"]
    assert "AddFullyConnected" in r.detail


@needs_models
def test_sweep_audits_the_firmware_resolver_for_every_family(capsys):
    """The nudge used to be hardcoded to LiSenNet, so NSNet2 never got checked."""
    from lane.__main__ import main

    main(["audit", "--all", "--no-load", "--width", "200"])
    out = capsys.readouterr().out
    assert out.count("G-RESOLVER-FIRMWARE") >= 5
    nsnet2 = out.split("nsnet2_plain_streaming.tflite   [")[1].split("=" * 200)[0]
    assert "model_ops_micro.cpp" in nsnet2
    # ...and it says the model cannot boot, which the old sweep never did.
    assert "RESOLVER-FIRMWARE=FAIL" in out
    assert "FULLY_CONNECTED" in nsnet2


@needs_models
def test_sweep_no_longer_prints_resolver_pass_for_a_model_that_cannot_boot(capsys):
    from lane.__main__ import main

    main(["audit", "--all", "--no-load", "--width", "200"])
    summary = capsys.readouterr().out.split("SUMMARY")[1]
    row = [ln for ln in summary.splitlines()
           if "nsnet2_plain_streaming.tflite" in ln][0]
    # The hand-written reference file still matches...
    assert "RESOLVER-EXACT=PASS" in row
    # ...but the file on the board does not, and the row now says so.
    assert "RESOLVER-FIRMWARE=FAIL" in row
    assert "BLOCKED" in row


def test_summary_calls_out_incomplete_models_separately_from_blocked():
    incomplete = AuditReport(
        model_path="/tmp/a.tflite",
        results=run_gates({}, gates=[_skipping_gate("G-GONE", "hard-fail")]),
    )
    blocked = AuditReport(
        model_path="/tmp/b.tflite", results=run_gates({}, gates=[_failing_gate("G-BAD")])
    )
    text = render_text([incomplete, blocked], width=120)
    assert "1/2 models blocked: b.tflite" in text
    assert "deploy-blocking gate could not run" in text
    assert "a.tflite" in text.split("models blocked")[1]


# ==========================================================================
# REGRESSION: end-to-end -- each defect must reach the VERDICT, not just exist
# as an importable predicate.  Every one of these produced `verdict: CLEAR`
# and exit 0 before the fix.
# ==========================================================================


def _full_ctx(model, **extra):
    """A context in which everything except the defect under test is healthy."""
    ctx = {
        "model": model,
        "tflm_tree": "/synthetic/tflm",
        "supported_ops": set(model.op_histogram),
        "accelerated": set(model.op_histogram),
        "accel_status": {"ok": True},
        "resolver_path": "/synthetic/ops.cpp",
        "resolver_ops": set(model.op_histogram),
        "resolver_n_registrations": len(model.op_histogram),
        "resolver_arity": len(model.op_histogram),
        "resolver_arity_source": "literal",
        "firmware_resolver_path": "/synthetic/model_ops_micro.cpp",
        "firmware_resolver_ops": set(model.op_histogram),
        "firmware_resolver_n_registrations": len(model.op_histogram),
        "load_result": {
            "ok": True, "returncode": 0, "signal": None, "stderr": "",
            "n_inputs": len(model.inputs), "n_outputs": len(model.outputs),
        },
    }
    ctx.update(extra)
    return ctx


def _verdict(model, **extra):
    return AuditReport(
        model_path="/tmp/x.tflite", results=run_gates(_full_ctx(model, **extra))
    )


def _healthy_model(**kw):
    kw.setdefault("tensors", [mk_tensor(0, "feat"), mk_tensor(1, "state")])
    kw.setdefault("inputs", (0, 1))
    kw.setdefault("outputs", (0, 1))
    kw.setdefault("ops", {"CONV_2D": 2, "ADD": 1})
    return mk_model(**kw)


def test_end_to_end_healthy_synthetic_model_is_clear():
    """The control: without a defect the same context really does say CLEAR."""
    rep = _verdict(_healthy_model())
    assert rep.verdict == "CLEAR", [r.detail for r in rep.results if not r.passed]
    assert exit_code([rep]) == EXIT_OK


@pytest.mark.parametrize(
    "name,model_kw,expect_gate",
    [
        (
            "negative per-tensor scale",
            dict(tensors=[mk_tensor(0, "feat"), mk_tensor(1, "state"),
                          mk_tensor(2, "mid", scales=(-0.00697891,))]),
            "G-SCALE-POSITIVE",
        ),
        (
            "negative per-axis filter scale",
            dict(tensors=[mk_tensor(0, "feat"), mk_tensor(1, "state"),
                          mk_tensor(2, "filt", shape=(6,), qdim=0,
                                    scales=(0.01, -0.0103, 0.02, 0.03, 0.04, 0.05),
                                    zps=(0,) * 6)]),
            "G-SCALE-POSITIVE",
        ),
        (
            "zero output scale",
            dict(tensors=[mk_tensor(0, "feat"), mk_tensor(1, "state"),
                          mk_tensor(2, "conv_out", scales=(0.0,))]),
            "G-SCALE-POSITIVE",
        ),
        (
            "short per-axis scale vector",
            dict(tensors=[mk_tensor(0, "feat"), mk_tensor(1, "state"),
                          mk_tensor(2, "w", shape=(12,), qdim=0,
                                    scales=tuple(0.01 * (i + 1) for i in range(9)),
                                    zps=(0,) * 9)]),
            "G-QUANT-STRUCT",
        ),
        (
            "zero_point out of dtype range",
            dict(tensors=[mk_tensor(0, "feat"), mk_tensor(1, "state"),
                          mk_tensor(2, "mid", dtype="INT8", zps=(4242,))]),
            "G-QUANT-STRUCT",
        ),
        (
            "dropped state output",
            dict(inputs=(0, 1), outputs=(0,)),
            "G-IO-SYMMETRY",
        ),
        (
            "second subgraph",
            dict(n_subgraphs=2),
            "G-SINGLE-SUBGRAPH",
        ),
    ],
)
def test_end_to_end_each_defect_blocks_with_exit_1(name, model_kw, expect_gate):
    rep = _verdict(_healthy_model(**model_kw))
    assert rep.verdict == "BLOCKED", name
    assert expect_gate in [r.gate.id for r in rep.hard_fails], name
    assert exit_code([rep]) == EXIT_BLOCKED, name


def test_end_to_end_a_missing_tflm_tree_is_incomplete_not_clear():
    """The default state off this box: no checkout, so G-OPSET cannot run."""
    ctx = _full_ctx(_healthy_model())
    ctx.pop("supported_ops")
    ctx["tflm_tree"] = None
    ctx["tflm_tree_error"] = "no tflite-micro checkout found"
    rep = AuditReport(model_path="/tmp/x.tflite", results=run_gates(ctx))
    assert rep.verdict == "INCOMPLETE"
    assert [r.gate.id for r in rep.hard_skips] == ["G-OPSET"]
    assert exit_code([rep]) == EXIT_INCOMPLETE


def test_end_to_end_an_unreadable_resolver_is_incomplete_not_clear():
    ctx = _full_ctx(_healthy_model())
    for k in ("resolver_ops", "resolver_n_registrations", "resolver_arity"):
        ctx.pop(k, None)
    ctx["resolver_error"] = "could not read /x/ops.cpp: [Errno 13] Permission denied"
    rep = AuditReport(model_path="/tmp/x.tflite", results=run_gates(ctx))
    assert rep.verdict == "INCOMPLETE"
    assert {"G-RESOLVER-EXACT", "G-RESOLVER-ARITY"} <= {
        r.gate.id for r in rep.hard_skips
    }
    assert exit_code([rep]) == EXIT_INCOMPLETE


def test_a_passing_superset_resolver_reports_the_graphs_op_count_not_its_own():
    """`len(resolver_ops)` was the denominator; on a superset it is a false claim."""
    ctx = {
        "model": mk_model(ops={"CONV_2D": 1, "ADD": 1}),
        "firmware_resolver_path": "/x/model_ops_micro.cpp",
        "firmware_resolver_ops": {"CONV_2D", "ADD", "TANH"},
        "firmware_resolver_n_registrations": 3,
    }
    r = check_resolver_firmware(ctx)
    assert r.passed  # a surplus registration is not a boot blocker
    assert "all 2 op(s) the graph uses" in r.detail
    assert "3 op(s) the graph uses" not in r.detail
    assert "TANH" in r.detail  # ...but it is still reported


def test_firmware_gate_still_hard_fails_on_a_missing_registration():
    r = check_resolver_firmware(
        {
            "model": mk_model(ops={"FULLY_CONNECTED": 16, "ADD": 1}),
            "firmware_resolver_path": "/x/model_ops_micro.cpp",
            "firmware_resolver_ops": {"ADD", "CONV_2D"},
            "firmware_resolver_n_registrations": 2,
        }
    )
    assert not r.passed and r.is_hard_fail
    assert r.evidence["missing"] == ["FULLY_CONNECTED"]
