"""tests/test_quality.py - gates for common/quality.py (DNSMOS / NISQA / SCOREQ).

The fast tests are pure logic and pull no model weights. The monotonicity gate at
the bottom runs the real backends and needs ~380 MB of cached weights, so it is
marked ``slow`` and skipped unless you ask for it:

    uv run pytest tests/test_quality.py                 # fast gates only
    uv run pytest tests/test_quality.py -m slow         # + the real-backend gate

The invariant this file exists to protect is SCOREQ-ref's polarity. It is an
embedding *distance*, so lower is better, while every other column is
higher-is-better. Read it the wrong way round and every quantization delta in
the report silently flips sign.
"""
from __future__ import annotations

import numpy as np
import pytest

from common.quality import (
    ALL_METRICS,
    HIGHER_IS_BETTER,
    METRIC_COLUMNS,
    QualitySuite,
    columns_for,
    resolve_metrics,
)


# ---------------------------------------------------------------------------
# metric selection
# ---------------------------------------------------------------------------

def test_resolve_metrics_all_and_none():
    assert resolve_metrics("all") == ALL_METRICS
    assert resolve_metrics("none") == ()
    assert resolve_metrics("") == ()
    assert resolve_metrics(None) == ()


def test_resolve_metrics_subset_and_whitespace():
    assert resolve_metrics("dnsmos,nisqa") == ("dnsmos", "nisqa")
    assert resolve_metrics(" pesq , scoreq ") == ("pesq", "scoreq")


def test_resolve_metrics_rejects_unknown():
    with pytest.raises(ValueError, match="unknown metric"):
        resolve_metrics("dnsmos,bogus")


def test_columns_for_is_stable_and_flat():
    assert columns_for(("dnsmos", "nisqa")) == [
        "dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl",
        "nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud",
    ]


# ---------------------------------------------------------------------------
# polarity — the load-bearing invariant
# ---------------------------------------------------------------------------

def test_every_produced_column_declares_a_polarity():
    """A new column without a HIGHER_IS_BETTER entry would KeyError in the report."""
    produced = {c for cols in METRIC_COLUMNS.values() for c in cols}
    assert produced == set(HIGHER_IS_BETTER), (
        "METRIC_COLUMNS and HIGHER_IS_BETTER disagree; "
        "the report renderer orients deltas from the latter"
    )


def test_scoreq_ref_is_lower_is_better_and_is_alone_in_that():
    assert HIGHER_IS_BETTER["scoreq_ref"] is False
    inverted = [c for c, hib in HIGHER_IS_BETTER.items() if not hib]
    assert inverted == ["scoreq_ref"]


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def test_summarize_ignores_failed_utterances():
    per_utt = {"pesq": np.array([3.0, np.nan, 1.0])}
    assert QualitySuite.summarize(per_utt)["pesq"] == pytest.approx(2.0)


def test_summarize_all_nan_is_nan_not_a_crash():
    assert np.isnan(QualitySuite.summarize({"pesq": np.array([np.nan, np.nan])})["pesq"])


def test_failures_counts_nan():
    per_utt = {"pesq": np.array([3.0, np.nan, np.nan]), "nisqa_mos": np.array([4.0, 4.0, 4.0])}
    assert QualitySuite.failures(per_utt) == {"pesq": 2, "nisqa_mos": 0}


# ---------------------------------------------------------------------------
# reference handling
# ---------------------------------------------------------------------------

def test_reference_requiring_metrics_reject_missing_refs():
    suite = QualitySuite(metrics=("pesq",))
    with pytest.raises(ValueError, match="need clean references"):
        suite.score([np.zeros(16000, dtype=np.float32)], refs=None)


def test_no_reference_metrics_do_not_require_refs():
    """dnsmos/nisqa never see the clean signal, so refs=None must be allowed."""
    suite = QualitySuite(metrics=("dnsmos", "nisqa"))
    assert suite._needs_refs() is False


def test_ref_est_count_mismatch_raises():
    suite = QualitySuite(metrics=("pesq",))
    with pytest.raises(ValueError, match="2 refs but 1 ests"):
        suite.score([np.zeros(16000, dtype=np.float32)],
                    refs=[np.zeros(16000, dtype=np.float32)] * 2)


def test_small_istft_length_mismatch_is_tolerated():
    """NSNet2's iSTFT returns up to one hop fewer samples; that is not a bug.

    Scored untrimmed, exactly as the published eval scripts do -- this is what
    lets the PESQ column reproduce RESULTS_*.md.
    """
    suite = QualitySuite(metrics=("pesq",), n_jobs=1)
    ref = np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32)
    est = ref[:15900].copy()                       # ~0.6% short, one hop
    out = suite.score([est], refs=[ref])           # must not raise
    assert "pesq" in out


def test_gross_length_mismatch_raises():
    """A 2x length difference is a misalignment bug and must not be scored silently."""
    suite = QualitySuite(metrics=("pesq",), n_jobs=1)
    ref = np.zeros(16000, dtype=np.float32)
    with pytest.raises(ValueError, match="misalignment bug"):
        suite.score([np.zeros(8000, dtype=np.float32)], refs=[ref])


# ---------------------------------------------------------------------------
# real backends
# ---------------------------------------------------------------------------

def test_backend_versions_covers_everything_that_moves_a_score():
    from common.quality import backend_versions

    v = backend_versions()
    # torchmetrics ships the DNSMOS/NISQA graphs and scoreq the SCOREQ graph;
    # onnxruntime decides the last decimal. An audit trail without these is not one.
    for pkg in ("torchmetrics", "scoreq", "onnxruntime", "torch"):
        assert v.get(pkg), f"{pkg} version missing from the provenance block"


def test_audit_bundle_refuses_to_misalign_conditions():
    """The bundle stores utt_ids once and indexes every condition against them.

    A condition scored over a different utterance set would silently line up with
    the wrong utterances -- the one bug that would make the audit trail lie.
    """
    from benchmarks.report import audit_bundle

    good = {"utt_ids": ["a", "b"], "n_utts": 2, "means": {}, "failures": {},
            "per_utt": {"pesq": [1.0, 2.0]}}
    bad = {**good, "utt_ids": ["a", "c"]}
    with pytest.raises(ValueError, match="different utterance set"):
        audit_bundle({("m1", "fp32"): good, ("m2", "fp32"): bad})


@pytest.mark.slow
def test_clean_outscores_noisy_on_every_metric():
    """The sanity gate: a metric that cannot rank clean above noisy is mis-wired.

    Uses the real VoiceBank test split and the real weights (~380 MB, cached
    after first run).
    """
    from common.dataset import load_voicebank_demand

    split = load_voicebank_demand()["test"].select(range(8))
    clean = [np.asarray(i["clean"]["array"], dtype=np.float32) for i in split]
    noisy = [np.asarray(i["noisy"]["array"], dtype=np.float32) for i in split]

    suite = QualitySuite(metrics=ALL_METRICS, n_jobs=8)
    c = QualitySuite.summarize(suite.score(clean, refs=clean))
    n = QualitySuite.summarize(suite.score(noisy, refs=clean))

    for col, higher_is_better in HIGHER_IS_BETTER.items():
        if higher_is_better:
            assert c[col] > n[col], f"{col}: clean {c[col]:.3f} !> noisy {n[col]:.3f}"
        else:
            assert c[col] < n[col], f"{col}: clean {c[col]:.3f} !< noisy {n[col]:.3f}"

    # Self-comparison pins the reference-mode metrics to their exact extremes.
    assert c["scoreq_ref"] == pytest.approx(0.0, abs=1e-4)
    assert c["pesq"] == pytest.approx(4.64, abs=0.05)
