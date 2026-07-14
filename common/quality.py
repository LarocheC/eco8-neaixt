"""Perceptual speech-quality metrics beyond PESQ.

PESQ (``common/metrics.py``) stays the primary number reported across this repo,
but it is an intrusive, ITU-P.862 signal-comparison measure designed for codec
and telephony degradations. It is known to under-reflect the artefacts that
neural enhancers actually introduce (musical noise, over-suppression, phase
smearing), so a model can trade real perceived quality for PESQ. The three
metrics here are learned MOS estimators trained on human listening tests, and
they disagree with PESQ often enough to be worth reporting alongside it.

    metric   reference?      what it predicts                     range
    ------   ----------      --------------------------------     -----
    DNSMOS   no-reference    ITU P.835 SIG / BAK / OVRL + P.808    1-5, higher better
    NISQA    no-reference    overall MOS + 4 degradation dims      1-5, higher better
    SCOREQ   both            NR: MOS-like; REF: embedding distance see below

Provenance
----------
DNSMOS and NISQA come via ``torchmetrics``, which wraps the *official* weights:
Microsoft's DNS-Challenge P.835/P.808 ONNX graphs, and Gabriel Mittag's
``nisqa.tar``. We deliberately do not use the standalone ``nisqa`` PyPI package:
it hard-pins ``torch==2.2.1`` and would be uninstallable alongside this repo's
torch. SCOREQ comes from its own package (Ragano et al., NeurIPS 2024), using
its ONNX backend -- the ``pytorch`` extra needs ``fairseq``, which is not
installable on modern torch.

Weights are downloaded on first use (~380 MB for SCOREQ) and cached under
``~/.torchmetrics`` and ``~/.cache/scoreq``.

A note on SCOREQ's two modes, because they are not interchangeable:
  * ``scoreq_nr``  -- no-reference, MOS-like. Higher is better.
  * ``scoreq_ref`` -- full-reference, and it is an L2 *distance* between the
    enhanced and clean embeddings, NOT a MOS. **Lower is better**, 0.0 means
    "indistinguishable from the reference". Reading it as a MOS inverts the
    ranking, so ``HIGHER_IS_BETTER`` below is load-bearing for any table you
    render from these numbers.

Cost, measured on the 824-utterance VoiceBank-DEMAND test split (32-core box):
DNSMOS dominates at ~4 min, SCOREQ ~1.5 min/mode, NISQA ~8 s, PESQ ~20 s.
DNSMOS and NISQA are bound by CPU-side librosa mel extraction rather than by the
nets themselves, so they are parallelised with *processes*; SCOREQ is a large
wav2vec2 graph whose ONNX Runtime session releases the GIL and is thread-safe,
so it is parallelised with *threads* and the 378 MB session is loaded once.
"""
from __future__ import annotations

import hashlib
import warnings

import numpy as np
from joblib import Parallel, delayed

from common.metrics import eval_pesq

ALL_METRICS = ("pesq", "dnsmos", "nisqa", "scoreq")

# Column -> is a higher value better? Consumed by report renderers to orient
# deltas. scoreq_ref is a distance and inverts; see the module docstring.
HIGHER_IS_BETTER = {
    "pesq": True,
    "dnsmos_ovrl": True,
    "dnsmos_sig": True,
    "dnsmos_bak": True,
    "dnsmos_p808": True,
    "nisqa_mos": True,
    "nisqa_noi": True,
    "nisqa_dis": True,
    "nisqa_col": True,
    "nisqa_loud": True,
    "scoreq_nr": True,
    "scoreq_ref": False,
}

# Which columns each metric produces, in the order the backend returns them.
_DNSMOS_COLS = ("dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl")
_NISQA_COLS = ("nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud")
_SCOREQ_COLS = ("scoreq_nr", "scoreq_ref")

METRIC_COLUMNS = {
    "pesq": ("pesq",),
    "dnsmos": _DNSMOS_COLS,
    "nisqa": _NISQA_COLS,
    "scoreq": _SCOREQ_COLS,
}


def columns_for(metrics) -> list[str]:
    """Flatten a metric selection into its output column names, in a stable order."""
    out = []
    for m in metrics:
        out.extend(METRIC_COLUMNS[m])
    return out


def backend_versions() -> dict:
    """Versions of everything that can move a score. Recorded alongside results.

    A MOS number is only auditable against the code that produced it: torchmetrics
    ships the DNSMOS/NISQA graphs and has changed their preprocessing between
    releases, and ORT's kernels decide the last decimal. Without this block a
    future re-run that disagrees is unattributable.
    """
    import importlib.metadata as md

    out = {}
    for pkg in ("torchmetrics", "scoreq", "onnxruntime", "torch", "torchaudio",
                "librosa", "pesq", "numpy"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def resolve_metrics(spec) -> tuple[str, ...]:
    """Parse a CLI --metrics value ('all', 'none', or 'dnsmos,nisqa') into a tuple."""
    if spec is None or spec.strip().lower() in ("", "none"):
        return ()
    if spec.strip().lower() == "all":
        return ALL_METRICS
    picked = tuple(s.strip().lower() for s in spec.split(",") if s.strip())
    unknown = [m for m in picked if m not in ALL_METRICS]
    if unknown:
        raise ValueError(
            f"unknown metric(s) {unknown}; choose from {list(ALL_METRICS)}, 'all', or 'none'"
        )
    return picked


# --- workers -----------------------------------------------------------------
# Top-level so joblib's loky backend can pickle them. Each imports its backend
# lazily *inside* the worker: `import scoreq` in particular has an import-time
# side effect (it instantiates a default Scoreq(), downloading a 378 MB ONNX),
# so it must never be imported at module scope here.


def _dnsmos_one(x, sr, personalized):
    import torch
    from torchmetrics.functional.audio.dnsmos import (
        deep_noise_suppression_mean_opinion_score as _dnsmos,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # num_threads=1: we parallelise across utterances, so each worker's
            # ORT session must stay single-threaded or the pools fight.
            v = _dnsmos(
                torch.from_numpy(x), sr, personalized, device="cpu", num_threads=1
            )
            return np.asarray(v, dtype=np.float64)
        except Exception:
            return np.full(len(_DNSMOS_COLS), np.nan)


def _nisqa_one(x, sr):
    import torch
    from torchmetrics.functional.audio.nisqa import (
        non_intrusive_speech_quality_assessment as _nisqa,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            v = _nisqa(torch.from_numpy(x), sr)
            return np.asarray(v, dtype=np.float64)
        except Exception:
            # NISQA needs enough frames to build its mel segments; very short
            # utterances legitimately fail rather than being worth a score.
            return np.full(len(_NISQA_COLS), np.nan)


def _pesq_one(ref, est, sr):
    s = eval_pesq(ref, est, sr)
    return np.nan if s == -1 else float(s)  # -1 is common/metrics' failure sentinel


class _Scoreq:
    """Lazily-built SCOREQ sessions, feeding in-memory audio straight to ONNX.

    ``Scoreq.predict()`` only accepts file paths -- it would have us write
    ~30k temporary wavs across a sweep. Its preprocessing is just "load, mono,
    resample to 16 kHz", and our audio is already mono 16 kHz float32, so we
    replicate the tail of that pipeline (``dynamic_pad`` to the wav2vec2 stride)
    and call the session directly.
    """

    def __init__(self, sr):
        import scoreq  # noqa: PLC0415 -- import-time side effect, see module docstring
        from scoreq.scoreq import dynamic_pad

        if sr != 16000:
            raise ValueError(f"SCOREQ is a 16 kHz model; got sr={sr}")
        self._pad = dynamic_pad

        # `import scoreq` eagerly builds a default Scoreq(natural, nr) and loads
        # its 378 MB graph. Reuse that instance rather than paying for a second
        # copy -- but verify it is the config we want, in case upstream changes
        # the default.
        default = getattr(scoreq, "scoreq", None)
        if (
            default is not None
            and getattr(default, "data_domain", None) == "natural"
            and getattr(default, "mode", None) == "nr"
            and getattr(default, "use_onnx", False)
        ):
            self.nr = default
        else:
            self.nr = scoreq.Scoreq(data_domain="natural", mode="nr", use_onnx=True)
        self.ref = scoreq.Scoreq(data_domain="natural", mode="ref", use_onnx=True)
        self._ref_emb_cache = {}

    def _embed(self, inst, x):
        import torch

        arr = self._pad(torch.from_numpy(np.ascontiguousarray(x))
                        .unsqueeze(0)).numpy()
        name = inst.session.get_inputs()[0].name
        return inst.session.run(None, {name: arr})[0]

    def nr_score(self, x):
        try:
            return float(self._embed(self.nr, x).item())
        except Exception:
            return np.nan

    def ref_score(self, est, ref):
        """L2 distance between enhanced and clean embeddings. Lower is better."""
        try:
            # The clean references are identical across every model/condition in
            # a sweep, so their embeddings are computed once and reused. Keyed by
            # content hash rather than identity so it survives re-reading refs.
            key = hashlib.blake2b(
                np.ascontiguousarray(ref, dtype=np.float32).tobytes(), digest_size=16
            ).digest()
            emb_ref = self._ref_emb_cache.get(key)
            if emb_ref is None:
                emb_ref = self._embed(self.ref, ref)
                self._ref_emb_cache[key] = emb_ref
            emb_est = self._embed(self.ref, est)
            return float(np.linalg.norm(emb_est - emb_ref))
        except Exception:
            return np.nan


class QualitySuite:
    """Scores (enhanced, clean) waveform pairs with any subset of the metrics.

    Reuse one instance across a sweep: the SCOREQ sessions and the clean-
    reference embedding cache are held on the instance, so scoring the Nth model
    is meaningfully cheaper than the first.

        suite = QualitySuite(metrics=("pesq", "dnsmos"))
        per_utt = suite.score(ests, refs)           # {column: (N,) array, nan on failure}
        means   = QualitySuite.summarize(per_utt)   # {column: float}

    All waveforms are 1-D float32 numpy at ``sr``. ``refs`` may be None only if
    no reference-requiring metric (pesq, scoreq's ref mode) is selected.
    """

    def __init__(self, metrics=ALL_METRICS, sr=16000, n_jobs=16, dnsmos_personalized=False):
        self.metrics = tuple(metrics)
        unknown = [m for m in self.metrics if m not in ALL_METRICS]
        if unknown:
            raise ValueError(f"unknown metric(s) {unknown}; choose from {list(ALL_METRICS)}")
        self.sr = sr
        self.n_jobs = n_jobs
        self.dnsmos_personalized = dnsmos_personalized
        self._scoreq = None  # built on first use; loading it costs ~380 MB

    @property
    def columns(self) -> list[str]:
        return columns_for(self.metrics)

    def _needs_refs(self) -> bool:
        return "pesq" in self.metrics or "scoreq" in self.metrics

    def score(self, ests, refs=None) -> dict[str, np.ndarray]:
        """Score N utterances. Returns {column: (N,) float array}, nan where a metric failed."""
        ests = [np.asarray(e, dtype=np.float32).squeeze() for e in ests]
        if refs is not None:
            refs = [np.asarray(r, dtype=np.float32).squeeze() for r in refs]
            if len(refs) != len(ests):
                raise ValueError(f"got {len(refs)} refs but {len(ests)} ests")
            # Deliberately NOT trimmed to a common length. An iSTFT can return up
            # to one hop fewer samples than it was given (NSNet2 does; ConvFSENet
            # and LiSenNet pass an explicit length and come back exact), and the
            # published eval scripts hand PESQ the untrimmed pair. PESQ aligns
            # internally and SCOREQ mean-pools over time, so both tolerate it --
            # and matching the published call is what lets the PESQ column here
            # reproduce RESULTS_*.md rather than sit 0.006 away from it.
            for r, e in zip(refs, ests):
                if abs(len(r) - len(e)) > 0.02 * max(len(r), len(e)):
                    raise ValueError(
                        f"ref/est lengths differ by >2% ({len(r)} vs {len(e)}); "
                        "that is a misalignment bug, not iSTFT edge trimming"
                    )
        elif self._needs_refs():
            raise ValueError(
                f"metrics {self.metrics} need clean references, but refs=None"
            )

        out: dict[str, np.ndarray] = {}

        if "pesq" in self.metrics:
            vals = Parallel(n_jobs=self.n_jobs)(
                delayed(_pesq_one)(r, e, self.sr) for r, e in zip(refs, ests)
            )
            out["pesq"] = np.asarray(vals, dtype=np.float64)

        if "dnsmos" in self.metrics:
            rows = Parallel(n_jobs=self.n_jobs)(
                delayed(_dnsmos_one)(e, self.sr, self.dnsmos_personalized) for e in ests
            )
            arr = np.vstack(rows)
            for i, c in enumerate(_DNSMOS_COLS):
                out[c] = arr[:, i]

        if "nisqa" in self.metrics:
            rows = Parallel(n_jobs=self.n_jobs)(
                delayed(_nisqa_one)(e, self.sr) for e in ests
            )
            arr = np.vstack(rows)
            for i, c in enumerate(_NISQA_COLS):
                out[c] = arr[:, i]

        if "scoreq" in self.metrics:
            if self._scoreq is None:
                self._scoreq = _Scoreq(self.sr)
            sq = self._scoreq
            # Threads, not processes: the session is 378 MB and ORT releases the
            # GIL, so 16 threads share one copy instead of forking 16 of them.
            nr = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(sq.nr_score)(e) for e in ests
            )
            rf = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(sq.ref_score)(e, r) for e, r in zip(ests, refs)
            )
            out["scoreq_nr"] = np.asarray(nr, dtype=np.float64)
            out["scoreq_ref"] = np.asarray(rf, dtype=np.float64)

        return out

    @staticmethod
    def summarize(per_utt) -> dict[str, float]:
        """Mean of each column, ignoring failed utterances (nan)."""
        means = {}
        for col, vals in per_utt.items():
            vals = np.asarray(vals, dtype=np.float64)
            ok = vals[~np.isnan(vals)]
            means[col] = float(ok.mean()) if ok.size else float("nan")
        return means

    @staticmethod
    def failures(per_utt) -> dict[str, int]:
        """Count of utterances each column failed to score."""
        return {
            col: int(np.isnan(np.asarray(v, dtype=np.float64)).sum())
            for col, v in per_utt.items()
        }
