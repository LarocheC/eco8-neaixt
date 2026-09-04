"""Cross-family benchmarking of the published models on the VBD test split.

This is the top layer of the repo: unlike ``common/`` (which every model family
imports and which therefore may not import *any* family), ``benchmarks/`` is
allowed to import from ``nsnet2/``, ``convfsenet/`` and ``lisennet/``. It reuses
each family's own enhancement code rather than reimplementing the streaming
loops, which are subtle and easy to get subtly wrong.

Two stages, deliberately decoupled:

    benchmarks.enhance   published checkpoint -> enhanced audio on disk
    benchmarks.score     enhanced audio       -> metric JSON

Enhancement is expensive (single-threaded ONNX Runtime, by design, so the RTF
numbers stay meaningful) and deterministic. Scoring is cheap-ish, parallel, and
the set of metrics we care about keeps growing. Caching the enhanced audio in
between means adding a metric later costs a re-score, not a re-inference.
"""
