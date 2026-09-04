"""Inventory of the models published to the HuggingFace Hub.

The local ``cp_*/`` directories are working state -- some are stale, some were
renamed (the "monarch" -> "blockdiag" correction), some no longer exist. The Hub
repos are the artefacts of record, and are what the RESULTS_*.md tables refer
to, so the benchmark pulls from there and nowhere else.

Each family exposes both a FP32 and an int8 ONNX graph under a fixed file
naming convention; LiSenNet additionally has a real-time condition (noisy phase
instead of Griffin-Lim) which is the number it actually deploys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import snapshot_download

NSNET2_REPO = "claroche1/sparse-nsnet2-checkpoints"
CONVFSENET_REPO = "claroche1/convfsenet"
LISENNET_REPO = "claroche1/LiSenNet"


@dataclass(frozen=True)
class Model:
    family: str          # nsnet2 | convfsenet | lisennet
    name: str            # report label, unique across families
    repo: str            # HF repo id
    subfolder: str       # "" for a repo root
    conditions: tuple[str, ...] = field(default=("fp32", "int8"))

    @property
    def slug(self) -> str:
        return f"{self.family}__{self.name}"

    def local_dir(self, root: Path) -> Path:
        d = Path(root) / self.repo.replace("/", "__")
        return d / self.subfolder if self.subfolder else d


# The 13 NSNet2 structured-factorization variants (RESULTS_NSNET2.md).
_NSNET2 = [
    "baseline",
    "blockdiag_8", "blockdiag_fc", "blockdiag_full", "wide_blockdiag",
    "monarch_8", "monarch_fc", "monarch_full", "wide_monarch",
    "butterfly_2blocks", "butterfly_fc", "butterfly_full", "butterfly_ortho",
]

# LiSenNet reports a third condition: int8 mask + noisy phase, i.e. dropping the
# non-causal Griffin-Lim for the causal phase you can actually stream. That is
# the deployed configuration, so it gets a row of its own.
_LISENNET_CONDS = ("fp32", "int8", "int8_rt")

MODELS: list[Model] = (
    [Model("nsnet2", v, NSNET2_REPO, v) for v in _NSNET2]
    + [
        Model("convfsenet", "convfsenet", CONVFSENET_REPO, ""),
        Model("convfsenet", "128-256", CONVFSENET_REPO, "128-256"),
    ]
    + [
        Model("lisennet", v, LISENNET_REPO, v, _LISENNET_CONDS)
        for v in ("gru", "conv", "conv-hardened", "conv-hardened-deep")
    ]
)

# Unprocessed noisy input and the clean reference, scored through the identical
# metric path. The learned MOS metrics are uncalibrated across corpora, so an
# absolute DNSMOS of 3.1 means nothing until you know where noisy and clean land
# on the same axis.
BASELINES = ("noisy", "clean")


def by_slug(slug: str) -> Model:
    for m in MODELS:
        if m.slug == slug:
            return m
    raise KeyError(f"no published model with slug {slug!r}")


def fetch(root: Path, repos=None) -> None:
    """Download the published repos into ``root`` (idempotent; skips cached files)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for repo in repos or (NSNET2_REPO, CONVFSENET_REPO, LISENNET_REPO):
        dest = root / repo.replace("/", "__")
        print(f"fetching {repo} -> {dest}")
        snapshot_download(repo_id=repo, local_dir=str(dest))
