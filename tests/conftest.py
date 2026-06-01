"""Pytest configuration for the sparse-nsnet2 test suite.

Session-scoped determinism fixture (D-09). The CUBLAS_WORKSPACE_CONFIG
environment variable MUST be set BEFORE ``import torch`` (top-of-file,
NOT inside an autouse fixture) — see RESEARCH.md "Pitfall:
CUBLAS_WORKSPACE_CONFIG set too late". cuBLAS reads the env var at the
first CUDA op, which can happen during torch import; an autouse fixture
runs after that.
"""

import os

# Set BEFORE any torch import; setdefault preserves a developer override.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def _determinism():
    """Enforced for every test — see D-09."""
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    yield
    # No teardown — process-global toggles; pytest exits after the session.
