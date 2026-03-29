from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml_service.embedding.base import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for tests — no model download."""

    def __init__(self, dim: int = 384, seed: int = 0) -> None:
        self._dim = dim
        self._rng = np.random.RandomState(seed)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._rng.randn(len(texts), self._dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    @property
    def dim(self) -> int:
        return self._dim


@pytest.fixture
def fake_embed() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def skill_alias_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "roadmap" / "week1" / "skill-alias.json"
