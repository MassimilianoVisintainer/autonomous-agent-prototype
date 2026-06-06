"""Tests for src/retrieval.py.

All tests use a small synthetic corpus and a fake embedder — no real
SentenceTransformer model is loaded.
"""

import datetime
from unittest.mock import patch

import numpy as np
import pytest

from src import retrieval
from src.data_loaders import KnowledgeBaseChunk
from src.retrieval import RetrievedChunk, retrieve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(kb_id: str, content: str) -> KnowledgeBaseChunk:
    return KnowledgeBaseChunk(
        kb_id=kb_id,
        category="test",
        topic=kb_id.lower(),
        content=content,
        source_doc="Test Doc v1.0",
        last_updated=datetime.date(2025, 1, 1),
        applies_to_products=[],
        requires_authority_check=False,
    )


SYNTHETIC_CORPUS = [
    _make_chunk("RT-001", "Standard return window is 30 days."),
    _make_chunk("SH-001", "Shipping to EU takes 3-5 business days."),
    _make_chunk("PR-001", "Laptop stand supports 11 to 17 inch screens."),
    _make_chunk("AC-001", "Password reset via the Forgot password link."),
]

# Fixed embeddings: each chunk gets a unit vector in a distinct direction.
# Query will point toward chunk index 2 (PR-001).
_DIM = 4
_CHUNK_EMBEDDINGS = np.eye(_DIM, dtype=np.float32)  # shape (4, 4)
_QUERY_TOWARD_CHUNK_2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)


class FakeEmbedder:
    """Returns deterministic embeddings without loading any model weights."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        n = len(texts)
        if n == len(SYNTHETIC_CORPUS):
            # Encoding the corpus — return one unit vector per chunk
            return _CHUNK_EMBEDDINGS[:n].copy()
        # Encoding a single query — return a fixed vector toward chunk index 2
        return _QUERY_TOWARD_CHUNK_2.reshape(1, _DIM).copy()


# ---------------------------------------------------------------------------
# Fixture: reset module-level cache between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_retrieval_state():
    """Clear cached model, corpus, and embeddings before each test."""
    retrieval._model = None
    retrieval._corpus = None
    retrieval._embeddings = None
    yield
    retrieval._model = None
    retrieval._corpus = None
    retrieval._embeddings = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_retrieve_empty_query_returns_empty_list():
    assert retrieve("") == []


def test_retrieve_whitespace_query_returns_empty_list():
    assert retrieve("   ") == []


def test_retrieve_returns_top_k_results():
    with patch.object(retrieval, "_get_model", return_value=FakeEmbedder()), \
         patch("src.retrieval.load_knowledge_base", return_value=SYNTHETIC_CORPUS):
        results = retrieve("some query", k=2)
    assert len(results) == 2


def test_retrieve_results_sorted_by_score_descending():
    with patch.object(retrieval, "_get_model", return_value=FakeEmbedder()), \
         patch("src.retrieval.load_knowledge_base", return_value=SYNTHETIC_CORPUS):
        results = retrieve("some query", k=4)
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


def test_retrieved_chunk_includes_full_metadata():
    with patch.object(retrieval, "_get_model", return_value=FakeEmbedder()), \
         patch("src.retrieval.load_knowledge_base", return_value=SYNTHETIC_CORPUS):
        results = retrieve("some query", k=1)
    assert len(results) == 1
    rc = results[0]
    assert isinstance(rc, RetrievedChunk)
    assert isinstance(rc.chunk, KnowledgeBaseChunk)
    # All 8 fields must be accessible
    _ = rc.chunk.kb_id
    _ = rc.chunk.content
    _ = rc.chunk.source_doc
    _ = rc.chunk.last_updated
    _ = rc.chunk.applies_to_products
    _ = rc.chunk.requires_authority_check


def test_retrieve_returns_empty_on_model_failure():
    class BrokenEmbedder:
        def encode(self, *args, **kwargs):
            raise RuntimeError("GPU out of memory")

    with patch.object(retrieval, "_get_model", return_value=BrokenEmbedder()), \
         patch("src.retrieval.load_knowledge_base", return_value=SYNTHETIC_CORPUS):
        results = retrieve("some query")
    assert results == []
