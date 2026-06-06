"""Retrieval-augmented generation pipeline — §3.1.2 of the thesis.

Implements dense-embedding-only retrieval using sentence-transformers'
all-MiniLM-L6-v2 model. Chunk embeddings are computed once at first use and
held in memory for the lifetime of the process.

Deliberately excludes BM25 hybrid scoring and cross-encoder re-ranking so
that each component's marginal contribution can be measured independently
during evaluation (later slices).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from src.data_loaders import KnowledgeBaseChunk, load_knowledge_base

logger = logging.getLogger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None
_corpus: list[KnowledgeBaseChunk] | None = None
_embeddings: np.ndarray | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _initialize_corpus() -> None:
    global _corpus, _embeddings
    if _corpus is not None:
        return
    _corpus = load_knowledge_base()
    model = _get_model()
    texts = [chunk.content for chunk in _corpus]
    raw = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    _embeddings = np.array(raw, dtype=np.float32)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: KnowledgeBaseChunk
    score: float


def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    """Return the top-k knowledge base chunks most similar to the query."""
    if not query.strip():
        return []

    try:
        _initialize_corpus()
        model = _get_model()
        query_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        query_vec = np.array(query_emb[0], dtype=np.float32)
        scores = _embeddings @ query_vec
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            RetrievedChunk(chunk=_corpus[i], score=float(scores[i]))
            for i in top_indices
        ]
    except Exception:
        logger.exception("Retrieval failed for query %r", query)
        return []
