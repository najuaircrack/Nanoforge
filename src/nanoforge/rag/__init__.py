"""Retrieval-Augmented Generation for Nanoforge.

Optional, lazily-imported. Nothing here is pulled in by ``import nanoforge`` — the
webapp and CLI import from this package only when RAG is explicitly used.

The default embedding backend is a pure-numpy hashed TF-IDF vectorizer, so the
whole feature works with only ``numpy`` installed. Better backends (the loaded
Nanoforge model, or sentence-transformers) are opportunistic upgrades.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing torch/engine at module load
    from nanoforge.rag.retriever import Retriever

__all__ = ["load_index", "build_index", "chunk_text"]


def load_index(index_dir: str | Path, engine=None) -> "Retriever":
    """Load a persisted index and return a ready-to-query :class:`Retriever`.

    Reconstructs the embedder described by the stored index metadata. For a
    ``model`` backend, pass the loaded ``GenerationEngine`` as ``engine`` so the
    query encoder matches the one used at index time.
    """
    from nanoforge.rag.embeddings import embedding_identity, get_embedder
    from nanoforge.rag.retriever import Retriever
    from nanoforge.rag.store import NumpyVectorStore

    store = NumpyVectorStore.load(index_dir)
    embedder = get_embedder(
        backend=store.meta.backend,
        engine=engine,
        tfidf_state=store.meta.embedder_state,
        strict=True,
    )
    if store.meta.schema_version == 2 and store.meta.encoder_identity:
        if embedding_identity(embedder) != store.meta.encoder_identity:
            raise ValueError("index encoder identity mismatch; rebuild with the original encoder")
    return Retriever(store, embedder)


def build_index(*args, **kwargs):
    """Thin re-export of :func:`nanoforge.rag.indexer.build_index` (lazy)."""
    from nanoforge.rag.indexer import build_index as _build_index

    return _build_index(*args, **kwargs)


def chunk_text(*args, **kwargs):
    """Thin re-export of :func:`nanoforge.rag.chunker.chunk_text` (lazy)."""
    from nanoforge.rag.chunker import chunk_text as _chunk_text

    return _chunk_text(*args, **kwargs)
