"""Streaming document ingestion + index building for RAG.

Reuses ``iter_dataset_records`` from the data pipeline so any format the trainer
can ingest (text/JSONL/CSV/Parquet/HF/archives/HTTP) can also be indexed. Nothing
here loads a full corpus into RAM — records are streamed and chunked on the fly.

TF-IDF needs its IDF table fitted before embedding, so that backend is two-pass:
  pass 1 — stream up to ``idf_fit_limit`` chunks and fit IDF
  pass 2 — stream again and embed every chunk into the store
Other backends are single-pass (embed on the fly).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from nanoforge.rag.chunker import chunk_text
from nanoforge.rag.embeddings import TfIdfEmbedder, get_embedder
from nanoforge.rag.store import NumpyVectorStore


def _iter_chunks(
    input_paths: list[Path],
    text_key: str,
    text_columns: list[str] | None,
    max_chunk_chars: int,
    overlap_chars: int,
) -> Iterator[tuple[str, str, dict]]:
    """Yield (text_chunk, source, metadata) for every chunk of every record."""
    from nanoforge.data.formats import iter_dataset_records

    for record in iter_dataset_records(
        input_paths,
        text_key=text_key,
        text_columns=tuple(text_columns) if text_columns else None,
    ):
        for chunk in chunk_text(record.text, max_chunk_chars, overlap_chars):
            yield chunk, record.source, dict(record.metadata)


def build_index(
    input_paths: list[Path],
    out_dir: Path,
    backend: str = "tfidf",
    max_chunk_chars: int = 512,
    overlap_chars: int = 64,
    text_key: str = "text",
    text_columns: list[str] | None = None,
    batch_size: int = 64,
    max_docs: int | None = None,
    idf_fit_limit: int = 100_000,
    progress_callback=None,
    engine=None,
    st_model: str = "all-MiniLM-L6-v2",
    tfidf_vocab: int = 16_384,
) -> NumpyVectorStore:
    """Build and persist a RAG index. Returns the resulting store."""
    input_paths = [Path(p) for p in input_paths]

    embedder = get_embedder(
        backend=backend,
        engine=engine,
        st_model=st_model,
        tfidf_vocab=tfidf_vocab,
    )

    # Record which backend actually got used (get_embedder may have fallen back).
    effective_backend = backend
    embedder_state: dict = {}

    if isinstance(embedder, TfIdfEmbedder):
        effective_backend = "tfidf"
        # Pass 1: fit IDF from a bounded sample of chunks.
        fit_texts: list[str] = []
        for chunk, _src, _md in _iter_chunks(
            input_paths, text_key, text_columns, max_chunk_chars, overlap_chars
        ):
            fit_texts.append(chunk)
            if len(fit_texts) >= idf_fit_limit:
                break
        embedder.fit(fit_texts)
        embedder_state = embedder.to_dict()

    # Pass 2 (or single pass for non-tfidf): stream, embed, persist.
    chunks = _iter_chunks(
        input_paths, text_key, text_columns, max_chunk_chars, overlap_chars
    )
    store = NumpyVectorStore.build_and_save(
        out_dir,
        embedder,
        chunks,
        backend=effective_backend,
        embedder_state=embedder_state,
        batch_size=batch_size,
        max_docs=max_docs,
        progress_callback=progress_callback,
    )
    return store
