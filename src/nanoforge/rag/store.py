"""Persisted numpy vector store for RAG.

An index directory holds three files, written atomically (temp + rename):

    vectors.npy      float32 [N, dim]        — L2-normalised document vectors
    metadata.json    [{text_chunk, source, metadata}]  — one entry per vector
    index_meta.json  IndexMeta               — self-describing backend descriptor

The index is self-describing: ``index_meta.json`` records which embedding backend
built it plus the embedder state (e.g. the TF-IDF IDF table), so a query-time
embedder can be reconstructed without any external config.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

_VECTORS_FILE = "vectors.npy"
_METADATA_FILE = "metadata.json"
_META_FILE = "index_meta.json"


@dataclass
class IndexMeta:
    backend: str
    dim: int
    num_docs: int
    embedder_state: dict = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "IndexMeta":
        return cls(
            backend=d["backend"],
            dim=int(d["dim"]),
            num_docs=int(d["num_docs"]),
            embedder_state=d.get("embedder_state", {}),
            created_at=d.get("created_at", ""),
        )


def _atomic_write_bytes(path: Path, write_fn) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        write_fn(fh)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class NumpyVectorStore:
    """In-memory view over a persisted numpy index."""

    def __init__(self, vectors: np.ndarray, docs: list[dict], meta: IndexMeta) -> None:
        self.vectors = vectors
        self.docs = docs
        self.meta = meta

    # ── Load ──────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, index_dir: str | Path) -> "NumpyVectorStore":
        base = Path(index_dir)
        if not base.exists():
            raise FileNotFoundError(f"Index directory not found: {base}")
        vec_path = base / _VECTORS_FILE
        meta_path = base / _META_FILE
        docs_path = base / _METADATA_FILE
        for p in (vec_path, meta_path, docs_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing index file: {p}")
        vectors = np.load(vec_path).astype(np.float32)
        meta = IndexMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
        docs = json.loads(docs_path.read_text(encoding="utf-8"))
        return cls(vectors, docs, meta)

    # ── Build + save (streaming) ────────────────────────────────────────────────

    @classmethod
    def build_and_save(
        cls,
        index_dir: str | Path,
        embedder,
        chunks: Iterable[tuple[str, str, dict]],
        backend: str,
        embedder_state: dict | None = None,
        batch_size: int = 64,
        max_docs: int | None = None,
        progress_callback=None,
    ) -> "NumpyVectorStore":
        """Embed ``chunks`` (text, source, metadata) in batches and persist.

        Vectors are accumulated per batch and stacked once at the end. Nothing
        beyond one batch of raw text is held before embedding.
        """
        base = Path(index_dir)
        base.mkdir(parents=True, exist_ok=True)

        vec_batches: list[np.ndarray] = []
        docs: list[dict] = []
        buf_text: list[str] = []
        buf_meta: list[tuple[str, dict]] = []
        n_total = 0

        def _flush() -> None:
            nonlocal buf_text, buf_meta
            if not buf_text:
                return
            vecs = embedder.embed_batch(buf_text)
            vec_batches.append(vecs.astype(np.float32))
            for text, (source, md) in zip(buf_text, buf_meta):
                docs.append({"text_chunk": text, "source": source, "metadata": md})
            buf_text = []
            buf_meta = []

        for text, source, md in chunks:
            if max_docs is not None and n_total >= max_docs:
                break
            buf_text.append(text)
            buf_meta.append((source, md))
            n_total += 1
            if len(buf_text) >= batch_size:
                _flush()
                if progress_callback:
                    progress_callback(n_total)
        _flush()
        if progress_callback:
            progress_callback(n_total)

        if vec_batches:
            vectors = np.vstack(vec_batches).astype(np.float32)
        else:
            dim = int(getattr(embedder, "dim", 1))
            vectors = np.zeros((0, dim), dtype=np.float32)

        meta = IndexMeta(
            backend=backend,
            dim=int(vectors.shape[1]) if vectors.size else int(getattr(embedder, "dim", 1)),
            num_docs=len(docs),
            embedder_state=embedder_state or {},
            created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        )

        _atomic_write_bytes(base / _VECTORS_FILE, lambda fh: np.save(fh, vectors))
        _atomic_write_text(base / _METADATA_FILE, json.dumps(docs, ensure_ascii=False))
        _atomic_write_text(base / _META_FILE, json.dumps(meta.to_dict(), ensure_ascii=False))

        return cls(vectors, docs, meta)
