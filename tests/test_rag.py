"""Tests for the nanoforge.rag package.

Covers: TF-IDF determinism, retrieval correctness, chunker, store roundtrip,
top-k ordering, and the no-index chat-stream regression.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nanoforge.rag.chunker import chunk_text
from nanoforge.rag.embeddings import TfIdfEmbedder, get_embedder
from nanoforge.rag.retriever import Retriever
from nanoforge.rag.store import NumpyVectorStore


# ── Chunker ───────────────────────────────────────────────────────────────────

def test_chunk_text_basic():
    chunks = chunk_text("hello world", max_chars=100, overlap=10)
    assert chunks == ["hello world"]


def test_chunk_text_splits():
    text = "a" * 1000
    chunks = chunk_text(text, max_chars=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_chunk_text_overlap():
    text = "abcdefghij" * 10  # 100 chars
    chunks = chunk_text(text, max_chars=20, overlap=5)
    # Each consecutive pair should share the last 5 chars of the first
    for a, b in zip(chunks, chunks[1:]):
        assert a[-5:] == b[:5]


def test_chunk_text_empty():
    assert chunk_text("", max_chars=100, overlap=10) == []
    assert chunk_text("   ", max_chars=100, overlap=10) == []


def test_chunk_text_invalid_args():
    with pytest.raises(ValueError):
        chunk_text("x", max_chars=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_text("x", max_chars=10, overlap=10)


# ── TF-IDF embedder ───────────────────────────────────────────────────────────

DOCS = [
    "the sky is blue and beautiful",
    "rust is a systems programming language",
    "transformers use self-attention mechanisms",
    "the ocean is deep and blue",
    "python is a high-level programming language",
]


def _fitted_embedder(vocab_size: int = 4096) -> TfIdfEmbedder:
    e = TfIdfEmbedder(vocab_size=vocab_size)
    e.fit(DOCS)
    return e


def test_tfidf_determinism():
    e = _fitted_embedder()
    v1 = e.embed_batch(["transformers use attention"])
    v2 = e.embed_batch(["transformers use attention"])
    assert np.allclose(v1, v2), "TF-IDF must be deterministic across calls"


def test_tfidf_self_cosine():
    e = _fitted_embedder()
    v = e.embed_batch(["transformers use attention"])[0]
    cos = float(v @ v)
    assert abs(cos - 1.0) < 1e-5, "L2-normalised vector should have self-cosine ≈ 1"


def test_tfidf_l2_normalised():
    e = _fitted_embedder()
    vecs = e.embed_batch(DOCS)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_tfidf_fit_changes_idf():
    e = TfIdfEmbedder(vocab_size=4096)
    v_before = e.embed_batch(["hello world"])[0].copy()
    # Corpus with *differential* document frequency: "hello" is common, "world"
    # is rare. Smoothed IDF = ln((1+n)/(1+df))+1 only diverges from the unfitted
    # all-ones default when terms differ in how many documents contain them.
    e.fit(["hello there"] * 9 + ["hello world"])
    v_after = e.embed_batch(["hello world"])[0]
    # After fitting, IDF weights differ across terms so the vector should change.
    assert not np.allclose(v_before, v_after)


def test_tfidf_serialisation_roundtrip():
    e = _fitted_embedder()
    d = e.to_dict()
    e2 = TfIdfEmbedder.from_dict(d)
    v1 = e.embed_batch(["rust programming"])
    v2 = e2.embed_batch(["rust programming"])
    assert np.allclose(v1, v2), "Serialised embedder must produce identical vectors"


def test_get_embedder_fallback():
    # Requesting an unavailable backend should fall back to TfIdfEmbedder
    emb = get_embedder("sentence-transformers")
    assert isinstance(emb, TfIdfEmbedder)


def test_get_embedder_tfidf_with_state():
    e = _fitted_embedder()
    state = e.to_dict()
    e2 = get_embedder("tfidf", tfidf_state=state)
    assert isinstance(e2, TfIdfEmbedder)
    v1 = e.embed_batch(["ocean deep"])
    v2 = e2.embed_batch(["ocean deep"])
    assert np.allclose(v1, v2)


# ── NumpyVectorStore ──────────────────────────────────────────────────────────

def _build_store(tmp_dir: str) -> NumpyVectorStore:
    e = _fitted_embedder()
    chunks = [(doc, f"src_{i}", {"idx": i}) for i, doc in enumerate(DOCS)]
    return NumpyVectorStore.build_and_save(
        tmp_dir, e, chunks, backend="tfidf", embedder_state=e.to_dict()
    )


def test_store_files_created():
    with tempfile.TemporaryDirectory() as d:
        _build_store(d)
        assert (Path(d) / "vectors.npy").exists()
        assert (Path(d) / "metadata.json").exists()
        assert (Path(d) / "index_meta.json").exists()


def test_store_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        store = _build_store(d)
        store2 = NumpyVectorStore.load(d)
        assert np.allclose(store.vectors, store2.vectors)
        assert len(store2.docs) == len(DOCS)
        assert store2.meta.num_docs == len(DOCS)
        assert store2.meta.backend == "tfidf"


def test_store_meta_has_embedder_state():
    with tempfile.TemporaryDirectory() as d:
        store = _build_store(d)
        store2 = NumpyVectorStore.load(d)
        assert "idf" in store2.meta.embedder_state


def test_store_load_missing_dir():
    with pytest.raises(FileNotFoundError):
        NumpyVectorStore.load("/nonexistent/path/xyz")


def test_store_max_docs():
    with tempfile.TemporaryDirectory() as d:
        e = _fitted_embedder()
        chunks = [(doc, "src", {}) for doc in DOCS]
        store = NumpyVectorStore.build_and_save(
            d, e, chunks, backend="tfidf", max_docs=2
        )
        assert store.meta.num_docs == 2


# ── Retriever ─────────────────────────────────────────────────────────────────

def _make_retriever(tmp_dir: str) -> Retriever:
    store = _build_store(tmp_dir)
    embedder = TfIdfEmbedder.from_dict(store.meta.embedder_state)
    return Retriever(store, embedder)


def test_retriever_top1_exact_match():
    with tempfile.TemporaryDirectory() as d:
        r = _make_retriever(d)
        hits = r.query("transformers use self-attention mechanisms", top_k=1)
        assert len(hits) == 1
        assert hits[0]["text_chunk"] == "transformers use self-attention mechanisms"


def test_retriever_top_k_count():
    with tempfile.TemporaryDirectory() as d:
        r = _make_retriever(d)
        hits = r.query("programming language", top_k=2)
        assert len(hits) == 2


def test_retriever_sorted_descending():
    with tempfile.TemporaryDirectory() as d:
        r = _make_retriever(d)
        hits = r.query("blue ocean sky", top_k=3)
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True)


def test_retriever_result_fields():
    with tempfile.TemporaryDirectory() as d:
        r = _make_retriever(d)
        hits = r.query("rust systems", top_k=1)
        assert "text_chunk" in hits[0]
        assert "source" in hits[0]
        assert "metadata" in hits[0]
        assert "score" in hits[0]


def test_retriever_empty_query():
    with tempfile.TemporaryDirectory() as d:
        r = _make_retriever(d)
        hits = r.query("", top_k=3)
        assert hits == []


def test_retriever_empty_store():
    e = _fitted_embedder()
    dim = e.dim
    from nanoforge.rag.store import IndexMeta
    store = NumpyVectorStore(
        vectors=np.zeros((0, dim), dtype=np.float32),
        docs=[],
        meta=IndexMeta(backend="tfidf", dim=dim, num_docs=0),
    )
    r = Retriever(store, e)
    assert r.query("anything", top_k=3) == []


# ── load_index integration ────────────────────────────────────────────────────

def test_load_index_roundtrip():
    from nanoforge.rag import load_index
    with tempfile.TemporaryDirectory() as d:
        _build_store(d)
        r = load_index(d)
        hits = r.query("ocean deep blue", top_k=1)
        assert len(hits) == 1
        assert hits[0]["score"] > 0.0


# ── Webapp no-index regression ────────────────────────────────────────────────

def test_no_index_system_is_none(monkeypatch):
    """When no RAG index is loaded, chat_stream_sse must pass system=None."""
    pytest.importorskip("fastapi")
    import nanoforge.webapp as webapp

    captured: dict = {}

    class FakeEngine:
        def chat_stream(self, messages, max_new_tokens=512, sampling=None, system=None):
            captured["system"] = system
            return iter([])

    monkeypatch.setitem(webapp._rag_holder, "retriever", None)
    monkeypatch.setitem(webapp._engine_holder, "engine", FakeEngine())

    app = webapp.create_app()
    from fastapi.testclient import TestClient
    client = TestClient(app)
    client.post("/api/chat/stream", json={"messages": [["user", "hello"]], "max_new_tokens": 16})
    assert captured.get("system") is None


def test_with_index_system_has_context(monkeypatch, tmp_path):
    """When a RAG index is loaded, system should start with 'Relevant context:'."""
    pytest.importorskip("fastapi")
    import nanoforge.webapp as webapp
    from nanoforge.rag import load_index

    _build_store(str(tmp_path))
    retriever = load_index(str(tmp_path))

    captured: dict = {}

    class FakeEngine:
        def chat_stream(self, messages, max_new_tokens=512, sampling=None, system=None):
            captured["system"] = system
            return iter([])

    monkeypatch.setitem(webapp._rag_holder, "retriever", retriever)
    monkeypatch.setitem(webapp._rag_holder, "top_k", 2)
    monkeypatch.setitem(webapp._engine_holder, "engine", FakeEngine())

    app = webapp.create_app()
    from fastapi.testclient import TestClient
    client = TestClient(app)
    client.post(
        "/api/chat/stream",
        json={"messages": [["user", "transformers attention"]], "max_new_tokens": 16},
    )
    assert captured.get("system", "").startswith("Relevant context:")
