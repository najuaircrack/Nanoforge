"""Embedding backends for RAG.

The default is ``TfIdfEmbedder`` — pure numpy, zero extra dependencies,
deterministic across processes and restarts (uses md5, not Python's hash()).
Better backends (``ModelEmbedder``, ``StEmbedder``) are opportunistic upgrades
that are only instantiated when the required library/model is available.

All embedders return float32 arrays of shape ``[N, dim]`` that are L2-normalised,
so cosine similarity reduces to a dot product at query time.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, Iterable, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from nanoforge.generation.engine import GenerationEngine


# ── Protocol ─────────────────────────────────────────────────────────────────

@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Return float32 ``[N, dim]`` L2-normalised vectors."""
        ...


# ── Helpers ───────────────────────────────────────────────────────────────────

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + 1e-9)


def _md5_hash(token: str, vocab_size: int) -> int:
    """Deterministic bucket index using md5 (not Python's hash())."""
    return int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little") % vocab_size


def _ngrams(tokens: list[str], n: int) -> Iterable[str]:
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


# ── TF-IDF (pure numpy, always available) ────────────────────────────────────

class TfIdfEmbedder:
    """Hashed n-gram TF-IDF embedder.

    Vocabulary is a fixed-size hash table so no explicit vocab-building pass is
    needed. IDF weights are fitted from a sample of documents via ``fit()``.
    The fitted state is serialisable via ``to_dict()`` / ``from_dict()`` so it
    can be stored inside the index and restored on load.
    """

    def __init__(
        self,
        vocab_size: int = 16_384,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.vocab_size = vocab_size
        self.ngram_range = ngram_range
        self.dim = vocab_size
        # IDF table — initialised to 1.0 (neutral) until fit() is called
        self._idf: np.ndarray = np.ones(vocab_size, dtype=np.float32)
        self._fitted = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, texts: Iterable[str]) -> None:
        """Compute smoothed IDF weights from ``texts``."""
        df = np.zeros(self.vocab_size, dtype=np.int64)
        n = 0
        for text in texts:
            tokens = _tokenize(text)
            seen: set[int] = set()
            for ng in range(self.ngram_range[0], self.ngram_range[1] + 1):
                for gram in _ngrams(tokens, ng):
                    b = _md5_hash(gram, self.vocab_size)
                    seen.add(b)
            for b in seen:
                df[b] += 1
            n += 1
        if n == 0:
            return
        # sklearn-style smooth IDF: log((1+N)/(1+df)) + 1
        self._idf = (np.log((1.0 + n) / (1.0 + df.astype(np.float32))) + 1.0).astype(np.float32)
        self._fitted = True

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_one(self, text: str) -> np.ndarray:
        tokens = _tokenize(text)
        if not tokens:
            return np.zeros(self.vocab_size, dtype=np.float32)
        counts = np.zeros(self.vocab_size, dtype=np.float32)
        total = 0
        for ng in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for gram in _ngrams(tokens, ng):
                counts[_md5_hash(gram, self.vocab_size)] += 1
                total += 1
        if total == 0:
            return counts
        tf = counts / total
        return tf * self._idf

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        vecs = np.stack([self._embed_one(t) for t in texts]).astype(np.float32)
        return _l2_normalize(vecs)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "ngram_range": list(self.ngram_range),
            "idf": self._idf.tolist(),
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TfIdfEmbedder":
        obj = cls(vocab_size=d["vocab_size"], ngram_range=tuple(d["ngram_range"]))
        obj._idf = np.array(d["idf"], dtype=np.float32)
        obj._fitted = d.get("fitted", True)
        return obj


# ── Model embedder (uses the loaded GenerationEngine backbone) ────────────────

class ModelEmbedder:
    """Mean-pool the backbone hidden states of a loaded ``GenerationEngine``.

    Calls ``engine.model.backbone(input_ids, caches=None, use_cache=False)``
    which returns ``(hidden [B,T,d_model], new_caches, aux_losses)``. This is
    stateless and does not touch the KV cache or generation state.
    """

    def __init__(self, engine: "GenerationEngine", batch_size: int = 8) -> None:
        self._engine = engine
        self._batch_size = batch_size
        self.dim: int = engine.model.backbone.config.d_model

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        import torch

        vecs: list[np.ndarray] = []
        with torch.inference_mode():
            for text in texts:
                ids = self._engine.tokenizer.encode(text, add_bos=True, add_eos=False)
                ids = ids[-self._engine.model.config.max_seq_len :]
                x = torch.tensor([ids], dtype=torch.long, device=self._engine.device)
                hidden, _, _ = self._engine.model.backbone(x, caches=None, use_cache=False)
                vec = hidden[0].mean(dim=0).float().cpu().numpy()
                vecs.append(vec)
        out = np.stack(vecs).astype(np.float32)
        return _l2_normalize(out)


# ── sentence-transformers backend ─────────────────────────────────────────────

class StEmbedder:
    """sentence-transformers backend. Only usable when the package is installed."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim: int = self._model.get_sentence_embedding_dimension()

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)


# ── Factory ───────────────────────────────────────────────────────────────────

def get_embedder(
    backend: str,
    engine: "GenerationEngine | None" = None,
    st_model: str = "all-MiniLM-L6-v2",
    tfidf_vocab: int = 16_384,
    tfidf_state: dict | None = None,
) -> Embedder:
    """Return an embedder for ``backend``, falling back to TF-IDF on any error.

    Parameters
    ----------
    backend:
        ``"tfidf"``, ``"model"``, or ``"sentence-transformers"``.
    engine:
        Required when ``backend="model"``.
    tfidf_state:
        Serialised ``TfIdfEmbedder`` state from a prior ``to_dict()`` call.
        When provided, the returned embedder is pre-fitted and ready to query.
    """
    if backend == "tfidf":
        if tfidf_state:
            return TfIdfEmbedder.from_dict(tfidf_state)
        return TfIdfEmbedder(vocab_size=tfidf_vocab)

    if backend == "model":
        if engine is None:
            # Graceful fallback: no engine available at load time
            if tfidf_state:
                return TfIdfEmbedder.from_dict(tfidf_state)
            return TfIdfEmbedder(vocab_size=tfidf_vocab)
        try:
            return ModelEmbedder(engine)
        except Exception:
            pass

    if backend == "sentence-transformers":
        try:
            return StEmbedder(st_model)
        except ImportError:
            pass

    # Final fallback
    if tfidf_state:
        return TfIdfEmbedder.from_dict(tfidf_state)
    return TfIdfEmbedder(vocab_size=tfidf_vocab)
