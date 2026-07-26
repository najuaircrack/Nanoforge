"""Cosine top-k retrieval over a loaded :class:`NumpyVectorStore`.

All stored vectors and query vectors are L2-normalised, so cosine similarity is
just a dot product. Top-k selection uses ``np.argpartition`` (O(N)) then sorts
only the k winners.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nanoforge.rag.embeddings import Embedder
    from nanoforge.rag.store import NumpyVectorStore


class Retriever:
    def __init__(self, store: "NumpyVectorStore", embedder: "Embedder") -> None:
        self.store = store
        self.embedder = embedder

    def query(self, text: str, top_k: int = 3) -> list[dict]:
        """Return the ``top_k`` most similar documents to ``text``.

        Each result is ``{text_chunk, source, metadata, score}`` sorted by
        descending cosine similarity. Returns ``[]`` for an empty index.
        """
        vectors = self.store.vectors
        if vectors.shape[0] == 0 or not text.strip():
            return []

        q = self.embedder.embed_batch([text])[0].astype(np.float32)  # [dim]
        scores = vectors @ q  # [N]

        k = min(top_k, scores.shape[0])
        # O(N) partial selection of the top-k, then sort those k descending.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        results: list[dict] = []
        for i in top_idx:
            doc = self.store.docs[int(i)]
            results.append(
                {
                    "text_chunk": doc.get("text_chunk", ""),
                    "source": doc.get("source", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": float(scores[int(i)]),
                }
            )
        return results
