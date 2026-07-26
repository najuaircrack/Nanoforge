"""Retrieve-and-template answering for personal/factual questions.

A small chat model is often far too small to reliably answer factual questions or to
comprehend context injected into its prompt. Instead of asking it to *know* facts, we
keep a small knowledge base of question->answer facts: each fact's paraphrased
*questions* are indexed as retrieval keys, and the *answer* rides along in each
document's metadata. (See the project docs for the facts-file format and the index
builder.)

At chat time we embed the incoming user turn, retrieve the single best-matching key,
and — only if its cosine similarity clears a threshold — return that fact's answer
verbatim, bypassing the model. A miss returns ``None`` so the caller falls through to
normal model generation. Correctness is therefore a property of the facts file, not of
the model.

This module is intentionally torch-free and lazily imported (it lives in the ``rag``
package), so importing it costs nothing until a facts index is actually loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing the retriever/store at module load
    from nanoforge.rag.retriever import Retriever


class PersonalAnswerer:
    """Answer known personal questions from a facts index; ``None`` on a miss.

    Parameters
    ----------
    retriever:
        A ready :class:`nanoforge.rag.retriever.Retriever` over an index whose
        documents are question phrases carrying ``metadata["answer"]``.
    threshold:
        Minimum cosine similarity (top-1) required to return a templated answer.
        Below it, :meth:`answer` returns ``None`` and the caller should let the
        model handle the turn. Default 0.5: empirically this passes genuine
        question matches (exact keys score ~1.0, close paraphrases ~0.6+) while
        rejecting near-misses to the *wrong* fact and all non-factual chatter
        (which score well below 0.5 against the question keys).
    """

    def __init__(self, retriever: "Retriever", threshold: float = 0.5) -> None:
        self.retriever = retriever
        self.threshold = float(threshold)

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        engine=None,
        threshold: float = 0.5,
    ) -> "PersonalAnswerer | None":
        """Load a facts index and wrap it. Returns ``None`` if it can't be loaded.

        ``engine`` is only needed if the index was built with the ``model``
        embedding backend; for the default TF-IDF facts index it is ignored.
        """
        try:
            from nanoforge.rag import load_index

            retriever = load_index(index_dir, engine=engine)
        except Exception:
            return None
        return cls(retriever, threshold=threshold)

    def answer(self, user_text: str) -> str | None:
        """Return a fact answer for ``user_text`` if one clears the threshold.

        Returns ``None`` when the text is empty, retrieval finds nothing, the best
        hit is below :attr:`threshold`, or the matched fact carries no usable
        answer — in every such case the caller should defer to the model.
        """
        if not user_text or not user_text.strip():
            return None
        try:
            hits = self.retriever.query(user_text, top_k=1)
        except Exception:
            return None
        if not hits:
            return None
        top = hits[0]
        if float(top.get("score", 0.0)) < self.threshold:
            return None
        answer = (top.get("metadata") or {}).get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return None
        return answer.strip()

    def best_match(self, user_text: str) -> dict | None:
        """Return the raw top-1 hit (text_chunk/answer/score) for debugging/tuning."""
        if not user_text or not user_text.strip():
            return None
        try:
            hits = self.retriever.query(user_text, top_k=1)
        except Exception:
            return None
        return hits[0] if hits else None
