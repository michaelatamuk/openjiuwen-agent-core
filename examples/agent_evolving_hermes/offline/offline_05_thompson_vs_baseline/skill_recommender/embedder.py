# coding: utf-8
"""Prompt embedder for similarity search.

Two backends are supported:

  "tfidf"  (default) — TF-IDF vectorizer + cosine similarity.
      No additional dependencies; works fully offline.
      Less semantic, but fast and reproducible.

  "openai" — OpenAI text-embedding-3-small via the openai SDK.
      Requires OPENAI_API_KEY in the environment.
      Semantically richer; makes API calls (costs money, needs network).

Usage
-----
    from skill_recommender.embedder import Embedder

    emb = Embedder(method="tfidf")
    emb.fit(["How do I reset my router?", "Invoice payment instructions", ...])

    similarities = emb.similarities("My router keeps dropping connections")
    # → np.ndarray of shape (n_corpus_texts,) with cosine similarities in [0, 1]
"""
from __future__ import annotations

import os
from typing import Literal

import numpy as np

Backend = Literal["tfidf", "openai"]


class Embedder:
    """Fits on a corpus of texts and computes cosine similarities for queries."""

    def __init__(self, method: Backend = "tfidf") -> None:
        self._method  = method
        self._fitted  = False
        self._corpus: list[str] = []

        if method == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10_000,
                sublinear_tf=True,
            )
            self._matrix = None          # sparse, set after fit
        elif method == "openai":
            try:
                import openai  # noqa: F401
            except ImportError:
                raise ImportError(
                    "openai package is required for the 'openai' embedder backend.\n"
                    "Install it with: pip install openai"
                ) from None
            self._openai_model  = "text-embedding-3-small"
            self._corpus_vecs: np.ndarray | None = None
        else:
            raise ValueError(f"method must be 'tfidf' or 'openai'; got {method!r}")

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, texts: list[str]) -> "Embedder":
        """Fit the embedder on *texts* (the known example_input corpus)."""
        self._corpus = [t or "" for t in texts]

        if self._method == "tfidf":
            self._matrix = self._vectorizer.fit_transform(self._corpus)
        else:
            self._corpus_vecs = self._embed_openai(self._corpus)

        self._fitted = True
        return self

    # ── Similarities ─────────────────────────────────────────────────────────

    def similarities(self, query: str) -> np.ndarray:
        """Return cosine similarities between *query* and every corpus text.

        Returns
        -------
        np.ndarray of shape (n_corpus_texts,), values in [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before similarities().")

        if self._method == "tfidf":
            return self._similarities_tfidf(query)
        else:
            return self._similarities_openai(query)

    # ── TF-IDF backend ────────────────────────────────────────────────────────

    def _similarities_tfidf(self, query: str) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self._vectorizer.transform([query])
        sims  = cosine_similarity(q_vec, self._matrix).flatten()
        return sims.astype(float)

    # ── OpenAI backend ────────────────────────────────────────────────────────

    def _embed_openai(self, texts: list[str]) -> np.ndarray:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        client  = openai.OpenAI(api_key=api_key)

        # Batch to stay within API limits
        batch_size = 512
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp  = client.embeddings.create(model=self._openai_model, input=batch)
            all_vecs.extend(item.embedding for item in resp.data)

        mat = np.array(all_vecs, dtype=float)
        # L2-normalise so dot product == cosine similarity
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def _similarities_openai(self, query: str) -> np.ndarray:
        q_vec = self._embed_openai([query])[0]  # shape (dim,)
        return (self._corpus_vecs @ q_vec).astype(float)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path) -> None:
        """Persist the fitted embedder to *path* (pickle)."""
        import pickle
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path) -> "Embedder":
        """Load a previously saved embedder from *path*."""
        import pickle
        from pathlib import Path
        with open(Path(path), "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected Embedder, got {type(obj)}")
        return obj
