# coding: utf-8
"""Collaborative prompt-skill recommender (Phase 0 / Signal 1).

Given a query prompt, finds the most similar training examples in the
3-D scoring matrix and returns the skills (and associated fitness metrics)
that performed best on those similar examples.

Algorithm
---------
1. Embed all known ``example_input`` texts from the matrix (TF-IDF or OpenAI).
2. For the query, compute cosine similarity against every corpus text.
3. Keep only rows with similarity >= ``sim_threshold``  (default 0.25).
4. For each (skill_name, metric) pair, compute a weighted score:

       weighted_score = Σ sim_i × norm_score_i  /  Σ sim_i

   where the sum is over all matching rows for that pair.
5. Keep only (skill, metric) results with weighted_score >= ``score_threshold``
   (default 0.2) AND supported by at least ``min_examples`` similar examples.
6. Return the top-k results sorted by weighted_score, each annotated with the
   similar example snippets that drove the recommendation.
"""
from __future__ import annotations

import pandas as pd

from .embedder import Embedder, Backend
from .recommender_scores_matrix import norm_columns, metric_columns


class SkillRecommender:
    """Collaborative prompt-skill recommender.

    Fits once on the scoring matrix, then ``recommend()`` returns the top-k
    (skill, metric) pairs whose examples are most similar to the query,
    scored by similarity-weighted average of normalised metric scores.
    """

    def __init__(
        self,
        matrix_df: pd.DataFrame,
        embedder_method: Backend = "tfidf",
    ) -> None:
        """
        Parameters
        ----------
        matrix_df : pd.DataFrame
            Output of ``load_scores_matrix()``.  Must have columns
            ``example_input``, ``skill_name``, ``norm_<metric>``, …
        embedder_method : str
            ``"tfidf"`` (default, offline) or ``"openai"``.
        """
        self._df        = matrix_df.copy().reset_index(drop=True)
        self._metrics   = metric_columns(matrix_df)
        self._norm_cols = norm_columns(matrix_df)

        if not self._metrics:
            raise ValueError("No score_* columns found in matrix_df.  Check your oracle data.")

        texts = self._df["example_input"].fillna("").tolist()
        self._embedder = Embedder(method=embedder_method)
        self._embedder.fit(texts)

    def recommend(
        self,
        query: str,
        sim_threshold: float = 0.25,
        score_threshold: float = 0.20,
        min_examples: int = 1,
        top_k: int = 10,
        top_examples: int = 3,
    ) -> list[dict]:
        """Return collaborative skill recommendations for *query*.

        Parameters
        ----------
        query : str
            The prompt to find matching skills for.
        sim_threshold : float
            Minimum cosine similarity to include an example row  [0, 1].
        score_threshold : float
            Minimum score for a result to be returned.
        min_examples : int
            Minimum number of similar examples required to recommend a skill.
        top_k : int
            Maximum number of results returned.
        top_examples : int
            Number of similar example snippets to include per result (default 3).

        Returns
        -------
        list[dict], sorted by score descending::

            [
              {
                "skill":    "smarthub-support",
                "metric":   "bag_of_words",
                "score":    0.61,          # collaborative score
                "collaborative_score": 0.58,
                "n_examples": 3,
                "similar_examples": [...],
              },
              ...
            ]
        """
        similarities = self._embedder.similarities(query)

        mask = similarities >= sim_threshold
        if not mask.any():
            return []

        sim_vals = similarities[mask]
        sub_df   = self._df[mask].copy()
        sub_df["_sim"] = sim_vals

        results: list[dict] = []

        for skill_name, skill_df in sub_df.groupby("skill_name"):
            sim_weights = skill_df["_sim"].values
            total_sim   = sim_weights.sum()

            skill_df_sorted = skill_df.sort_values("_sim", ascending=False)
            snippets = []
            for _, row in skill_df_sorted.head(top_examples).iterrows():
                snippets.append({
                    "input":      str(row.get("example_input",    "") or "")[:160],
                    "expected":   str(row.get("example_expected", "") or "")[:120],
                    "output":     str(row.get("candidate_output", "") or "")[:160],
                    "similarity": round(float(row["_sim"]), 4),
                })

            for metric in self._metrics:
                norm_col = f"norm_{metric}"
                if norm_col not in skill_df.columns:
                    continue
                raw_norm = skill_df[norm_col]
                if raw_norm.isna().all():
                    continue
                norm_vals = raw_norm.fillna(0).values
                if total_sim == 0:
                    continue

                score = float((sim_weights * norm_vals).sum() / total_sim)

                if score < score_threshold:
                    continue
                if len(skill_df) < min_examples:
                    continue

                results.append({
                    "skill":                str(skill_name),
                    "metric":               metric,
                    "score":                round(score, 4),
                    "collaborative_score":  round(score, 4),
                    "n_examples":           int(len(skill_df)),
                    "mean_similarity":      round(float(sim_weights.mean()), 4),
                    "similar_examples":     snippets,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    @property
    def n_examples(self) -> int:
        """Total number of (skill, example) pairs in the matrix."""
        return len(self._df)

    @property
    def skills(self) -> list[str]:
        """All unique skill names in the matrix."""
        return sorted(self._df["skill_name"].unique().tolist())

    @property
    def metrics(self) -> list[str]:
        """All metric names found in the matrix."""
        return list(self._metrics)
