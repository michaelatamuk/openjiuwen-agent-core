# coding: utf-8
"""Factory function for building a Phase-0 SkillRecommender from disk."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from .embedder import Backend
from .recommender_scores_matrix import load_scores_matrix
from .recommender import SkillRecommender


def build_recommender(
    oracle_dir: Path | str,
    variant: Literal["baseline", "evolved", "both"] = "baseline",
    embedder_method: Backend = "tfidf",
) -> SkillRecommender:
    """Load matrix from *oracle_dir* and return a collaborative ``SkillRecommender``.

    Parameters
    ----------
    oracle_dir : Path | str
        Directory with ``scoring_matrix_*.json`` files.
    variant : str
        Which layer of the 3-D matrix to use: ``"baseline"``, ``"evolved"``,
        or ``"both"``.
    embedder_method : str
        ``"tfidf"`` (default) or ``"openai"``.
    """
    df = load_scores_matrix(Path(oracle_dir).expanduser(), variant=variant)
    return SkillRecommender(df, embedder_method=embedder_method)
