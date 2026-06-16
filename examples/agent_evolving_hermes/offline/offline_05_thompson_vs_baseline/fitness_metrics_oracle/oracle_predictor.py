import pickle
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from .data_features import ALL_FEATURE_COLS
from .data_preparator import _fill_numeric


class OraclePredictor:
    """Wraps a trained sklearn Pipeline and exposes a simple predict() interface.

    Intended to be pickled with ``save()`` and restored with ``load()``.
    """

    def __init__(self, pipeline: Any, metrics_seen: list[str], feature_meta: dict) -> None:
        self._pipeline = pipeline
        self._metrics_seen = metrics_seen
        self._feature_meta = feature_meta  # diagnostics / provenance

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, skill_metadata: dict, candidate_metrics: list[str], model: str = "") -> list[dict]:
        """Rank candidate_metrics by predicted improvement (best first).

        Parameters
        ----------
        skill_metadata : dict
            A-priori skill features. Recognised keys:
              description, n_examples_trainset, n_examples_valset,
              n_examples_holdout, baseline_skill_chars, baseline_skill_body_chars
        candidate_metrics : list[str]
            Metric names to rank (e.g. ["bag_of_words", "f1", "graph"]).
        model : str
            LLM model string (e.g. "deepseek/deepseek-chat").
            Use "" or omit if unknown — the model will fall back to the
            closest seen value via OneHotEncoder's handle_unknown="ignore".

        Returns
        -------
        list[dict]
            Sorted best-first:
            [{"metric": str, "predicted_improvement": float}, ...]
        """
        rows = [{"description":               skill_metadata.get("description", ""),
                 "n_examples_trainset":       skill_metadata.get("n_examples_trainset") or 0,
                 "n_examples_valset":         skill_metadata.get("n_examples_valset") or 0,
                 "n_examples_holdout":        skill_metadata.get("n_examples_holdout") or 0,
                 "baseline_skill_chars":      skill_metadata.get("baseline_skill_chars") or 0,
                 "baseline_skill_body_chars": skill_metadata.get("baseline_skill_body_chars") or 0,
                 "metric": m,
                 "model":  model or "unknown"}
                for m in candidate_metrics
                ]
        df = pd.DataFrame(rows)
        _fill_numeric(df)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = self._pipeline.predict(df[ALL_FEATURE_COLS])

        results = [{"metric": m, "predicted_improvement": float(p)}
                   for m, p in zip(candidate_metrics, preds)]
        results.sort(key=lambda x: x["predicted_improvement"], reverse=True)
        return results

    @property
    def metrics_seen(self) -> list[str]:
        """Metric names the oracle was trained on."""
        return list(self._metrics_seen)

    @property
    def meta(self) -> dict:
        """Provenance info: n_runs, n_skills, CV stats, etc."""
        return dict(self._feature_meta)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Serialise this predictor to *path* via pickle."""
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        print(f"  Saved: {path}")

    @classmethod
    def load(cls, path: Path | str) -> "OraclePredictor":
        """Load a previously saved OraclePredictor."""
        path = Path(path).expanduser()
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected OraclePredictor, got {type(obj).__name__}")
        return obj
