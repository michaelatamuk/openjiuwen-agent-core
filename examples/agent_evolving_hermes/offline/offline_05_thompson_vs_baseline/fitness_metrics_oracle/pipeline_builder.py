from __future__ import annotations

from .data_features import NUM_COLS, TEXT_COL, CAT_MODEL, CAT_METRIC

# ── scikit-learn import guard ──────────────────────────────────────────────────

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ── Model construction ─────────────────────────────────────────────────────────

def _build_pipeline(n_runs: int) -> Pipeline:
    """Return an unfitted sklearn Pipeline.

    Uses lighter settings when data is scarce (< 20 runs) to reduce overfitting.
    """
    small        = n_runs < 20
    n_estimators = 30  if small else 100
    max_depth    = 2   if small else 3
    # TruncatedSVD requires n_components < n_samples; use floor(n_runs/2) so
    # that even the smallest training fold (≈half the data) stays above n_svd.
    n_svd = max(1, min(8, n_runs // 2))

    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), NUM_COLS),
                                                   ("metric_ohe", OneHotEncoder(handle_unknown="ignore",
                                                                                sparse_output=False), CAT_METRIC),
                                                   ("model_ohe", OneHotEncoder(handle_unknown="ignore",
                                                                               sparse_output=False), CAT_MODEL),
                                                   ("desc_text", Pipeline([("tfidf", TfidfVectorizer(max_features=200,
                                                                                                     ngram_range=(1, 2))
                                                                            ),
                                                                           ("svd",   TruncatedSVD(n_components=n_svd,
                                                                                                  random_state=42)),]
                                                                          ), TEXT_COL,  # single column name → passed as 1-D array to TfidfVectorizer
                                                    )],
                                     sparse_threshold=0,  # always produce dense output
                                    )

    return Pipeline([("prep", preprocessor),
                     ("gbr",  GradientBoostingRegressor(n_estimators=n_estimators,
                                                        max_depth=max_depth,
                                                        learning_rate=0.05,
                                                        subsample=0.8,
                                                        random_state=42)),
                     ])
