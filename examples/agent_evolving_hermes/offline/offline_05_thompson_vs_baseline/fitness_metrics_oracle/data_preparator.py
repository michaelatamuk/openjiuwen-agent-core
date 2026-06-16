import pandas as pd

from .data_features import NUM_COLS, TARGET, TEXT_COL, GROUP_COL


def _fill_numeric(df: pd.DataFrame) -> None:
    """Fill missing numeric feature columns with 0 in-place."""
    for col in NUM_COLS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, feature-ready DataFrame with only rows that have a valid label."""
    df = df.copy()
    df = df[df[TARGET].notna()].copy()
    _fill_numeric(df)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df["model"]   = df["model"].fillna("unknown").astype(str)
    df["metric"]  = df["metric"].fillna("unknown").astype(str)
    if GROUP_COL not in df.columns:
        df[GROUP_COL] = "unknown_skill"
    return df.reset_index(drop=True)

