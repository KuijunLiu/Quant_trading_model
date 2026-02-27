import pandas as pd
import numpy as np


def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing for monthly CRSP-style stock data.

    Assumes columns include (at least): date, permno, ret, prc, shrout.
    - Parses dates and numeric fields
    - Fixes negative prices, computes market cap
    - Drops invalid dates and duplicate (date, permno) rows
    - Filters out non-positive prices/shares
    - Drops rows with missing returns
    - Sorts by permno and date

    Returns a new cleaned DataFrame.
    """
    df_clean = df.copy()

    # Parse types
    df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
    df_clean["prc"] = pd.to_numeric(df_clean["prc"], errors="coerce")
    df_clean["shrout"] = pd.to_numeric(df_clean["shrout"], errors="coerce")
    df_clean["ret"] = pd.to_numeric(df_clean["ret"], errors="coerce")

    # Drop invalid dates
    df_clean = df_clean.dropna(subset=["date"])

    # Fix negative prices (CRSP convention) and compute market cap
    df_clean["prc"] = df_clean["prc"].abs()
    df_clean["mkt_cap"] = df_clean["prc"] * df_clean["shrout"]

    # Drop rows with non-positive or missing price/shares/market cap
    df_clean = df_clean[
        (df_clean["prc"] > 0)
        & (df_clean["shrout"] > 0)
        & df_clean["mkt_cap"].notna()
    ]

    # Drop rows with missing returns (safer for backtests)
    df_clean = df_clean.dropna(subset=["ret"])

    # Ensure uniqueness at (date, permno) level
    if {"date", "permno"}.issubset(df_clean.columns):
        df_clean = df_clean.drop_duplicates(subset=["date", "permno"], keep="first")

    # Sort for panel / time-series work
    sort_cols = [c for c in ["permno", "date"] if c in df_clean.columns]
    if sort_cols:
        df_clean = df_clean.sort_values(sort_cols).reset_index(drop=True)

    return df_clean