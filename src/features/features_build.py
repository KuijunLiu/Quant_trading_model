from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


"""
features_build.py
=================
Builds lagged price and return features from the processed CRSP monthly stock data.

Input
-----
data/processed/crsp_monthly_processed.csv  (or .parquet)
    Clean CRSP monthly panel with at minimum the columns:
    date, permno, ret, prc, shrout.
    Optional enriching columns: mkt_cap, cfacpr, vol, comnam.

Output
------
data/processed/crsp_monthly_features.csv  (or .parquet)
    Same panel with the following feature columns appended:

    | Column             | Description                                              |
    |--------------------|----------------------------------------------------------|
    | adj_prc            | Split-adjusted closing price (prc / cfacpr)              |
    | turnover           | Monthly share turnover (vol / shrout)                    |
    | log_mkt_cap        | Natural log of market capitalisation                     |
    | ret_lag_{1,2,3}    | Return lagged 1, 2, 3 months                             |
    | mom_2_6            | Cumulative return over months t-6 to t-2 (skip-1 mom.)  |
    | mom_2_12           | Cumulative return over months t-12 to t-2                |
    | vol_{3,6,12}m      | Rolling return std-dev over 3, 6, 12 months              |
    | price_ma_6_ratio   | Lagged price / 6-month MA − 1                            |
    | price_ma_12_ratio  | Lagged price / 12-month MA − 1                           |
    | turnover_{3,12}m   | Rolling mean turnover over 3, 12 months                  |
    | ret_fwd_1m         | Forward 1-month return (prediction target / label)       |
    | {feature}_cs       | Cross-sectionally standardized version of each feature   |

Design notes
------------
* All rolling and lag operations are computed **within each permno group** so
  that month boundaries never cross between different stocks.
* Every feature is shifted by at least 1 period relative to the current row to
  prevent look-ahead bias (no future information leaks into the feature set).
* The script can be run directly from the command line:
      python -m src.features.features_build --input <path> --output <path>
  or imported as a library via `build_features(df)`.
"""


# `Path(__file__)` refers to the path of this Python file itself.
# `resolve()` converts it to an absolute path.
# `parents[2]` moves up two directory levels:
# src/features/features_build.py -> src/features -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
# The `/` operator on `Path` objects joins path segments and handles
# OS-specific separators automatically, which is cleaner than string concatenation.
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "processed" / "crsp_monthly_processed.csv"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "crsp_monthly_features.csv"
REQUIRED_COLUMNS = {"date", "permno", "ret", "prc", "shrout"}


def _resolve_path(path_str: str | None, default_path: Path) -> Path:
    # Convert the string to a `Path` if the user provided one; otherwise fall back to the default.
    # `expanduser()` expands a leading `~` to the user's home directory.
    path = Path(path_str).expanduser() if path_str else default_path
    # If the path is relative (e.g. `data/processed/a.csv`), interpret it relative to the repo root.
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_table(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".parquet":
        return pd.read_parquet(input_path)
    return pd.read_csv(input_path)


def _save_table(df: pd.DataFrame, output_path: Path) -> None:
    # `parent` is the directory containing the output file;
    # `mkdir(..., exist_ok=True)` creates it if it does not already exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


def _group_shift(series: pd.Series, groups: pd.Series, periods: int) -> pd.Series:
    # Shift within each stock group (permno).
    # With periods=1, each row receives the value from the *previous* month
    # for that stock, not simply the previous row in the full table.
    return series.groupby(groups, sort=False).shift(periods)


def _group_rolling(
    series: pd.Series,
    groups: pd.Series,
    shift_periods: int,
    window: int,
    metric: str,
    min_periods: int | None = None,
) -> pd.Series:
    # Lag the series before rolling to ensure the current month's data does not
    # leak into the current month's feature — the key guard against look-ahead bias.
    lagged = _group_shift(series, groups, shift_periods)
    rolling = lagged.groupby(groups, sort=False).rolling(
        window=window,
        min_periods=min_periods or window,
    )

    if metric == "mean":
        result = rolling.mean()
    elif metric == "std":
        result = rolling.std()
    elif metric == "sum":
        result = rolling.sum()
    else:
        raise ValueError(f"Unsupported rolling metric: {metric}")

    return result.reset_index(level=0, drop=True)


def _momentum_feature(
    returns: pd.Series,
    groups: pd.Series,
    start_lag: int,
    end_lag: int,
) -> pd.Series:
    if end_lag < start_lag:
        raise ValueError("end_lag must be greater than or equal to start_lag")

    # Monthly returns must be greater than -100 %; otherwise log1p is undefined.
    safe_returns = returns.where(returns > -1)
    # Summing log returns is numerically more stable than chaining (1+r) multiplications.
    log_returns = np.log1p(safe_returns)
    window = end_lag - start_lag + 1
    rolling_log_sum = _group_rolling(
        log_returns,
        groups,
        shift_periods=start_lag,
        window=window,
        metric="sum",
    )
    return np.expm1(rolling_log_sum)


# Feature columns that will be winsorized and cross-sectionally standardized.
# The label (ret_fwd_1m) and non-numeric identifier columns are intentionally excluded.
FEATURE_COLUMNS = [
    "ret_lag_1", "ret_lag_2", "ret_lag_3",
    "mom_2_6", "mom_2_12",
    "vol_3m", "vol_6m", "vol_12m",
    "price_ma_6_ratio", "price_ma_12_ratio",
    "log_mkt_cap",
    "turnover", "turnover_3m", "turnover_12m",
]


def _winsorize_cross_section(
    df: pd.DataFrame,
    columns: list[str],
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Winsorize each feature within each cross-section (date).

    Values below the `lower` quantile or above the `upper` quantile for a
    given month are clipped to those boundary values, preventing extreme
    outliers from distorting downstream models.
    """
    def _clip(group: pd.DataFrame) -> pd.DataFrame:
        for col in columns:
            if col not in group.columns:
                continue
            lo = group[col].quantile(lower)
            hi = group[col].quantile(upper)
            group[col] = group[col].clip(lo, hi)
        return group

    return df.groupby("date", group_keys=False).apply(_clip)


def _standardize_cross_section(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Z-score each feature within each cross-section (date).

    After winsorization, subtract the cross-sectional mean and divide by the
    cross-sectional standard deviation so that all features share a comparable
    scale across months.  Resulting columns are suffixed with `_cs`.
    """
    def _zscore(group: pd.DataFrame) -> pd.DataFrame:
        for col in columns:
            if col not in group.columns:
                continue
            mu = group[col].mean()
            sigma = group[col].std()
            # Avoid division by zero when all values in a cross-section are identical.
            group[f"{col}_cs"] = (group[col] - mu) / sigma if sigma > 0 else 0.0
        return group

    return df.groupby("date", group_keys=False).apply(_zscore)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    # Work on a copy to avoid mutating the caller's DataFrame.
    features = df.copy()

    if "mkt_cap" in features.columns:
        features["mkt_cap"] = pd.to_numeric(features["mkt_cap"], errors="coerce")
    else:
        # Compute market cap on the fly if the pre-processing stage did not retain it.
        features["mkt_cap"] = features["prc"] * features["shrout"]

    features = features.dropna(subset=["date", "permno", "prc", "shrout", "ret"])

    # Cache the grouping key — reused by every lag/rolling call below.
    groups = features["permno"]

    if "cfacpr" in features.columns:
        cfacpr = pd.to_numeric(features["cfacpr"], errors="coerce").replace(0, np.nan)
        # Divide by CRSP's cumulative price adjustment factor to obtain split-adjusted prices,
        # reducing distortions caused by stock splits and stock dividends.
        features["adj_prc"] = features["prc"] / cfacpr
    else:
        features["adj_prc"] = features["prc"]

    if "vol" in features.columns:
        volume = pd.to_numeric(features["vol"], errors="coerce")
        # Turnover approximates "shares traded this month / shares outstanding" —
        # a simple proxy for liquidity.
        features["turnover"] = volume / features["shrout"]
    else:
        features["turnover"] = np.nan

    # Log-transforming market cap makes its distribution more symmetric,
    # which is better suited for cross-sectional modelling.
    features["log_mkt_cap"] = np.log(features["mkt_cap"].where(features["mkt_cap"] > 0))
    features["ret_lag_1"] = _group_shift(features["ret"], groups, 1)
    features["ret_lag_2"] = _group_shift(features["ret"], groups, 2)
    features["ret_lag_3"] = _group_shift(features["ret"], groups, 3)

    # Skip the most recent month when computing momentum to avoid short-term reversal noise.
    features["mom_2_6"] = _momentum_feature(features["ret"], groups, start_lag=2, end_lag=6)
    features["mom_2_12"] = _momentum_feature(features["ret"], groups, start_lag=2, end_lag=12)

    # Volatility is measured as the rolling standard deviation of past returns.
    features["vol_3m"] = _group_rolling(features["ret"], groups, shift_periods=1, window=3, metric="std")
    features["vol_6m"] = _group_rolling(features["ret"], groups, shift_periods=1, window=6, metric="std")
    features["vol_12m"] = _group_rolling(features["ret"], groups, shift_periods=1, window=12, metric="std")

    ma_6 = _group_rolling(features["adj_prc"], groups, shift_periods=1, window=6, metric="mean")
    ma_12 = _group_rolling(features["adj_prc"], groups, shift_periods=1, window=12, metric="mean")
    lagged_price = _group_shift(features["adj_prc"], groups, 1)
    # Position of the last available price relative to its moving average —
    # a simple proxy for trend strength.
    features["price_ma_6_ratio"] = lagged_price / ma_6 - 1
    features["price_ma_12_ratio"] = lagged_price / ma_12 - 1

    features["turnover_3m"] = _group_rolling(
        features["turnover"],
        groups,
        shift_periods=1,
        window=3,
        metric="mean",
    )
    features["turnover_12m"] = _group_rolling(
        features["turnover"],
        groups,
        shift_periods=1,
        window=12,
        metric="mean",
    )

    # Forward 1-month return used as the supervised learning label / prediction target.
    features["ret_fwd_1m"] = _group_shift(features["ret"], groups, -1)

    # ------------------------------------------------------------------ #
    # Cross-sectional winsorization (clip outliers at 1 % / 99 %)         #
    # ------------------------------------------------------------------ #
    valid_feature_cols = [c for c in FEATURE_COLUMNS if c in features.columns]
    features = _winsorize_cross_section(features, valid_feature_cols)

    # ------------------------------------------------------------------ #
    # Cross-sectional z-score standardization (suffix: _cs)               #
    # ------------------------------------------------------------------ #
    features = _standardize_cross_section(features, valid_feature_cols)

    # Place the most important columns first for easier inspection in notebooks.
    ordered_columns = [
        "date",
        "permno",
        "comnam",
        "ret",
        "ret_fwd_1m",
        "prc",
        "adj_prc",
        "shrout",
        "mkt_cap",
        "log_mkt_cap",
        "ret_lag_1",
        "ret_lag_2",
        "ret_lag_3",
        "mom_2_6",
        "mom_2_12",
        "vol_3m",
        "vol_6m",
        "vol_12m",
        "price_ma_6_ratio",
        "price_ma_12_ratio",
        "turnover",
        "turnover_3m",
        "turnover_12m",
    ]
    remaining_columns = [column for column in features.columns if column not in ordered_columns]
    ordered_columns = [column for column in ordered_columns if column in features.columns] + remaining_columns

    return features.loc[:, ordered_columns]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build lagged CRSP monthly price features from data/processed."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Processed CRSP input file (.csv or .parquet).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Feature output file (.csv or .parquet).",
    )
    args = parser.parse_args()

    # When no CLI arguments are supplied, fall back to the default processed paths.
    input_path = _resolve_path(args.input, DEFAULT_INPUT_PATH)
    output_path = _resolve_path(args.output, DEFAULT_OUTPUT_PATH)

    raw_df = _load_table(input_path)
    feature_df = build_features(raw_df)
    _save_table(feature_df, output_path)

    print(f"Loaded {len(raw_df):,} rows from {input_path}")
    print(f"Saved {len(feature_df):,} rows with features to {output_path}")


if __name__ == "__main__":
    main()
