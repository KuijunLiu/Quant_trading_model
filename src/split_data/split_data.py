"""
split_data.py
=============
Splits the CRSP monthly feature panel into train / validation / test sets
using a strict time-based split to prevent look-ahead bias.

Also provides walk-forward (expanding window) cross-validation for
hyperparameter tuning on time-series data.

Default date boundaries (simple split)
---------------------------------------
  Train      : start        –  2015-12-31
  Validation : 2016-01-01   –  2019-12-31
  Test       : 2020-01-01   –  end

Input
-----
  data/processed/crsp_monthly_features.csv  (or .parquet)

Output (simple split)
---------------------
  data/processed/train.csv
  data/processed/validation.csv
  data/processed/test.csv

Usage
-----
  # Simple split
  python -m src.data.split_data
  python -m src.data.split_data --val-start 2016-01-01 --test-start 2020-01-01

  # Walk-forward CV (library use)
  from src.data.split_data import walk_forward_cv
  for fold, train, val in walk_forward_cv(df, initial_train_years=5, val_years=1):
      ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Generator

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "processed" / "crsp_monthly_features.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed"

DEFAULT_VAL_START  = "2016-01-01"
DEFAULT_TEST_START = "2020-01-01"


# ------------------------------------------------------------------ #
# I/O helpers                                                          #
# ------------------------------------------------------------------ #

def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"  Saved {len(df):>10,} rows  →  {path}")


# ------------------------------------------------------------------ #
# Simple train / validation / test split                               #
# ------------------------------------------------------------------ #

def split_data(
    df: pd.DataFrame,
    val_start: str = DEFAULT_VAL_START,
    test_start: str = DEFAULT_TEST_START,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, validation, test) splits based on the 'date' column.

    The split is strictly chronological:
      train      : date <  val_start
      validation : val_start  <= date <  test_start
      test       : date >= test_start
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    val_start_dt  = pd.Timestamp(val_start)
    test_start_dt = pd.Timestamp(test_start)

    train = df[df["date"] < val_start_dt].reset_index(drop=True)
    val   = df[(df["date"] >= val_start_dt) & (df["date"] < test_start_dt)].reset_index(drop=True)
    test  = df[df["date"] >= test_start_dt].reset_index(drop=True)

    return train, val, test


# ------------------------------------------------------------------ #
# Walk-forward (expanding window) cross-validation                     #
# ------------------------------------------------------------------ #

def walk_forward_cv(
    df: pd.DataFrame,
    initial_train_years: int = 5,
    val_years: int = 1,
    step_years: int = 1,
    gap_months: int = 1,
) -> Generator[tuple[int, pd.DataFrame, pd.DataFrame], None, None]:
    """Walk-forward (expanding window) cross-validation for panel time series.

    Each fold expands the training set by `step_years` and slides the
    validation window forward by the same amount. A `gap_months` buffer
    is inserted between train end and val start to avoid short-term leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with a 'date' column.
    initial_train_years : int
        Number of years in the first training window.
    val_years : int
        Number of years in each validation window.
    step_years : int
        Number of years to advance both windows per fold.
    gap_months : int
        Months of data to skip between train end and val start.

    Yields
    ------
    fold_index : int
    train_df   : pd.DataFrame
    val_df     : pd.DataFrame

    Example
    -------
    Assuming data spans 1994–2024 with initial_train_years=5:

      Fold  0 | Train: 1994-01 → 1998-12 | Val: 1999-02 → 2000-01
      Fold  1 | Train: 1994-01 → 1999-12 | Val: 2000-02 → 2001-01
      Fold  2 | Train: 1994-01 → 2000-12 | Val: 2001-02 → 2002-01
      ...
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    min_date = df["date"].min()
    max_date = df["date"].max()

    train_end = min_date + pd.DateOffset(years=initial_train_years) - pd.DateOffset(days=1)
    fold = 0

    while True:
        val_start = train_end + pd.DateOffset(months=gap_months)
        val_end   = val_start + pd.DateOffset(years=val_years) - pd.DateOffset(days=1)

        # Stop when the validation window starts beyond available data.
        if val_start > max_date:
            break

        # Clip val_end to the last available date if the window overshoots.
        val_end = min(val_end, max_date)

        train = df[df["date"] <= train_end].reset_index(drop=True)
        val   = df[(df["date"] >= val_start) & (df["date"] <= val_end)].reset_index(drop=True)

        if len(train) == 0 or len(val) == 0:
            break

        print(
            f"Fold {fold:>2d} | "
            f"Train: {train['date'].min().date()} → {train['date'].max().date()} "
            f"({len(train):,} rows) | "
            f"Val: {val['date'].min().date()} → {val['date'].max().date()} "
            f"({len(val):,} rows)"
        )

        yield fold, train, val

        fold      += 1
        train_end  = train_end + pd.DateOffset(years=step_years)


# ------------------------------------------------------------------ #
# CLI entry point (simple split only)                                  #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time-based train/validation/test split for the CRSP feature panel."
    )
    parser.add_argument("--input",      default=str(DEFAULT_INPUT_PATH),
                        help="Path to crsp_monthly_features.csv / .parquet")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory where split files will be written.")
    parser.add_argument("--val-start",  default=DEFAULT_VAL_START,
                        help="First date of the validation set  (default: 2016-01-01)")
    parser.add_argument("--test-start", default=DEFAULT_TEST_START,
                        help="First date of the test set        (default: 2020-01-01)")
    parser.add_argument("--fmt", choices=["csv", "parquet"], default="csv",
                        help="Output file format (default: csv)")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    ext = f".{args.fmt}"

    print(f"Loading features from: {input_path}")
    df = _load_table(input_path)
    print(f"Total rows: {len(df):,}")

    train, val, test = split_data(df, val_start=args.val_start, test_start=args.test_start)

    print(f"\nSplit boundaries:")
    print(f"  Train      : start  →  before {args.val_start}   ({len(train):,} rows)")
    print(f"  Validation : {args.val_start}  →  before {args.test_start}  ({len(val):,} rows)")
    print(f"  Test       : {args.test_start}  →  end             ({len(test):,} rows)")
    print()

    _save_table(train, output_dir / f"train{ext}")
    _save_table(val,   output_dir / f"validation{ext}")
    _save_table(test,  output_dir / f"test{ext}")

    print("\nDone.")


if __name__ == "__main__":
    main()
