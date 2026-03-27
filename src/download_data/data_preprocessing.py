from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
# Try the common raw-data filenames used in this repo before requiring --input.
DEFAULT_INPUT_CANDIDATES = (
    REPO_ROOT / "data" / "raw" / "crsp_monthly.csv",
    REPO_ROOT / "data" / "raw" / "msf.parquet",
    REPO_ROOT / "data" / "raw" / "msf.csv",
)
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "crsp_monthly_processed.csv"


def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean monthly CRSP-style stock data for downstream feature building.

    Required columns: date, prc, shrout, ret.
    """
    required_columns = {"date", "prc", "shrout", "ret"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    df_clean = df.copy()

    # Parse core fields into stable numeric/datetime types before filtering.
    df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
    df_clean["prc"] = pd.to_numeric(df_clean["prc"], errors="coerce")
    df_clean["shrout"] = pd.to_numeric(df_clean["shrout"], errors="coerce")
    df_clean["ret"] = pd.to_numeric(df_clean["ret"], errors="coerce")

    # Drop rows where the observation date cannot be interpreted.
    df_clean = df_clean.dropna(subset=["date"])

    # CRSP prices can be negative as a quote convention.
    df_clean["prc"] = df_clean["prc"].abs()
    # Market cap is needed by most downstream portfolio filters and features.
    df_clean["mkt_cap"] = df_clean["prc"] * df_clean["shrout"]

    # Remove rows with invalid price/share information after standardization.
    df_clean = df_clean[
        (df_clean["prc"] > 0)
        & (df_clean["shrout"] > 0)
        & df_clean["mkt_cap"].notna()
    ]
    # Missing returns are usually unusable for signal construction/backtests.
    df_clean = df_clean.dropna(subset=["ret"])

    if {"date", "permno"}.issubset(df_clean.columns):
        # Keep one row per stock-month observation.
        df_clean = df_clean.drop_duplicates(subset=["date", "permno"], keep="first")

    # Sort the panel so later rolling/grouped operations behave predictably.
    sort_cols = [column for column in ("permno", "date") if column in df_clean.columns]
    if sort_cols:
        df_clean = df_clean.sort_values(sort_cols)

    return df_clean.reset_index(drop=True)


def _resolve_input_path(input_path: str | None) -> Path:
    if input_path:
        resolved_path = Path(input_path).expanduser()
        if not resolved_path.is_absolute():
            # Treat relative paths as relative to the repository root.
            resolved_path = REPO_ROOT / resolved_path
        if not resolved_path.exists():
            raise FileNotFoundError(f"Input file not found: {resolved_path}")
        return resolved_path

    # Fall back to the project's standard raw-data locations.
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate

    candidates = "\n".join(f"- {path}" for path in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(
        "No raw CRSP file found. Checked:\n"
        f"{candidates}"
    )


def _load_crsp_data(input_path: Path) -> pd.DataFrame:
    # Match the loader to the on-disk file format.
    if input_path.suffix.lower() == ".parquet":
        return pd.read_parquet(input_path)
    return pd.read_csv(input_path)


def _save_processed_data(df: pd.DataFrame, output_path: Path) -> None:
    # Ensure the processed-data directory exists before writing the file.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess raw CRSP data and save it under data/processed."
    )
    parser.add_argument(
        "--input",
        help="Path to the raw CRSP file (.csv or .parquet). Defaults to data/raw.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path for the processed output file (.csv or .parquet).",
    )
    args = parser.parse_args()

    input_path = _resolve_input_path(args.input)
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        # Keep relative output paths anchored to the repository root as well.
        output_path = REPO_ROOT / output_path

    raw_df = _load_crsp_data(input_path)
    processed_df = data_preprocessing(raw_df)
    _save_processed_data(processed_df, output_path)

    print(f"Loaded {len(raw_df):,} rows from {input_path}")
    print(f"Saved {len(processed_df):,} processed rows to {output_path}")


if __name__ == "__main__":
    main()
