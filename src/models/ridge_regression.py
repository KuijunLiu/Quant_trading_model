"""
ridge_regression.py
===================
Trains a Ridge Regression model to predict one-month-ahead stock returns
(ret_fwd_1m) using the cross-sectionally standardized features from
crsp_monthly_features.csv.

Pipeline
--------
1. Load train / validation / test splits from data/processed/.
2. Tune the Ridge alpha hyperparameter via walk-forward cross-validation
   on the training set.
3. Re-train the final model on the full training set with the best alpha.
4. Evaluate on the held-out test set.
5. Save predictions, rankings, and a summary to outputs/model_outputs/ridge/.

Usage
-----
  python -m src.models.ridge_regression
  python -m src.models.ridge_regression --alpha-grid 0.01 0.1 1 10 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAIN_PATH = REPO_ROOT / "data" / "processed" / "train.csv"
DEFAULT_VAL_PATH   = REPO_ROOT / "data" / "processed" / "validation.csv"
DEFAULT_TEST_PATH  = REPO_ROOT / "data" / "processed" / "test.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "model_outputs" / "ridge"

LABEL_COL = "ret_fwd_1m"
# All cross-sectionally standardized feature columns.
FEATURE_COLS = [
    "ret_lag_1_cs",
    "ret_lag_2_cs",
    "ret_lag_3_cs",
    "mom_2_6_cs",
    "mom_2_12_cs",
    "vol_3m_cs",
    "vol_6m_cs",
    "vol_12m_cs",
    "price_ma_6_ratio_cs",
    "price_ma_12_ratio_cs",
    "log_mkt_cap_cs",
    "turnover_cs",
    "turnover_3m_cs",
    "turnover_12m_cs",
]

DEFAULT_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


# ------------------------------------------------------------------ #
# I/O helpers                                                          #
# ------------------------------------------------------------------ #

def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved {len(df):,} rows  →  {path}")


def _resolve_path(path: Path, default_path: Path) -> Path:
    resolved = path.expanduser() if str(path) else default_path
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


# ------------------------------------------------------------------ #
# Feature / label extraction                                           #
# ------------------------------------------------------------------ #

def _get_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Drop rows where any feature or the label is NaN, then return X, y, and the clean df."""
    missing_cols = [col for col in feature_cols + [LABEL_COL] if col not in df.columns]
    if missing_cols:
        missing = ", ".join(missing_cols)
        raise KeyError(f"Missing required modeling columns: {missing}")

    cols = feature_cols + [LABEL_COL]
    clean = df.dropna(subset=cols).reset_index(drop=True)
    if clean.empty:
        raise ValueError("No rows remain after dropping NaNs from features/label.")
    X = clean[feature_cols].to_numpy(dtype=np.float64)
    y = clean[LABEL_COL].to_numpy(dtype=np.float64)
    return X, y, clean


# ------------------------------------------------------------------ #
# Walk-forward CV for alpha tuning                                     #
# ------------------------------------------------------------------ #

def _walk_forward_cv_alpha(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    alpha_grid: list[float],
    initial_train_years: int = 5,
    val_years: int = 1,
    step_years: int = 1,
    gap_months: int = 1,
) -> float:
    """Select the best Ridge alpha via walk-forward CV on the training set.

    Returns the alpha with the lowest mean RMSE across all folds.
    """
    # Import here to avoid a circular dependency if split_data is in the same package.
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from split_data.split_data import walk_forward_cv

    alpha_rmse: dict[float, list[float]] = {a: [] for a in alpha_grid}

    print("\n--- Walk-forward CV for alpha tuning ---")
    for fold, fold_train, fold_val in walk_forward_cv(
        train_df,
        initial_train_years=initial_train_years,
        val_years=val_years,
        step_years=step_years,
        gap_months=gap_months,
    ):
        X_tr, y_tr, _ = _get_xy(fold_train, feature_cols)
        X_val, y_val, _ = _get_xy(fold_val, feature_cols)

        if len(X_tr) == 0 or len(X_val) == 0:
            continue

        for alpha in alpha_grid:
            model = Ridge(alpha=alpha)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            alpha_rmse[alpha].append(rmse)

    mean_rmse = {a: np.mean(v) for a, v in alpha_rmse.items() if v}
    if not mean_rmse:
        raise ValueError(
            "Walk-forward CV did not produce any valid folds. "
            "Please check the train split date range and feature availability."
        )
    best_alpha = min(mean_rmse, key=mean_rmse.__getitem__)

    print("\nCV results (mean RMSE per alpha):")
    for a, rmse in sorted(mean_rmse.items()):
        marker = "  ← best" if a == best_alpha else ""
        print(f"  alpha={a:>8.4f}  RMSE={rmse:.6f}{marker}")

    return best_alpha


# ------------------------------------------------------------------ #
# Evaluation helper                                                    #
# ------------------------------------------------------------------ #

def _evaluate(
    model: Ridge,
    df: pd.DataFrame,
    feature_cols: list[str],
    split_name: str,
) -> tuple[pd.DataFrame, dict]:
    """Run predictions on a split, print metrics, and return (predictions_df, metrics_dict).

    Metrics include:
      - RMSE
      - R²
      - Pearson IC  (linear correlation)
      - Rank   IC   (Spearman / rank correlation)
    """
    X, y, clean_df = _get_xy(df, feature_cols)
    preds = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, preds))
    r2   = r2_score(y, preds)
    ic_pearson = pd.Series(preds).corr(pd.Series(y))                     # linear IC
    ic_rank    = pd.Series(preds).corr(pd.Series(y), method="spearman")  # rank  IC

    print(f"\n{split_name} metrics:")
    print(f"  RMSE : {rmse:.6f}")
    print(f"  R²   : {r2:.6f}")
    print(f"  IC (Pearson) : {ic_pearson:.6f}")
    print(f"  IC (Rank)    : {ic_rank:.6f}")

    out_df = clean_df[["date", "permno"] + (["comnam"] if "comnam" in clean_df.columns else [])].copy()
    out_df["y_true"] = y
    out_df["y_pred"] = preds

    metrics = {
        "split": split_name,
        "rmse": rmse,
        "r2": r2,
        "ic_pearson": ic_pearson,
        "ic_rank": ic_rank,
        "n_rows": len(y),
    }
    return out_df, metrics


# ------------------------------------------------------------------ #
# Main training routine                                                #
# ------------------------------------------------------------------ #

def train(
    train_path: Path = DEFAULT_TRAIN_PATH,
    val_path: Path   = DEFAULT_VAL_PATH,
    test_path: Path  = DEFAULT_TEST_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    alpha_grid: list[float] = DEFAULT_ALPHA_GRID,
    feature_cols: list[str] = FEATURE_COLS,
) -> Ridge:
    print(f"Loading train  : {train_path}")
    train_df = _load(train_path)
    print(f"Loading val    : {val_path}")
    val_df   = _load(val_path)
    print(f"Loading test   : {test_path}")
    test_df  = _load(test_path)

    print(f"\nTrain rows : {len(train_df):,}")
    print(f"Val rows   : {len(val_df):,}")
    print(f"Test rows  : {len(test_df):,}")

    # ── Step 1: tune alpha via walk-forward CV on the training set ── #
    best_alpha = _walk_forward_cv_alpha(train_df, feature_cols, alpha_grid)
    print(f"\nBest alpha selected: {best_alpha}")

    # ── Step 2: train final model on the full train+val set ────────── #
    train_val_df = (
        pd.concat([train_df, val_df], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    X_train, y_train, _ = _get_xy(train_val_df, feature_cols)
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_train, y_train)
    print(f"\nFinal model trained on {len(X_train):,} samples  (alpha={best_alpha}) using train+val")

    # ── Step 3: evaluate only on the held-out test set ─────────────── #
    test_preds, test_metrics = _evaluate(final_model, test_df, feature_cols, "Test")

    # ── Step 4: save outputs ───────────────────────────────────────── #
    output_dir.mkdir(parents=True, exist_ok=True)

    _save(test_preds, output_dir / "test_predictions.csv")

    summary = pd.DataFrame([test_metrics])
    _save(summary, output_dir / "metrics_summary.csv")

    # Save model coefficients for interpretability.
    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": final_model.coef_})
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")
    _save(coef_df, output_dir / "coefficients.csv")

    print(f"\nAll outputs saved to: {output_dir}")
    return final_model


# ------------------------------------------------------------------ #
# CLI entry point                                                      #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Ridge Regression on CRSP features.")
    parser.add_argument("--train",      default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--val",        default=str(DEFAULT_VAL_PATH))
    parser.add_argument("--test",       default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--alpha-grid",
        nargs="+",
        type=float,
        default=DEFAULT_ALPHA_GRID,
        help="Space-separated list of alpha values to search (default: 0.01 0.1 1 10 100)",
    )
    args = parser.parse_args()

    train(
        train_path=Path(args.train),
        val_path=Path(args.val),
        test_path=Path(args.test),
        output_dir=Path(args.output_dir),
        alpha_grid=args.alpha_grid,
    )


if __name__ == "__main__":
    main()
