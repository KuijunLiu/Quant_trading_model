from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# `Path(__file__)` 表示“当前这个 Python 文件的路径”。
# `resolve()` 会把它变成绝对路径。
# `parents[2]` 表示往上退两层目录：
# src/features/features_build.py -> src/features -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
# `Path` 对象可以用 `/` 来拼接路径，读起来比字符串拼接更直观，也能自动处理不同系统的路径分隔符。
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "processed" / "crsp_monthly_processed.csv"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "crsp_monthly_features.csv"
REQUIRED_COLUMNS = {"date", "permno", "ret", "prc", "shrout"}


def _resolve_path(path_str: str | None, default_path: Path) -> Path:
    # 如果用户传了字符串路径，就把它转成 `Path`；否则使用默认路径。
    # `expanduser()` 会把 `~` 展开成用户家目录。
    path = Path(path_str).expanduser() if path_str else default_path
    # 如果是相对路径（例如 `data/processed/a.csv`），就默认按仓库根目录来解释。
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
    # `parent` 是输出文件所在的文件夹；`mkdir(..., exist_ok=True)` 表示“没有就创建，有就跳过”。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


def _group_shift(series: pd.Series, groups: pd.Series, periods: int) -> pd.Series:
    # 按股票（permno）分组后再做 shift。
    # 例如 periods=1 时，每只股票当前行拿到的是“上个月”的值，而不是整张表上一行的值。
    return series.groupby(groups, sort=False).shift(periods)


def _group_rolling(
    series: pd.Series,
    groups: pd.Series,
    shift_periods: int,
    window: int,
    metric: str,
    min_periods: int | None = None,
) -> pd.Series:
    # 先把序列整体滞后，再做 rolling，避免把“当前月”数据泄漏到当前月特征里。
    # 这是时间序列里避免 look-ahead bias（未来函数）的关键。
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

    # 月收益率不能小于 -100%，否则 log1p 无法计算。
    safe_returns = returns.where(returns > -1)
    # 先转成对数收益率再求和，比直接连乘 `(1+r)` 更稳定。
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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    # 复制一份，避免直接改到传进来的原始 DataFrame。
    features = df.copy()

    # 先统一数据类型。CRSP 导出的 CSV 常见字符串、缺失值、异常值混在一起，
    # 所以这里用 `errors="coerce"` 把无法解析的值转成 NaN，后面统一清洗。
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    features["permno"] = pd.to_numeric(features["permno"], errors="coerce")
    features["ret"] = pd.to_numeric(features["ret"], errors="coerce")
    # CRSP 的 `prc` 可能为负，负号通常只是报价约定，不代表真正的负价格。
    features["prc"] = pd.to_numeric(features["prc"], errors="coerce").abs()
    features["shrout"] = pd.to_numeric(features["shrout"], errors="coerce")

    if "mkt_cap" in features.columns:
        features["mkt_cap"] = pd.to_numeric(features["mkt_cap"], errors="coerce")
    else:
        # 如果预处理阶段没有保留市值，这里即时补算。
        features["mkt_cap"] = features["prc"] * features["shrout"]

    features = features.dropna(subset=["date", "permno", "prc", "shrout", "ret"])
    features = features[(features["prc"] > 0) & (features["shrout"] > 0)]
    # 时间序列特征必须先排序，否则 lag/rolling 会错位。
    features = features.sort_values(["permno", "date"]).reset_index(drop=True)

    # 后面很多特征都要“按股票分组”计算，所以把分组键单独拿出来复用。
    groups = features["permno"]

    if "cfacpr" in features.columns:
        cfacpr = pd.to_numeric(features["cfacpr"], errors="coerce").replace(0, np.nan)
        # 用 CRSP 的价格调整因子得到复权价格，减少拆股/送股对价格特征的干扰。
        features["adj_prc"] = features["prc"] / cfacpr
    else:
        features["adj_prc"] = features["prc"]

    if "vol" in features.columns:
        volume = pd.to_numeric(features["vol"], errors="coerce")
        # turnover 近似表示“本月成交股数 / 流通股数”，是一个粗略流动性指标。
        features["turnover"] = volume / features["shrout"]
    else:
        features["turnover"] = np.nan

    # 取对数后，市值分布更平滑，更适合做横截面建模。
    features["log_mkt_cap"] = np.log(features["mkt_cap"].where(features["mkt_cap"] > 0))
    features["ret_lag_1"] = _group_shift(features["ret"], groups, 1)
    features["ret_lag_2"] = _group_shift(features["ret"], groups, 2)
    features["ret_lag_3"] = _group_shift(features["ret"], groups, 3)

    # 跳过最近 1 个月，从更早的窗口计算 momentum，常见于避免短期反转噪声。
    features["mom_2_6"] = _momentum_feature(features["ret"], groups, start_lag=2, end_lag=6)
    features["mom_2_12"] = _momentum_feature(features["ret"], groups, start_lag=2, end_lag=12)

    # 波动率使用过去收益率的滚动标准差。
    features["vol_3m"] = _group_rolling(features["ret"], groups, shift_periods=1, window=3, metric="std")
    features["vol_6m"] = _group_rolling(features["ret"], groups, shift_periods=1, window=6, metric="std")
    features["vol_12m"] = _group_rolling(features["ret"], groups, shift_periods=1, window=12, metric="std")

    ma_6 = _group_rolling(features["adj_prc"], groups, shift_periods=1, window=6, metric="mean")
    ma_12 = _group_rolling(features["adj_prc"], groups, shift_periods=1, window=12, metric="mean")
    lagged_price = _group_shift(features["adj_prc"], groups, 1)
    # 当前可用价格（上月末）相对过去均线的位置，可理解为简单趋势强弱。
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

    # 向前看 1 个月收益率，通常作为监督学习里的预测目标（label）。
    features["ret_fwd_1m"] = _group_shift(features["ret"], groups, -1)

    # 把核心特征列排到前面，便于 notebook 里浏览和后续建模。
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

    # 命令行里不传参数时，直接使用默认的 processed 输入和输出位置。
    input_path = _resolve_path(args.input, DEFAULT_INPUT_PATH)
    output_path = _resolve_path(args.output, DEFAULT_OUTPUT_PATH)

    raw_df = _load_table(input_path)
    feature_df = build_features(raw_df)
    _save_table(feature_df, output_path)

    print(f"Loaded {len(raw_df):,} rows from {input_path}")
    print(f"Saved {len(feature_df):,} rows with features to {output_path}")


if __name__ == "__main__":
    main()
