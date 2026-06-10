import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, ensure_output_dirs


CONFIG_PATH = "configs/final_model_config.yaml"

LIVE_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "live_full500_with_stock_latent_neighbors.parquet"

OUTPUT_TABLE_PATH = PROJECT_ROOT / "outputs" / "tables" / "live_vs_research_dataset_comparison.csv"
OUTPUT_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "live_vs_research_dataset_comparison_report.txt"


def load_datasets(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    research_path = PROJECT_ROOT / config["paths"]["neighbor_dataset"]

    if not research_path.exists():
        raise FileNotFoundError(f"Research neighbor dataset not found: {research_path}")

    if not LIVE_DATASET_PATH.exists():
        raise FileNotFoundError(f"Live dataset not found: {LIVE_DATASET_PATH}")

    research = pd.read_parquet(research_path)
    live = pd.read_parquet(LIVE_DATASET_PATH)

    research["date"] = pd.to_datetime(research["date"]).dt.normalize()
    live["date"] = pd.to_datetime(live["date"]).dt.normalize()

    research["ticker"] = research["ticker"].astype(str).str.strip().str.upper()
    live["ticker"] = live["ticker"].astype(str).str.strip().str.upper()

    return research, live


def compare_columns(research: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    all_cols = sorted(set(research.columns) | set(live.columns))

    rows = []

    for col in all_cols:
        in_research = col in research.columns
        in_live = col in live.columns

        research_dtype = str(research[col].dtype) if in_research else None
        live_dtype = str(live[col].dtype) if in_live else None

        research_missing = float(research[col].isna().mean()) if in_research else None
        live_missing = float(live[col].isna().mean()) if in_live else None

        rows.append(
            {
                "column": col,
                "in_research": in_research,
                "in_live": in_live,
                "research_dtype": research_dtype,
                "live_dtype": live_dtype,
                "research_missing_pct": research_missing,
                "live_missing_pct": live_missing,
            }
        )

    return pd.DataFrame(rows)


def compare_core_overlap(research: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    common_dates = sorted(set(research["date"]) & set(live["date"]))
    common_tickers = sorted(set(research["ticker"]) & set(live["ticker"]))

    latest_common_date = max(common_dates)

    research_latest = research[
        (research["date"] == latest_common_date)
        & (research["ticker"].isin(common_tickers))
    ].copy()

    live_latest = live[
        (live["date"] == latest_common_date)
        & (live["ticker"].isin(common_tickers))
    ].copy()

    key_cols = [
        "ticker",
        "ret_1m",
        "ret_3m",
        "ret_6m",
        "ret_12m",
        "vol_12m",
        "stock_drawdown",
        "future_1m_return",
        "neighbor_count",
        "neighbor_avg_future_1m_return",
        "neighbor_avg_future_1m_excess_return",
        "neighbor_outperform_spy_1m_rate",
        "neighbor_positive_1m_return_rate",
    ]

    key_cols = [
        c for c in key_cols
        if c in research_latest.columns and c in live_latest.columns
    ]

    r = research_latest[["ticker"] + key_cols[1:]].copy()
    l = live_latest[["ticker"] + key_cols[1:]].copy()

    merged = r.merge(l, on="ticker", suffixes=("_research", "_live"), how="inner")

    rows = []

    for col in key_cols[1:]:
        r_col = f"{col}_research"
        l_col = f"{col}_live"

        diff = pd.to_numeric(merged[l_col], errors="coerce") - pd.to_numeric(merged[r_col], errors="coerce")

        rows.append(
            {
                "feature": col,
                "comparison_date": latest_common_date,
                "overlap_count": int(diff.notna().sum()),
                "mean_abs_diff": float(diff.abs().mean(skipna=True)),
                "median_abs_diff": float(diff.abs().median(skipna=True)),
                "max_abs_diff": float(diff.abs().max(skipna=True)),
                "research_missing_pct": float(merged[r_col].isna().mean()),
                "live_missing_pct": float(merged[l_col].isna().mean()),
            }
        )

    return pd.DataFrame(rows)


def write_report(
    research: pd.DataFrame,
    live: pd.DataFrame,
    column_compare: pd.DataFrame,
    core_compare: pd.DataFrame,
) -> None:
    research_only_cols = column_compare[
        (column_compare["in_research"] == True)
        & (column_compare["in_live"] == False)
    ]["column"].tolist()

    live_only_cols = column_compare[
        (column_compare["in_research"] == False)
        & (column_compare["in_live"] == True)
    ]["column"].tolist()

    common_cols = column_compare[
        (column_compare["in_research"] == True)
        & (column_compare["in_live"] == True)
    ]["column"].tolist()

    common_dates = sorted(set(research["date"]) & set(live["date"]))
    common_tickers = sorted(set(research["ticker"]) & set(live["ticker"]))

    lines = []
    lines.append("Live vs Research Model Dataset Comparison")
    lines.append("========================================")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Research shape: {research.shape}")
    lines.append(f"Research date range: {research['date'].min()} to {research['date'].max()}")
    lines.append(f"Research ticker count: {research['ticker'].nunique()}")
    lines.append("")
    lines.append(f"Live shape: {live.shape}")
    lines.append(f"Live date range: {live['date'].min()} to {live['date'].max()}")
    lines.append(f"Live ticker count: {live['ticker'].nunique()}")
    lines.append("")
    lines.append(f"Common dates: {len(common_dates)}")
    lines.append(f"Common tickers: {len(common_tickers)}")
    lines.append(f"Common columns: {len(common_cols)}")
    lines.append(f"Research-only columns: {len(research_only_cols)}")
    lines.append(f"Live-only columns: {len(live_only_cols)}")
    lines.append("")
    lines.append("Research-only columns:")
    lines.append(", ".join(research_only_cols) if research_only_cols else "None")
    lines.append("")
    lines.append("Live-only columns:")
    lines.append(", ".join(live_only_cols) if live_only_cols else "None")
    lines.append("")
    lines.append("Core overlap comparison:")
    lines.append(core_compare.to_string(index=False))
    lines.append("")
    lines.append("Column comparison head:")
    lines.append(column_compare.head(100).to_string(index=False))

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    config = load_config(CONFIG_PATH)
    ensure_output_dirs(config)

    os.makedirs(PROJECT_ROOT / "outputs" / "tables", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "outputs" / "reports", exist_ok=True)

    research, live = load_datasets(config)

    column_compare = compare_columns(research, live)
    core_compare = compare_core_overlap(research, live)

    column_compare.to_csv(OUTPUT_TABLE_PATH, index=False)

    write_report(
        research=research,
        live=live,
        column_compare=column_compare,
        core_compare=core_compare,
    )

    research_only = column_compare[
        (column_compare["in_research"] == True)
        & (column_compare["in_live"] == False)
    ]

    live_only = column_compare[
        (column_compare["in_research"] == False)
        & (column_compare["in_live"] == True)
    ]

    common_dates = sorted(set(research["date"]) & set(live["date"]))
    common_tickers = sorted(set(research["ticker"]) & set(live["ticker"]))

    print("")
    print("=" * 100)
    print("LIVE VS RESEARCH MODEL DATASET COMPARISON")
    print("=" * 100)
    print("Research shape:", research.shape)
    print("Research date range:", research["date"].min(), "to", research["date"].max())
    print("Research tickers:", research["ticker"].nunique())
    print("")
    print("Live shape:", live.shape)
    print("Live date range:", live["date"].min(), "to", live["date"].max())
    print("Live tickers:", live["ticker"].nunique())
    print("")
    print("Common dates:", len(common_dates))
    print("Common tickers:", len(common_tickers))
    print("Research-only columns:", len(research_only))
    print("Live-only columns:", len(live_only))
    print("")
    print("CORE FEATURE COMPARISON")
    print(core_compare.to_string(index=False))
    print("")
    print("RESEARCH-ONLY COLUMNS")
    print(research_only["column"].tolist())
    print("")
    print("LIVE-ONLY COLUMNS")
    print(live_only["column"].tolist())
    print("")
    print("Saved table:", OUTPUT_TABLE_PATH)
    print("Saved report:", OUTPUT_REPORT_PATH)


if __name__ == "__main__":
    main()