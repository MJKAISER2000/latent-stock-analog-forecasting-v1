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

LIVE_PRICES_PATH = PROJECT_ROOT / "data" / "processed" / "live_500_monthly_prices.parquet"
OUTPUT_TABLE_PATH = PROJECT_ROOT / "outputs" / "tables" / "live_vs_research_price_comparison.csv"
OUTPUT_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "live_vs_research_price_comparison_report.txt"


def load_prices(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    research_path = PROJECT_ROOT / config["paths"]["monthly_prices"]

    if not research_path.exists():
        raise FileNotFoundError(f"Research price file not found: {research_path}")

    if not LIVE_PRICES_PATH.exists():
        raise FileNotFoundError(f"Live price file not found: {LIVE_PRICES_PATH}")

    research = pd.read_parquet(research_path)
    live = pd.read_parquet(LIVE_PRICES_PATH)

    research.index = pd.to_datetime(research.index)
    live.index = pd.to_datetime(live.index)

    research = research.sort_index()
    live = live.sort_index()

    research.columns = [str(c).strip().upper() for c in research.columns]
    live.columns = [str(c).strip().upper() for c in live.columns]

    return research, live


def month_end_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index).to_period("M").to_timestamp("M")
    out = out.groupby(out.index).last()
    out = out.sort_index()
    return out


def compare_prices(research: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    research_me = month_end_index(research)
    live_me = month_end_index(live)

    common_dates = sorted(set(research_me.index) & set(live_me.index))
    common_tickers = sorted(set(research_me.columns) & set(live_me.columns))

    rows = []

    for ticker in common_tickers:
        r = pd.to_numeric(research_me.loc[common_dates, ticker], errors="coerce")
        l = pd.to_numeric(live_me.loc[common_dates, ticker], errors="coerce")

        valid = r.notna() & l.notna() & (r != 0)

        if valid.sum() == 0:
            rows.append(
                {
                    "ticker": ticker,
                    "overlap_count": 0,
                    "research_first_date": None,
                    "research_last_date": None,
                    "live_first_date": None,
                    "live_last_date": None,
                    "mean_abs_pct_diff": np.nan,
                    "median_abs_pct_diff": np.nan,
                    "max_abs_pct_diff": np.nan,
                    "latest_research_price": np.nan,
                    "latest_live_price": np.nan,
                    "latest_abs_pct_diff": np.nan,
                }
            )
            continue

        pct_diff = (l[valid] - r[valid]) / r[valid]
        abs_pct_diff = pct_diff.abs()

        latest_date = max(pd.Index(common_dates)[valid.values])
        latest_research = float(r.loc[latest_date])
        latest_live = float(l.loc[latest_date])
        latest_abs_pct_diff = abs(latest_live - latest_research) / latest_research

        r_non_missing = r.dropna()
        l_non_missing = l.dropna()

        rows.append(
            {
                "ticker": ticker,
                "overlap_count": int(valid.sum()),
                "research_first_date": r_non_missing.index.min() if len(r_non_missing) else None,
                "research_last_date": r_non_missing.index.max() if len(r_non_missing) else None,
                "live_first_date": l_non_missing.index.min() if len(l_non_missing) else None,
                "live_last_date": l_non_missing.index.max() if len(l_non_missing) else None,
                "mean_abs_pct_diff": float(abs_pct_diff.mean()),
                "median_abs_pct_diff": float(abs_pct_diff.median()),
                "max_abs_pct_diff": float(abs_pct_diff.max()),
                "latest_research_price": latest_research,
                "latest_live_price": latest_live,
                "latest_abs_pct_diff": float(latest_abs_pct_diff),
            }
        )

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values("latest_abs_pct_diff", ascending=False).reset_index(drop=True)

    return comparison


def write_report(
    research: pd.DataFrame,
    live: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    research_me = month_end_index(research)
    live_me = month_end_index(live)

    common_dates = sorted(set(research_me.index) & set(live_me.index))
    common_tickers = sorted(set(research_me.columns) & set(live_me.columns))

    research_only = sorted(set(research_me.columns) - set(live_me.columns))
    live_only = sorted(set(live_me.columns) - set(research_me.columns))

    high_diff = comparison[
        comparison["latest_abs_pct_diff"] > 0.05
    ].copy()

    lines = []
    lines.append("Live vs Research Price Comparison Report")
    lines.append("=======================================")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Research shape: {research.shape}")
    lines.append(f"Research date range: {research.index.min()} to {research.index.max()}")
    lines.append(f"Live shape: {live.shape}")
    lines.append(f"Live date range: {live.index.min()} to {live.index.max()}")
    lines.append("")
    lines.append(f"Common month-end dates: {len(common_dates)}")
    lines.append(f"Common tickers: {len(common_tickers)}")
    lines.append(f"Research-only tickers: {len(research_only)}")
    lines.append(f"Live-only tickers: {len(live_only)}")
    lines.append("")
    lines.append(f"Median latest abs pct diff: {comparison['latest_abs_pct_diff'].median():.6f}")
    lines.append(f"Mean latest abs pct diff: {comparison['latest_abs_pct_diff'].mean():.6f}")
    lines.append(f"Tickers with latest abs pct diff > 5%: {len(high_diff)}")
    lines.append("")
    lines.append("Top latest price differences:")
    lines.append(comparison.head(50).to_string(index=False))
    lines.append("")
    lines.append("Research-only tickers:")
    lines.append(", ".join(research_only) if research_only else "None")
    lines.append("")
    lines.append("Live-only tickers:")
    lines.append(", ".join(live_only) if live_only else "None")

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    config = load_config(CONFIG_PATH)
    ensure_output_dirs(config)

    os.makedirs(PROJECT_ROOT / "outputs" / "tables", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "outputs" / "reports", exist_ok=True)

    research, live = load_prices(config)
    comparison = compare_prices(research, live)

    comparison.to_csv(OUTPUT_TABLE_PATH, index=False)
    write_report(research, live, comparison)

    print("")
    print("=" * 100)
    print("LIVE VS RESEARCH PRICE COMPARISON")
    print("=" * 100)
    print("Research shape:", research.shape)
    print("Research date range:", research.index.min(), "to", research.index.max())
    print("Live shape:", live.shape)
    print("Live date range:", live.index.min(), "to", live.index.max())
    print("")
    print("Comparison shape:", comparison.shape)
    print("Median latest abs pct diff:", comparison["latest_abs_pct_diff"].median())
    print("Mean latest abs pct diff:", comparison["latest_abs_pct_diff"].mean())
    print("Tickers > 5% latest diff:", int((comparison["latest_abs_pct_diff"] > 0.05).sum()))
    print("")
    print("TOP DIFFERENCES")
    print(comparison.head(40).to_string(index=False))
    print("")
    print("Saved table:", OUTPUT_TABLE_PATH)
    print("Saved report:", OUTPUT_REPORT_PATH)


if __name__ == "__main__":
    main()