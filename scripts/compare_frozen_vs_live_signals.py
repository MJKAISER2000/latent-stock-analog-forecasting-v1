import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FROZEN_SIGNALS_PATH = PROJECT_ROOT / "outputs" / "paper_trading" / "paper_portfolio_signals.csv"
FROZEN_RUN_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "paper_trading" / "paper_trading_run_summary.csv"

LIVE_SIGNALS_PATH = PROJECT_ROOT / "outputs" / "paper_trading_live" / "live_portfolio_signals.csv"
LIVE_RUN_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "paper_trading_live" / "live_run_summary.csv"

OUTPUT_COMPARISON_PATH = PROJECT_ROOT / "outputs" / "tables" / "frozen_vs_live_signal_comparison.csv"
OUTPUT_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "frozen_vs_live_signal_comparison_report.txt"


def load_signal_file(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} signal file not found: {path}")

    df = pd.read_csv(path)

    df["signal_date"] = pd.to_datetime(
        df["signal_date"],
        errors="coerce",
    ).dt.normalize()

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["final_weight"] = pd.to_numeric(df["final_weight"], errors="coerce").fillna(0.0)

    if "branches" not in df.columns:
        df["branches"] = ""

    if "best_rank" not in df.columns:
        df["best_rank"] = np.nan

    if "regime_risk_on" not in df.columns:
        df["regime_risk_on"] = np.nan

    if "tech_drawdown" not in df.columns:
        df["tech_drawdown"] = np.nan

    if "model_name" not in df.columns:
        df["model_name"] = label

    return df


def latest_signal(df: pd.DataFrame) -> pd.DataFrame:
    latest_date = df["signal_date"].max()
    latest = df[df["signal_date"] == latest_date].copy()
    latest = latest.sort_values("final_weight", ascending=False).reset_index(drop=True)
    return latest


def load_latest_run_summary(path: Path) -> dict:
    if not path.exists():
        return {}

    df = pd.read_csv(path)

    if len(df) == 0:
        return {}

    if "signal_date" in df.columns:
        df["signal_date"] = pd.to_datetime(
            df["signal_date"],
            errors="coerce",
        ).dt.normalize()

        df = df.sort_values("signal_date")

    return df.tail(1).iloc[0].to_dict()


def compare_latest_signals(
    frozen: pd.DataFrame,
    live: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    frozen_latest = latest_signal(frozen)
    live_latest = latest_signal(live)

    frozen_date = frozen_latest["signal_date"].iloc[0]
    live_date = live_latest["signal_date"].iloc[0]

    frozen_cols = [
        "ticker",
        "final_weight",
        "branches",
        "best_rank",
        "regime_risk_on",
        "tech_drawdown",
    ]

    live_cols = [
        "ticker",
        "final_weight",
        "branches",
        "best_rank",
        "regime_risk_on",
        "tech_drawdown",
    ]

    frozen_small = frozen_latest[[c for c in frozen_cols if c in frozen_latest.columns]].copy()
    live_small = live_latest[[c for c in live_cols if c in live_latest.columns]].copy()

    frozen_small = frozen_small.rename(
        columns={
            "final_weight": "frozen_weight",
            "branches": "frozen_branches",
            "best_rank": "frozen_best_rank",
            "regime_risk_on": "frozen_regime_risk_on",
            "tech_drawdown": "frozen_tech_drawdown",
        }
    )

    live_small = live_small.rename(
        columns={
            "final_weight": "live_weight",
            "branches": "live_branches",
            "best_rank": "live_best_rank",
            "regime_risk_on": "live_regime_risk_on",
            "tech_drawdown": "live_tech_drawdown",
        }
    )

    comparison = frozen_small.merge(
        live_small,
        on="ticker",
        how="outer",
    )

    comparison["frozen_weight"] = comparison["frozen_weight"].fillna(0.0)
    comparison["live_weight"] = comparison["live_weight"].fillna(0.0)

    comparison["in_frozen"] = comparison["frozen_weight"] > 0
    comparison["in_live"] = comparison["live_weight"] > 0
    comparison["in_both"] = comparison["in_frozen"] & comparison["in_live"]

    comparison["weight_diff_live_minus_frozen"] = (
        comparison["live_weight"] - comparison["frozen_weight"]
    )

    comparison["abs_weight_diff"] = comparison["weight_diff_live_minus_frozen"].abs()

    comparison = comparison.sort_values(
        ["in_both", "abs_weight_diff", "live_weight", "frozen_weight"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    frozen_tickers = set(frozen_latest.loc[frozen_latest["ticker"] != "CASH", "ticker"])
    live_tickers = set(live_latest.loc[live_latest["ticker"] != "CASH", "ticker"])

    overlap = sorted(frozen_tickers & live_tickers)
    frozen_only = sorted(frozen_tickers - live_tickers)
    live_only = sorted(live_tickers - frozen_tickers)

    frozen_weight_series = comparison.set_index("ticker")["frozen_weight"]
    live_weight_series = comparison.set_index("ticker")["live_weight"]

    if len(comparison) > 1:
        weight_corr = frozen_weight_series.corr(live_weight_series)
    else:
        weight_corr = np.nan

    frozen_cash_weight = float(
        frozen_latest.loc[frozen_latest["ticker"] == "CASH", "final_weight"].sum()
    )

    live_cash_weight = float(
        live_latest.loc[live_latest["ticker"] == "CASH", "final_weight"].sum()
    )

    frozen_top = (
        frozen_latest[frozen_latest["ticker"] != "CASH"]
        .sort_values("final_weight", ascending=False)
        .head(10)
    )

    live_top = (
        live_latest[live_latest["ticker"] != "CASH"]
        .sort_values("final_weight", ascending=False)
        .head(10)
    )

    stats = {
        "frozen_signal_date": frozen_date,
        "live_signal_date": live_date,
        "frozen_model_name": frozen_latest["model_name"].iloc[0],
        "live_model_name": live_latest["model_name"].iloc[0],
        "frozen_name_count": len(frozen_tickers),
        "live_name_count": len(live_tickers),
        "overlap_count": len(overlap),
        "frozen_only_count": len(frozen_only),
        "live_only_count": len(live_only),
        "overlap_pct_of_frozen": len(overlap) / len(frozen_tickers) if len(frozen_tickers) > 0 else np.nan,
        "overlap_pct_of_live": len(overlap) / len(live_tickers) if len(live_tickers) > 0 else np.nan,
        "weight_correlation": weight_corr,
        "total_abs_weight_diff": float(comparison["abs_weight_diff"].sum()),
        "frozen_cash_weight": frozen_cash_weight,
        "live_cash_weight": live_cash_weight,
        "overlap_tickers": overlap,
        "frozen_only_tickers": frozen_only,
        "live_only_tickers": live_only,
        "frozen_top_tickers": frozen_top["ticker"].tolist(),
        "live_top_tickers": live_top["ticker"].tolist(),
    }

    return comparison, stats


def write_report(
    comparison: pd.DataFrame,
    stats: dict,
    frozen_summary: dict,
    live_summary: dict,
) -> None:
    lines = []
    lines.append("Frozen vs Live Portfolio Signal Comparison")
    lines.append("=========================================")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("Signal dates:")
    lines.append(f"Frozen signal date: {stats['frozen_signal_date']}")
    lines.append(f"Live signal date:   {stats['live_signal_date']}")
    lines.append("")

    lines.append("Model names:")
    lines.append(f"Frozen model: {stats['frozen_model_name']}")
    lines.append(f"Live model:   {stats['live_model_name']}")
    lines.append("")

    lines.append("Portfolio overlap:")
    lines.append(f"Frozen name count: {stats['frozen_name_count']}")
    lines.append(f"Live name count:   {stats['live_name_count']}")
    lines.append(f"Overlap count:     {stats['overlap_count']}")
    lines.append(f"Frozen-only count: {stats['frozen_only_count']}")
    lines.append(f"Live-only count:   {stats['live_only_count']}")
    lines.append(f"Overlap % frozen:  {stats['overlap_pct_of_frozen']:.2%}")
    lines.append(f"Overlap % live:    {stats['overlap_pct_of_live']:.2%}")
    lines.append(f"Weight correlation:{stats['weight_correlation']}")
    lines.append(f"Total abs weight diff: {stats['total_abs_weight_diff']:.4f}")
    lines.append("")

    lines.append("Cash / regime:")
    lines.append(f"Frozen cash weight: {stats['frozen_cash_weight']:.2%}")
    lines.append(f"Live cash weight:   {stats['live_cash_weight']:.2%}")

    if frozen_summary:
        lines.append(f"Frozen run summary top ticker: {frozen_summary.get('top_ticker', 'unknown')}")
        lines.append(f"Frozen regime risk-on: {frozen_summary.get('regime_risk_on', 'unknown')}")
        lines.append(f"Frozen tech drawdown: {frozen_summary.get('tech_drawdown', 'unknown')}")

    if live_summary:
        lines.append(f"Live run summary top ticker: {live_summary.get('top_ticker', 'unknown')}")
        lines.append(f"Live regime risk-on: {live_summary.get('regime_risk_on', 'unknown')}")
        lines.append(f"Live tech drawdown: {live_summary.get('tech_drawdown', 'unknown')}")

    lines.append("")

    lines.append("Overlap tickers:")
    lines.append(", ".join(stats["overlap_tickers"]) if stats["overlap_tickers"] else "None")
    lines.append("")

    lines.append("Frozen-only tickers:")
    lines.append(", ".join(stats["frozen_only_tickers"]) if stats["frozen_only_tickers"] else "None")
    lines.append("")

    lines.append("Live-only tickers:")
    lines.append(", ".join(stats["live_only_tickers"]) if stats["live_only_tickers"] else "None")
    lines.append("")

    lines.append("Frozen top 10:")
    lines.append(", ".join(stats["frozen_top_tickers"]) if stats["frozen_top_tickers"] else "None")
    lines.append("")

    lines.append("Live top 10:")
    lines.append(", ".join(stats["live_top_tickers"]) if stats["live_top_tickers"] else "None")
    lines.append("")

    lines.append("Full comparison table:")
    lines.append(comparison.to_string(index=False))

    os.makedirs(OUTPUT_REPORT_PATH.parent, exist_ok=True)

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(OUTPUT_COMPARISON_PATH.parent, exist_ok=True)
    os.makedirs(OUTPUT_REPORT_PATH.parent, exist_ok=True)

    frozen = load_signal_file(FROZEN_SIGNALS_PATH, label="frozen")
    live = load_signal_file(LIVE_SIGNALS_PATH, label="live")

    frozen_summary = load_latest_run_summary(FROZEN_RUN_SUMMARY_PATH)
    live_summary = load_latest_run_summary(LIVE_RUN_SUMMARY_PATH)

    comparison, stats = compare_latest_signals(
        frozen=frozen,
        live=live,
    )

    comparison.to_csv(OUTPUT_COMPARISON_PATH, index=False)

    write_report(
        comparison=comparison,
        stats=stats,
        frozen_summary=frozen_summary,
        live_summary=live_summary,
    )

    print("")
    print("=" * 100)
    print("FROZEN VS LIVE SIGNAL COMPARISON")
    print("=" * 100)
    print("Frozen signal date:", stats["frozen_signal_date"])
    print("Live signal date:", stats["live_signal_date"])
    print("")
    print("Frozen model:", stats["frozen_model_name"])
    print("Live model:", stats["live_model_name"])
    print("")
    print("Frozen names:", stats["frozen_name_count"])
    print("Live names:", stats["live_name_count"])
    print("Overlap:", stats["overlap_count"])
    print("Frozen-only:", stats["frozen_only_count"])
    print("Live-only:", stats["live_only_count"])
    print("Overlap % frozen:", f"{stats['overlap_pct_of_frozen']:.2%}")
    print("Overlap % live:", f"{stats['overlap_pct_of_live']:.2%}")
    print("Weight correlation:", stats["weight_correlation"])
    print("Total abs weight diff:", stats["total_abs_weight_diff"])
    print("")
    print("Frozen top 10:", stats["frozen_top_tickers"])
    print("Live top 10:", stats["live_top_tickers"])
    print("")
    print("Overlap tickers:", stats["overlap_tickers"])
    print("")
    print("Frozen-only tickers:", stats["frozen_only_tickers"])
    print("")
    print("Live-only tickers:", stats["live_only_tickers"])
    print("")
    print("FULL COMPARISON")
    print(comparison.to_string(index=False))
    print("")
    print("Saved comparison:", OUTPUT_COMPARISON_PATH)
    print("Saved report:", OUTPUT_REPORT_PATH)


if __name__ == "__main__":
    main()