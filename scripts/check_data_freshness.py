import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, ensure_output_dirs, check_required_files


CONFIG_PATH = "configs/final_model_config.yaml"


def get_paths(config: dict) -> dict:
    outputs_dir = Path(config["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "base_dataset": PROJECT_ROOT / config["paths"]["base_dataset"],
        "neighbor_dataset": PROJECT_ROOT / config["paths"]["neighbor_dataset"],
        "monthly_prices": PROJECT_ROOT / config["paths"]["monthly_prices"],
        "universe": PROJECT_ROOT / config["paths"]["universe"],
        "signal_ledger": PROJECT_ROOT / config["paths"]["outputs_dir"] / "paper_portfolio_signals.csv",
        "run_summary": PROJECT_ROOT / config["paths"]["outputs_dir"] / "paper_trading_run_summary.csv",
        "freshness_report": PROJECT_ROOT / config["paths"]["outputs_dir"] / "data_freshness_report.txt",
        "freshness_table": PROJECT_ROOT / config["paths"]["outputs_dir"] / "data_freshness_table.csv",
    }


def file_status(path: Path) -> dict:
    exists = path.exists()

    if not exists:
        return {
            "path": str(path),
            "exists": False,
            "modified_time": None,
            "size_mb": None,
        }

    stat = path.stat()

    return {
        "path": str(path),
        "exists": True,
        "modified_time": datetime.fromtimestamp(stat.st_mtime),
        "size_mb": stat.st_size / (1024 * 1024),
    }


def get_parquet_date_range(path: Path, date_col: str | None = None) -> dict:
    if not path.exists():
        return {
            "row_count": None,
            "column_count": None,
            "min_date": None,
            "max_date": None,
            "ticker_count": None,
            "error": "file_missing",
        }

    try:
        df = pd.read_parquet(path)

        out = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "min_date": None,
            "max_date": None,
            "ticker_count": None,
            "error": None,
        }

        if date_col is None:
            if "date" in df.columns:
                date_col = "date"
            elif isinstance(df.index, pd.DatetimeIndex):
                date_col = "__index__"

        if date_col == "__index__":
            dates = pd.to_datetime(df.index, errors="coerce").dropna()
        elif date_col in df.columns:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        else:
            dates = pd.Series([], dtype="datetime64[ns]")

        if len(dates) > 0:
            out["min_date"] = pd.Timestamp(dates.min()).normalize()
            out["max_date"] = pd.Timestamp(dates.max()).normalize()

        if "ticker" in df.columns:
            out["ticker_count"] = df["ticker"].astype(str).str.upper().nunique()
        else:
            out["ticker_count"] = None

        return out

    except Exception as exc:
        return {
            "row_count": None,
            "column_count": None,
            "min_date": None,
            "max_date": None,
            "ticker_count": None,
            "error": str(exc),
        }


def get_csv_date_range(path: Path, date_col: str | None = None) -> dict:
    if not path.exists():
        return {
            "row_count": None,
            "column_count": None,
            "min_date": None,
            "max_date": None,
            "ticker_count": None,
            "error": "file_missing",
        }

    try:
        df = pd.read_csv(path)

        out = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "min_date": None,
            "max_date": None,
            "ticker_count": None,
            "error": None,
        }

        if date_col is None:
            for candidate in ["date", "signal_date", "valuation_date", "price_date"]:
                if candidate in df.columns:
                    date_col = candidate
                    break

        if date_col is not None and date_col in df.columns:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()

            if len(dates) > 0:
                out["min_date"] = pd.Timestamp(dates.min()).normalize()
                out["max_date"] = pd.Timestamp(dates.max()).normalize()

        if "ticker" in df.columns:
            out["ticker_count"] = df["ticker"].astype(str).str.upper().nunique()

        return out

    except Exception as exc:
        return {
            "row_count": None,
            "column_count": None,
            "min_date": None,
            "max_date": None,
            "ticker_count": None,
            "error": str(exc),
        }


def summarize_core_files(paths: dict) -> pd.DataFrame:
    specs = [
        {
            "name": "base_dataset",
            "path": paths["base_dataset"],
            "kind": "parquet",
            "date_col": "date",
        },
        {
            "name": "neighbor_dataset",
            "path": paths["neighbor_dataset"],
            "kind": "parquet",
            "date_col": "date",
        },
        {
            "name": "monthly_prices",
            "path": paths["monthly_prices"],
            "kind": "parquet",
            "date_col": "__index__",
        },
        {
            "name": "universe",
            "path": paths["universe"],
            "kind": "csv",
            "date_col": None,
        },
        {
            "name": "signal_ledger",
            "path": paths["signal_ledger"],
            "kind": "csv",
            "date_col": "signal_date",
        },
        {
            "name": "run_summary",
            "path": paths["run_summary"],
            "kind": "csv",
            "date_col": "signal_date",
        },
    ]

    rows = []

    for spec in specs:
        status = file_status(spec["path"])

        if spec["kind"] == "parquet":
            info = get_parquet_date_range(spec["path"], spec["date_col"])
        else:
            info = get_csv_date_range(spec["path"], spec["date_col"])

        row = {
            "name": spec["name"],
            "kind": spec["kind"],
            **status,
            **info,
        }

        rows.append(row)

    return pd.DataFrame(rows)


def assess_freshness(summary: pd.DataFrame) -> dict:
    lookup = summary.set_index("name").to_dict(orient="index")

    monthly_price_date = lookup.get("monthly_prices", {}).get("max_date")
    base_date = lookup.get("base_dataset", {}).get("max_date")
    neighbor_date = lookup.get("neighbor_dataset", {}).get("max_date")
    signal_date = lookup.get("signal_ledger", {}).get("max_date")

    issues = []
    warnings = []

    if monthly_price_date is None:
        issues.append("Monthly price max date could not be read.")

    if neighbor_date is None:
        issues.append("Neighbor dataset max date could not be read.")

    if base_date is None:
        issues.append("Base dataset max date could not be read.")

    if monthly_price_date is not None and neighbor_date is not None:
        if pd.Timestamp(neighbor_date) < pd.Timestamp(monthly_price_date):
            warnings.append(
                f"Neighbor dataset is behind monthly prices: neighbor={neighbor_date}, prices={monthly_price_date}."
            )

    if monthly_price_date is not None and base_date is not None:
        if pd.Timestamp(base_date) < pd.Timestamp(monthly_price_date):
            warnings.append(
                f"Base dataset is behind monthly prices: base={base_date}, prices={monthly_price_date}."
            )

    if signal_date is not None and neighbor_date is not None:
        if pd.Timestamp(signal_date) < pd.Timestamp(neighbor_date):
            warnings.append(
                f"Latest signal is behind neighbor dataset: signal={signal_date}, neighbor={neighbor_date}."
            )

    if signal_date is not None and monthly_price_date is not None:
        if pd.Timestamp(signal_date) < pd.Timestamp(monthly_price_date):
            warnings.append(
                f"Latest signal is behind monthly prices: signal={signal_date}, prices={monthly_price_date}."
            )

    stale_days = None

    if monthly_price_date is not None:
        today = pd.Timestamp.today().normalize()
        stale_days = int((today - pd.Timestamp(monthly_price_date)).days)

        if stale_days > 45:
            warnings.append(
                f"Monthly price data may be stale: latest price date is {monthly_price_date}, {stale_days} days behind today."
            )

    ready_for_current_pipeline = len(issues) == 0

    return {
        "monthly_price_date": monthly_price_date,
        "base_dataset_date": base_date,
        "neighbor_dataset_date": neighbor_date,
        "latest_signal_date": signal_date,
        "stale_days_vs_today": stale_days,
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "ready_for_current_pipeline": ready_for_current_pipeline,
        "issues": issues,
        "warnings": warnings,
    }


def write_report(summary: pd.DataFrame, assessment: dict, report_path: Path) -> None:
    lines = []
    lines.append("Latent Market Twin Data Freshness Report")
    lines.append("=======================================")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Core date status:")
    lines.append(f"Monthly price date:   {assessment['monthly_price_date']}")
    lines.append(f"Base dataset date:    {assessment['base_dataset_date']}")
    lines.append(f"Neighbor dataset date:{assessment['neighbor_dataset_date']}")
    lines.append(f"Latest signal date:   {assessment['latest_signal_date']}")
    lines.append(f"Stale days vs today:  {assessment['stale_days_vs_today']}")
    lines.append("")
    lines.append(f"Ready for current pipeline: {assessment['ready_for_current_pipeline']}")
    lines.append(f"Issue count: {assessment['issue_count']}")
    lines.append(f"Warning count: {assessment['warning_count']}")
    lines.append("")

    lines.append("Issues:")
    if assessment["issues"]:
        for issue in assessment["issues"]:
            lines.append(f"- {issue}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("Warnings:")
    if assessment["warnings"]:
        for warning in assessment["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("Full file summary:")
    lines.append(summary.to_string(index=False))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    config = load_config(CONFIG_PATH)
    ensure_output_dirs(config)

    # Do not call check_required_files here because this script is partly meant
    # to diagnose missing files.
    paths = get_paths(config)

    summary = summarize_core_files(paths)
    assessment = assess_freshness(summary)

    summary.to_csv(paths["freshness_table"], index=False)
    write_report(summary, assessment, paths["freshness_report"])

    print("")
    print("=" * 100)
    print("DATA FRESHNESS CHECK")
    print("=" * 100)
    print(f"Monthly price date:    {assessment['monthly_price_date']}")
    print(f"Base dataset date:     {assessment['base_dataset_date']}")
    print(f"Neighbor dataset date: {assessment['neighbor_dataset_date']}")
    print(f"Latest signal date:    {assessment['latest_signal_date']}")
    print(f"Stale days vs today:   {assessment['stale_days_vs_today']}")
    print("")
    print(f"Ready for current pipeline: {assessment['ready_for_current_pipeline']}")
    print(f"Issues: {assessment['issue_count']}")
    print(f"Warnings: {assessment['warning_count']}")

    if assessment["issues"]:
        print("")
        print("ISSUES")
        for issue in assessment["issues"]:
            print("-", issue)

    if assessment["warnings"]:
        print("")
        print("WARNINGS")
        for warning in assessment["warnings"]:
            print("-", warning)

    print("")
    print("FULL FILE SUMMARY")
    print(summary.to_string(index=False))
    print("")
    print("Saved freshness table:", paths["freshness_table"])
    print("Saved freshness report:", paths["freshness_report"])


if __name__ == "__main__":
    main()