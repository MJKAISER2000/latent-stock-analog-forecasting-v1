import os
from typing import Any

import pandas as pd


def parse_dates_safe(series: pd.Series) -> pd.Series:
    """
    Robust date parsing for ledgers that may contain mixed formats like:
    2026-06-30
    2026-06-30 00:00:00
    """

    try:
        return pd.to_datetime(series, format="mixed", errors="coerce")
    except TypeError:
        return pd.to_datetime(series.astype(str), errors="coerce")


def normalize_signal_date(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def append_signal_ledger(
    final_portfolio: pd.DataFrame,
    regime_status: dict[str, Any],
    config: dict[str, Any],
    run_timestamp: str,
    overwrite_same_signal_date: bool = True,
) -> str:
    """
    Append latest final portfolio weights to a persistent paper-trading signal ledger.

    If overwrite_same_signal_date=True, remove previous rows for the same
    model_name + signal_date before appending the new signal.
    """

    outputs_dir = config["paths"]["outputs_dir"]
    os.makedirs(outputs_dir, exist_ok=True)

    ledger_path = os.path.join(outputs_dir, "paper_portfolio_signals.csv")

    rows = final_portfolio.copy()

    model_name = config["final_model_name"]
    signal_date = normalize_signal_date(regime_status["signal_date"])
    regime_price_date = normalize_signal_date(regime_status["regime_price_date"])

    rows["run_timestamp"] = run_timestamp
    rows["model_name"] = model_name
    rows["signal_date"] = signal_date
    rows["regime_price_date"] = regime_price_date
    rows["regime_rule"] = regime_status["rule"]
    rows["regime_risk_on"] = regime_status["risk_on"]
    rows["tech_drawdown"] = regime_status["tech_drawdown"]
    rows["cash_weight_when_off"] = regime_status["cash_weight_when_off"]

    preferred_cols = [
        "run_timestamp",
        "model_name",
        "signal_date",
        "regime_price_date",
        "regime_rule",
        "regime_risk_on",
        "tech_drawdown",
        "cash_weight_when_off",
        "ticker",
        "final_weight",
        "final_weight_before_regime",
        "branches",
        "avg_ranker_score",
        "best_rank",
    ]

    remaining_cols = [c for c in rows.columns if c not in preferred_cols]
    rows = rows[[c for c in preferred_cols if c in rows.columns] + remaining_cols]

    if os.path.exists(ledger_path):
        existing = pd.read_csv(ledger_path)

        if "signal_date" in existing.columns:
            existing["signal_date"] = parse_dates_safe(existing["signal_date"]).dt.normalize()

        if "regime_price_date" in existing.columns:
            existing["regime_price_date"] = parse_dates_safe(existing["regime_price_date"]).dt.normalize()

        if overwrite_same_signal_date:
            keep_mask = ~(
                (existing["model_name"] == model_name)
                & (existing["signal_date"] == signal_date)
            )
            existing = existing[keep_mask].copy()

        combined = pd.concat([existing, rows], ignore_index=True)
    else:
        combined = rows.copy()

    combined["signal_date"] = parse_dates_safe(combined["signal_date"]).dt.normalize()

    if "regime_price_date" in combined.columns:
        combined["regime_price_date"] = parse_dates_safe(
            combined["regime_price_date"]
        ).dt.normalize()

    combined = combined.sort_values(
        ["signal_date", "model_name", "final_weight"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    combined.to_csv(ledger_path, index=False)

    return ledger_path


def append_run_summary(
    config: dict[str, Any],
    run_timestamp: str,
    signal_date: pd.Timestamp,
    regime_status: dict[str, Any],
    final_portfolio: pd.DataFrame,
    overwrite_same_signal_date: bool = True,
) -> str:
    """
    Append one row per run to a persistent model run summary ledger.

    If overwrite_same_signal_date=True, remove previous summary rows for the same
    model_name + signal_date before appending.
    """

    outputs_dir = config["paths"]["outputs_dir"]
    os.makedirs(outputs_dir, exist_ok=True)

    summary_path = os.path.join(outputs_dir, "paper_trading_run_summary.csv")

    model_name = config["final_model_name"]
    signal_date = normalize_signal_date(signal_date)
    regime_price_date = normalize_signal_date(regime_status["regime_price_date"])

    stock_portfolio = final_portfolio[final_portfolio["ticker"] != "CASH"].copy()

    cash_weight = float(
        final_portfolio.loc[
            final_portfolio["ticker"] == "CASH",
            "final_weight",
        ].sum()
    )

    row = {
        "run_timestamp": run_timestamp,
        "model_name": model_name,
        "signal_date": signal_date,
        "regime_price_date": regime_price_date,
        "regime_rule": regime_status["rule"],
        "regime_risk_on": regime_status["risk_on"],
        "tech_drawdown": regime_status["tech_drawdown"],
        "cash_weight_when_off": regime_status["cash_weight_when_off"],
        "portfolio_name_count": len(stock_portfolio),
        "cash_weight": cash_weight,
        "largest_weight": float(final_portfolio["final_weight"].max()),
        "top_ticker": str(final_portfolio.iloc[0]["ticker"]),
    }

    new_row = pd.DataFrame([row])

    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)

        if "signal_date" in existing.columns:
            existing["signal_date"] = parse_dates_safe(existing["signal_date"]).dt.normalize()

        if "regime_price_date" in existing.columns:
            existing["regime_price_date"] = parse_dates_safe(
                existing["regime_price_date"]
            ).dt.normalize()

        if overwrite_same_signal_date:
            keep_mask = ~(
                (existing["model_name"] == model_name)
                & (existing["signal_date"] == signal_date)
            )
            existing = existing[keep_mask].copy()

        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row.copy()

    combined["signal_date"] = parse_dates_safe(combined["signal_date"]).dt.normalize()

    if "regime_price_date" in combined.columns:
        combined["regime_price_date"] = parse_dates_safe(
            combined["regime_price_date"]
        ).dt.normalize()

    combined = combined.sort_values(["signal_date", "model_name"]).reset_index(drop=True)

    combined.to_csv(summary_path, index=False)

    return summary_path