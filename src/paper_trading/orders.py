import os
from typing import Any

import numpy as np
import pandas as pd


def build_order_sheet(
    final_portfolio: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict[str, Any],
    run_timestamp: str,
) -> pd.DataFrame:
    """
    Convert final portfolio weights into target dollar allocations and whole-share paper orders.
    """

    starting_cash = float(config["paper_trading"]["starting_cash"])

    portfolio = final_portfolio.copy()
    portfolio["ticker"] = portfolio["ticker"].astype(str).str.strip().str.upper()
    portfolio["date"] = pd.to_datetime(portfolio["date"])

    signal_date = pd.Timestamp(portfolio["date"].iloc[0])

    available_prices = prices[prices.index <= signal_date].copy()

    if len(available_prices) == 0:
        raise ValueError(f"No prices available at or before signal_date={signal_date}")

    price_date = pd.Timestamp(available_prices.index.max())
    latest_prices = available_prices.loc[price_date]

    rows = []

    for _, row in portfolio.iterrows():
        ticker = row["ticker"]
        weight = float(row["final_weight"])
        target_dollars = starting_cash * weight

        if ticker == "CASH":
            latest_price = np.nan
            target_shares = np.nan
            rounded_shares = np.nan
            rounded_dollars = target_dollars
            order_type = "HOLD_CASH"
        else:
            if ticker not in latest_prices.index:
                latest_price = np.nan
                target_shares = np.nan
                rounded_shares = np.nan
                rounded_dollars = np.nan
                order_type = "MISSING_PRICE"
            else:
                latest_price = float(latest_prices[ticker])

                if latest_price <= 0 or pd.isna(latest_price):
                    target_shares = np.nan
                    rounded_shares = np.nan
                    rounded_dollars = np.nan
                    order_type = "BAD_PRICE"
                else:
                    target_shares = target_dollars / latest_price
                    rounded_shares = float(np.floor(target_shares))
                    rounded_dollars = rounded_shares * latest_price
                    order_type = "BUY_TO_TARGET"

        rows.append(
            {
                "run_timestamp": run_timestamp,
                "signal_date": signal_date,
                "price_date": price_date,
                "ticker": ticker,
                "order_type": order_type,
                "final_weight": weight,
                "target_dollars": target_dollars,
                "latest_price": latest_price,
                "target_shares": target_shares,
                "rounded_shares": rounded_shares,
                "rounded_dollars": rounded_dollars,
                "branches": row.get("branches", ""),
                "best_rank": row.get("best_rank", np.nan),
                "avg_ranker_score": row.get("avg_ranker_score", np.nan),
            }
        )

    orders = pd.DataFrame(rows)

    invested_rounded = orders.loc[
        orders["ticker"] != "CASH",
        "rounded_dollars",
    ].sum(skipna=True)

    explicit_cash = orders.loc[
        orders["ticker"] == "CASH",
        "target_dollars",
    ].sum(skipna=True)

    leftover_cash_from_rounding = starting_cash - invested_rounded - explicit_cash

    if leftover_cash_from_rounding < 0:
        leftover_cash_from_rounding = 0.0

    orders["starting_cash"] = starting_cash
    orders["invested_rounded_total"] = invested_rounded
    orders["explicit_cash_target"] = explicit_cash
    orders["leftover_cash_from_rounding"] = leftover_cash_from_rounding

    orders = orders.sort_values("final_weight", ascending=False).reset_index(drop=True)

    return orders


def save_order_sheet(
    orders: pd.DataFrame,
    config: dict[str, Any],
    run_timestamp: str,
) -> str:
    outputs_dir = config["paths"]["outputs_dir"]
    os.makedirs(outputs_dir, exist_ok=True)

    path = os.path.join(
        outputs_dir,
        f"paper_trade_orders_{run_timestamp}.csv",
    )

    orders.to_csv(path, index=False)

    return path


def append_order_ledger(
    orders: pd.DataFrame,
    config: dict[str, Any],
    overwrite_same_signal_date: bool = True,
) -> str:
    outputs_dir = config["paths"]["outputs_dir"]
    os.makedirs(outputs_dir, exist_ok=True)

    ledger_path = os.path.join(outputs_dir, "paper_trade_orders_ledger.csv")

    rows = orders.copy()
    model_name = config["final_model_name"]
    rows["model_name"] = model_name

    signal_date = pd.Timestamp(rows["signal_date"].iloc[0])

    if os.path.exists(ledger_path):
        existing = pd.read_csv(ledger_path)
        existing["signal_date"] = pd.to_datetime(existing["signal_date"])

        if overwrite_same_signal_date:
            keep_mask = ~(
                (existing["model_name"] == model_name)
                & (existing["signal_date"] == signal_date)
            )
            existing = existing[keep_mask].copy()

        combined = pd.concat([existing, rows], ignore_index=True)
    else:
        combined = rows.copy()

    combined["signal_date"] = pd.to_datetime(combined["signal_date"])
    combined = combined.sort_values(
        ["signal_date", "model_name", "final_weight"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    combined.to_csv(ledger_path, index=False)

    return ledger_path