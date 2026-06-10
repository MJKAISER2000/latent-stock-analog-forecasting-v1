import os
from typing import Any

import numpy as np
import pandas as pd


def get_holdings_path(config: dict[str, Any]) -> str:
    outputs_dir = config["paths"]["outputs_dir"]
    os.makedirs(outputs_dir, exist_ok=True)

    return os.path.join(outputs_dir, "current_paper_holdings.csv")


def initialize_holdings_from_orders(
    orders: pd.DataFrame,
    config: dict[str, Any],
    overwrite: bool = False,
) -> str:
    """
    Initialize current paper holdings from a paper order sheet.

    This is intended for the first paper-trading run, where the portfolio starts from cash.
    Later, rebalance logic should update holdings from fills.
    """

    holdings_path = get_holdings_path(config)

    if os.path.exists(holdings_path) and not overwrite:
        print(f"Holdings file already exists, not overwriting: {holdings_path}")
        return holdings_path

    orders = orders.copy()
    orders["ticker"] = orders["ticker"].astype(str).str.strip().str.upper()
    orders["signal_date"] = pd.to_datetime(orders["signal_date"])
    orders["price_date"] = pd.to_datetime(orders["price_date"])

    stock_orders = orders[
        (orders["ticker"] != "CASH")
        & (orders["order_type"] == "BUY_TO_TARGET")
    ].copy()

    rows = []

    for _, row in stock_orders.iterrows():
        shares = float(row["rounded_shares"])

        if pd.isna(shares) or shares <= 0:
            continue

        latest_price = float(row["latest_price"])
        market_value = shares * latest_price

        rows.append(
            {
                "ticker": row["ticker"],
                "shares": shares,
                "avg_entry_price": latest_price,
                "last_price": latest_price,
                "market_value": market_value,
                "last_signal_date": row["signal_date"],
                "last_price_date": row["price_date"],
                "last_updated": row["run_timestamp"],
                "source": "initial_orders",
            }
        )

    holdings = pd.DataFrame(rows)

    starting_cash = float(config["paper_trading"]["starting_cash"])
    invested = float(holdings["market_value"].sum()) if len(holdings) > 0 else 0.0

    explicit_cash = float(
        orders.loc[orders["ticker"] == "CASH", "target_dollars"].sum()
    )

    leftover_cash = float(
        orders["leftover_cash_from_rounding"].iloc[0]
        if "leftover_cash_from_rounding" in orders.columns
        else 0.0
    )

    cash = starting_cash - invested

    # Use the computed leftover if available and sensible.
    if leftover_cash >= 0:
        cash = explicit_cash + leftover_cash

    cash_row = {
        "ticker": "CASH",
        "shares": cash,
        "avg_entry_price": 1.0,
        "last_price": 1.0,
        "market_value": cash,
        "last_signal_date": orders["signal_date"].iloc[0],
        "last_price_date": orders["price_date"].iloc[0],
        "last_updated": orders["run_timestamp"].iloc[0],
        "source": "initial_cash",
    }

    holdings = pd.concat([holdings, pd.DataFrame([cash_row])], ignore_index=True)

    total_value = float(holdings["market_value"].sum())

    if total_value > 0:
        holdings["current_weight"] = holdings["market_value"] / total_value
    else:
        holdings["current_weight"] = 0.0

    holdings = holdings.sort_values("market_value", ascending=False).reset_index(drop=True)

    holdings.to_csv(holdings_path, index=False)

    return holdings_path


def load_current_holdings(config: dict[str, Any]) -> pd.DataFrame:
    holdings_path = get_holdings_path(config)

    if not os.path.exists(holdings_path):
        raise FileNotFoundError(
            f"Current holdings file does not exist yet: {holdings_path}"
        )

    holdings = pd.read_csv(holdings_path)
    holdings["ticker"] = holdings["ticker"].astype(str).str.strip().str.upper()

    for col in [
        "shares",
        "avg_entry_price",
        "last_price",
        "market_value",
        "current_weight",
    ]:
        if col in holdings.columns:
            holdings[col] = pd.to_numeric(holdings[col], errors="coerce")

    for col in ["last_signal_date", "last_price_date"]:
        if col in holdings.columns:
            holdings[col] = pd.to_datetime(holdings[col], errors="coerce")

    return holdings


def mark_holdings_to_market(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Update holdings using latest available prices at or before as_of_date.
    """

    holdings = holdings.copy()
    holdings["ticker"] = holdings["ticker"].astype(str).str.strip().str.upper()

    as_of_date = pd.Timestamp(as_of_date)

    available_prices = prices[prices.index <= as_of_date].copy()

    if len(available_prices) == 0:
        raise ValueError(f"No price data at or before as_of_date={as_of_date}")

    price_date = pd.Timestamp(available_prices.index.max())
    latest_prices = available_prices.loc[price_date]

    updated_rows = []

    for _, row in holdings.iterrows():
        ticker = row["ticker"]
        shares = float(row["shares"])

        if ticker == "CASH":
            last_price = 1.0
            market_value = shares
        else:
            if ticker not in latest_prices.index:
                last_price = row.get("last_price", np.nan)
            else:
                last_price = float(latest_prices[ticker])

            if pd.isna(last_price):
                market_value = np.nan
            else:
                market_value = shares * last_price

        updated = row.to_dict()
        updated["last_price"] = last_price
        updated["market_value"] = market_value
        updated["last_price_date"] = price_date

        updated_rows.append(updated)

    out = pd.DataFrame(updated_rows)

    total_value = out["market_value"].sum(skipna=True)

    if total_value > 0:
        out["current_weight"] = out["market_value"] / total_value
    else:
        out["current_weight"] = 0.0

    out = out.sort_values("market_value", ascending=False).reset_index(drop=True)

    return out


def save_current_holdings(
    holdings: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    holdings_path = get_holdings_path(config)
    holdings.to_csv(holdings_path, index=False)
    return holdings_path


def print_holdings_summary(holdings: pd.DataFrame) -> None:
    stock_holdings = holdings[holdings["ticker"] != "CASH"].copy()
    cash_weight = float(
        holdings.loc[holdings["ticker"] == "CASH", "current_weight"].sum()
    )

    total_value = float(holdings["market_value"].sum())

    print("")
    print("=" * 80)
    print("CURRENT PAPER HOLDINGS")
    print("=" * 80)
    print("Total portfolio value:", round(total_value, 2))
    print("Stock positions:", len(stock_holdings))
    print("Cash weight:", round(cash_weight, 4))

    print("")
    print(holdings.to_string(index=False))