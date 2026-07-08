"""Apply the latest live rebalance orders to the live holdings file.

The monthly rebuild pipeline generates a new target signal and rebalance orders
(current holdings vs. target), but it does not execute them. This script "executes"
the latest rebalance for the paper portfolio.

Execution is done at *live* prices as of the moment this runs, not the signal-month
price. That means:
  - the value redeployed is the current portfolio valued at today's prices, and
  - each new position's entry price is the price it was actually bought at today,
so the since-entry P&L starts from zero at purchase (not from a month-old price).

Run it after run_live_rebuild_pipeline.py when you want the portfolio to move into
the new picks.
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

for extra in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from src.utils.config import load_config, ensure_output_dirs
from check_ltsaf_live_value import get_quotes


CONFIG_PATH = "configs/live_model_config.yaml"


def get_paths(config: dict) -> dict:
    outputs_dir = PROJECT_ROOT / config["paths"]["outputs_dir"]
    outputs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "outputs_dir": outputs_dir,
        "latest_rebalance": outputs_dir / "latest_live_rebalance_orders.csv",
        "rebalance_ledger": outputs_dir / "live_rebalance_orders_ledger.csv",
        "current_live_holdings": outputs_dir / "current_live_holdings.csv",
        "live_holdings_summary": outputs_dir / "latest_live_holdings_summary.txt",
    }


def load_latest_rebalance(paths: dict) -> pd.DataFrame:
    path = paths["latest_rebalance"]
    if not path.exists():
        path = paths["rebalance_ledger"]
    if not path.exists():
        raise FileNotFoundError(
            "No rebalance orders found. Run scripts/run_live_rebuild_pipeline.py first."
        )

    orders = pd.read_csv(path)
    orders["ticker"] = orders["ticker"].astype(str).str.strip().str.upper()
    orders["signal_date"] = pd.to_datetime(orders["signal_date"], errors="coerce").dt.normalize()

    latest_signal_date = orders["signal_date"].max()
    return orders[orders["signal_date"] == latest_signal_date].copy()


def load_current_holdings(paths: dict) -> pd.DataFrame:
    path = paths["current_live_holdings"]
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "shares", "last_price", "market_value"])

    h = pd.read_csv(path)
    h["ticker"] = h["ticker"].astype(str).str.strip().str.upper()
    h["shares"] = pd.to_numeric(h["shares"], errors="coerce").fillna(0.0)
    h["last_price"] = pd.to_numeric(h.get("last_price"), errors="coerce")
    return h


def fetch_live_prices(tickers: list[str]) -> dict:
    tickers = sorted({t for t in tickers if t and t != "CASH"})
    if not tickers:
        return {}
    quotes = get_quotes(tickers)
    quotes["ticker"] = quotes["ticker"].astype(str).str.strip().str.upper()
    quotes["current_price"] = pd.to_numeric(quotes["current_price"], errors="coerce")
    return dict(zip(quotes["ticker"], quotes["current_price"]))


def current_portfolio_value(holdings: pd.DataFrame, live_prices: dict) -> float:
    """Value the current holdings at live prices (fallback to last_price if a quote is missing)."""
    total = 0.0
    for _, row in holdings.iterrows():
        ticker = row["ticker"]
        shares = float(row["shares"])
        if ticker == "CASH":
            total += shares  # cash is valued 1:1
            continue
        price = live_prices.get(ticker)
        if price is None or pd.isna(price) or price <= 0:
            price = row["last_price"]  # stale fallback
        if pd.notna(price) and price > 0:
            total += shares * float(price)
    return float(total)


def build_holdings(orders: pd.DataFrame, deploy_value: float, live_prices: dict) -> pd.DataFrame:
    signal_date = orders["signal_date"].iloc[0]
    today = datetime.now().strftime("%Y-%m-%d")

    targets = orders[orders["ticker"] != "CASH"].copy()
    targets["target_weight"] = pd.to_numeric(targets["target_weight"], errors="coerce").fillna(0.0)

    rows = []
    for _, row in targets.iterrows():
        ticker = row["ticker"]
        weight = float(row["target_weight"])
        price = live_prices.get(ticker)

        if weight <= 0 or price is None or pd.isna(price) or price <= 0:
            continue

        target_dollars = weight * deploy_value
        shares = float(np.floor(target_dollars / price))  # whole shares only
        if shares <= 0:
            continue

        rows.append(
            {
                "ticker": ticker,
                "shares": shares,
                "avg_entry_price": price,   # entry = price actually bought at, today
                "last_price": price,
                "market_value": shares * price,
                "last_signal_date": signal_date,
                "last_price_date": today,
                "last_updated": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "source": "live_rebalance_apply",
            }
        )

    holdings = pd.DataFrame(rows)
    invested = float(holdings["market_value"].sum()) if len(holdings) > 0 else 0.0
    cash = max(0.0, deploy_value - invested)

    cash_row = {
        "ticker": "CASH",
        "shares": cash,
        "avg_entry_price": 1.0,
        "last_price": 1.0,
        "market_value": cash,
        "last_signal_date": signal_date,
        "last_price_date": today,
        "last_updated": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "source": "live_rebalance_cash",
    }

    holdings = pd.concat([holdings, pd.DataFrame([cash_row])], ignore_index=True)
    total_value = float(holdings["market_value"].sum())
    holdings["current_weight"] = holdings["market_value"] / total_value if total_value > 0 else 0.0
    return holdings.sort_values("market_value", ascending=False).reset_index(drop=True)


def write_summary(holdings: pd.DataFrame, path: Path) -> None:
    stock = holdings[holdings["ticker"] != "CASH"]
    total_value = float(holdings["market_value"].sum())
    cash_value = float(holdings.loc[holdings["ticker"] == "CASH", "market_value"].sum())
    cash_weight = cash_value / total_value if total_value > 0 else 0.0

    lines = [
        "LTSAF Live Holdings Summary (after rebalance)",
        "============================================",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total value: ${total_value:,.2f}",
        f"Cash value: ${cash_value:,.2f}",
        f"Cash weight: {cash_weight:.2%}",
        f"Stock positions: {len(stock)}",
        "",
        "Full holdings:",
        holdings.to_string(index=False),
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    config = load_config(CONFIG_PATH)
    ensure_output_dirs(config)
    paths = get_paths(config)

    orders = load_latest_rebalance(paths)
    current = load_current_holdings(paths)

    signal_date = pd.Timestamp(orders["signal_date"].iloc[0]).date()

    all_tickers = list(current["ticker"]) + list(orders["ticker"])
    print("Fetching live prices to execute the rebalance at today's prices...")
    live_prices = fetch_live_prices(all_tickers)

    deploy_value = current_portfolio_value(current, live_prices)
    if deploy_value <= 0:
        deploy_value = float(pd.to_numeric(orders["portfolio_value"], errors="coerce").dropna().iloc[0])

    holdings = build_holdings(orders, deploy_value, live_prices)

    target = paths["current_live_holdings"]
    if target.exists():
        backup = target.with_suffix(f".pre_rebalance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        backup.write_bytes(target.read_bytes())
        print("Backed up previous holdings to:", backup)

    holdings.to_csv(target, index=False)
    write_summary(holdings, paths["live_holdings_summary"])

    cash_value = float(holdings.loc[holdings["ticker"] == "CASH", "market_value"].sum())
    total_value = float(holdings["market_value"].sum())

    print("")
    print("=" * 100)
    print("LIVE PORTFOLIO REBALANCED (executed at today's prices)")
    print("=" * 100)
    print(f"Applied signal date: {signal_date}")
    print(f"Redeployed value: ${deploy_value:,.2f}")
    print(f"Stock positions: {int((holdings['ticker'] != 'CASH').sum())}")
    print(f"Total value: ${total_value:,.2f}")
    print(f"Cash: ${cash_value:,.2f} ({cash_value / total_value:.2%})")
    print("Saved:", target)
    print("")
    print(holdings.to_string(index=False))


if __name__ == "__main__":
    main()
