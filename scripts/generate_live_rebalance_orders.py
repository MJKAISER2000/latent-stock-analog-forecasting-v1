import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, ensure_output_dirs
from src.data.loaders import load_monthly_prices


CONFIG_PATH = "configs/live_model_config.yaml"

LIVE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_trading_live"
LIVE_HOLDINGS_PATH = LIVE_OUTPUT_DIR / "current_live_holdings.csv"
LIVE_SIGNALS_PATH = LIVE_OUTPUT_DIR / "live_portfolio_signals.csv"

LATEST_REBALANCE_PATH = LIVE_OUTPUT_DIR / "latest_live_rebalance_orders.csv"
REBALANCE_LEDGER_PATH = LIVE_OUTPUT_DIR / "live_rebalance_orders_ledger.csv"
REBALANCE_SUMMARY_PATH = LIVE_OUTPUT_DIR / "latest_live_rebalance_summary.txt"


def load_live_holdings() -> pd.DataFrame:
    if not LIVE_HOLDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Live holdings not found: {LIVE_HOLDINGS_PATH}. "
            "Run scripts/initialize_live_holdings.py first."
        )

    holdings = pd.read_csv(LIVE_HOLDINGS_PATH)
    holdings["ticker"] = holdings["ticker"].astype(str).str.strip().str.upper()
    holdings["shares"] = pd.to_numeric(holdings["shares"], errors="coerce").fillna(0.0)
    holdings["market_value"] = pd.to_numeric(holdings["market_value"], errors="coerce").fillna(0.0)

    return holdings


def load_latest_live_target() -> pd.DataFrame:
    if not LIVE_SIGNALS_PATH.exists():
        raise FileNotFoundError(
            f"Live signals not found: {LIVE_SIGNALS_PATH}. "
            "Run scripts/run_live_final_pipeline.py first."
        )

    signals = pd.read_csv(LIVE_SIGNALS_PATH)
    signals["signal_date"] = pd.to_datetime(signals["signal_date"], errors="coerce").dt.normalize()
    signals["ticker"] = signals["ticker"].astype(str).str.strip().str.upper()
    signals["final_weight"] = pd.to_numeric(signals["final_weight"], errors="coerce").fillna(0.0)

    latest_signal_date = signals["signal_date"].max()
    target = signals[signals["signal_date"] == latest_signal_date].copy()

    target = target.sort_values("final_weight", ascending=False).reset_index(drop=True)

    return target


def mark_holdings_with_model_prices(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    holdings = holdings.copy()
    as_of_date = pd.Timestamp(as_of_date).normalize()

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).normalize()

    available = prices[prices.index <= as_of_date]

    if len(available) == 0:
        raise ValueError(f"No live monthly prices available at or before {as_of_date}")

    price_date = pd.Timestamp(available.index.max()).normalize()
    latest_prices = available.loc[price_date]

    rows = []

    for _, row in holdings.iterrows():
        ticker = row["ticker"]
        shares = float(row["shares"])

        if ticker == "CASH":
            last_price = 1.0
            market_value = shares
        elif ticker in latest_prices.index:
            last_price = float(latest_prices[ticker])
            market_value = shares * last_price
        else:
            last_price = np.nan
            market_value = np.nan

        updated = row.to_dict()
        updated["last_price"] = last_price
        updated["market_value"] = market_value
        updated["last_price_date"] = price_date
        rows.append(updated)

    out = pd.DataFrame(rows)

    total_value = float(out["market_value"].sum(skipna=True))

    if total_value > 0:
        out["current_weight"] = out["market_value"] / total_value
    else:
        out["current_weight"] = 0.0

    return out.sort_values("market_value", ascending=False).reset_index(drop=True), price_date


def build_live_rebalance_orders(
    holdings: pd.DataFrame,
    target: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config["final_model_name"]

    signal_date = pd.Timestamp(target["signal_date"].iloc[0]).normalize()

    marked_holdings, price_date = mark_holdings_with_model_prices(
        holdings=holdings,
        prices=prices,
        as_of_date=signal_date,
    )

    latest_prices = prices.loc[price_date]

    portfolio_value = float(marked_holdings["market_value"].sum(skipna=True))
    current_cash = float(
        marked_holdings.loc[marked_holdings["ticker"] == "CASH", "market_value"].sum()
    )

    all_tickers = sorted(set(marked_holdings["ticker"].tolist()) | set(target["ticker"].tolist()))

    rows = []

    for ticker in all_tickers:
        current = marked_holdings[marked_holdings["ticker"] == ticker]
        desired = target[target["ticker"] == ticker]

        current_shares = float(current["shares"].sum()) if len(current) else 0.0
        current_value = float(current["market_value"].sum()) if len(current) else 0.0
        current_weight = current_value / portfolio_value if portfolio_value > 0 else 0.0

        target_weight = float(desired["final_weight"].sum()) if len(desired) else 0.0
        target_value = portfolio_value * target_weight

        branches = ""
        best_rank = np.nan
        avg_ranker_score = np.nan

        if len(desired):
            if "branches" in desired.columns:
                branches = desired["branches"].iloc[0]
            if "best_rank" in desired.columns:
                best_rank = desired["best_rank"].iloc[0]
            if "avg_ranker_score" in desired.columns:
                avg_ranker_score = desired["avg_ranker_score"].iloc[0]

        if ticker == "CASH":
            latest_price = 1.0
            target_shares = target_value
            raw_trade_shares = target_value - current_shares
            trade_shares = raw_trade_shares
            trade_value = trade_shares
            action = "CASH_ADJUST"

        elif ticker not in latest_prices.index:
            latest_price = np.nan
            target_shares = np.nan
            raw_trade_shares = np.nan
            trade_shares = np.nan
            trade_value = np.nan
            action = "MISSING_PRICE"

        else:
            latest_price = float(latest_prices[ticker])

            if pd.isna(latest_price) or latest_price <= 0:
                target_shares = np.nan
                raw_trade_shares = np.nan
                trade_shares = np.nan
                trade_value = np.nan
                action = "BAD_PRICE"
            else:
                target_shares = target_value / latest_price
                raw_trade_shares = target_shares - current_shares

                if raw_trade_shares > 0:
                    trade_shares = float(np.floor(raw_trade_shares))
                else:
                    trade_shares = float(np.ceil(raw_trade_shares))

                trade_value = trade_shares * latest_price

                if trade_shares > 0:
                    action = "BUY"
                elif trade_shares < 0:
                    action = "SELL"
                else:
                    action = "HOLD"

        rows.append(
            {
                "run_timestamp": run_timestamp,
                "model_name": model_name,
                "signal_date": signal_date,
                "price_date": price_date,
                "ticker": ticker,
                "action": action,
                "current_shares": current_shares,
                "target_shares": target_shares,
                "trade_shares": trade_shares,
                "latest_price": latest_price,
                "current_value": current_value,
                "target_value": target_value,
                "trade_value": trade_value,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "weight_diff": target_weight - current_weight,
                "branches": branches,
                "best_rank": best_rank,
                "avg_ranker_score": avg_ranker_score,
            }
        )

    orders = pd.DataFrame(rows)

    non_cash = orders[orders["ticker"] != "CASH"].copy()

    total_buy_value = float(
        non_cash.loc[
            (non_cash["action"] == "BUY") & (non_cash["trade_value"] > 0),
            "trade_value",
        ].sum(skipna=True)
    )

    total_sell_value = float(
        -non_cash.loc[
            (non_cash["action"] == "SELL") & (non_cash["trade_value"] < 0),
            "trade_value",
        ].sum(skipna=True)
    )

    estimated_cash_after_trades = current_cash + total_sell_value - total_buy_value

    orders["portfolio_value"] = portfolio_value
    orders["current_cash"] = current_cash
    orders["total_buy_value"] = total_buy_value
    orders["total_sell_value"] = total_sell_value
    orders["estimated_cash_after_trades"] = estimated_cash_after_trades

    sort_map = {
        "SELL": 0,
        "BUY": 1,
        "HOLD": 2,
        "CASH_ADJUST": 3,
        "MISSING_PRICE": 4,
        "BAD_PRICE": 5,
    }

    orders["action_sort"] = orders["action"].map(sort_map).fillna(9)

    orders = orders.sort_values(
        ["action_sort", "ticker"],
        ascending=[True, True],
    ).drop(columns=["action_sort"]).reset_index(drop=True)

    summary = {
        "run_timestamp": run_timestamp,
        "model_name": model_name,
        "signal_date": signal_date,
        "price_date": price_date,
        "portfolio_value": portfolio_value,
        "current_cash": current_cash,
        "total_buy_value": total_buy_value,
        "total_sell_value": total_sell_value,
        "estimated_cash_after_trades": estimated_cash_after_trades,
        "buy_order_count": int((orders["action"] == "BUY").sum()),
        "sell_order_count": int((orders["action"] == "SELL").sum()),
        "hold_count": int((orders["action"] == "HOLD").sum()),
        "missing_price_count": int((orders["action"] == "MISSING_PRICE").sum()),
        "bad_price_count": int((orders["action"] == "BAD_PRICE").sum()),
    }

    return orders, summary


def save_outputs(orders: pd.DataFrame, summary: dict) -> None:
    LIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    orders.to_csv(LATEST_REBALANCE_PATH, index=False)

    if REBALANCE_LEDGER_PATH.exists():
        existing = pd.read_csv(REBALANCE_LEDGER_PATH)
        existing["signal_date"] = pd.to_datetime(existing["signal_date"], errors="coerce").dt.normalize()

        signal_date = pd.Timestamp(summary["signal_date"]).normalize()
        model_name = summary["model_name"]

        keep = ~(
            (existing["model_name"] == model_name)
            & (existing["signal_date"] == signal_date)
        )

        existing = existing[keep].copy()
        combined = pd.concat([existing, orders], ignore_index=True)
    else:
        combined = orders.copy()

    combined["signal_date"] = pd.to_datetime(combined["signal_date"], errors="coerce").dt.normalize()
    combined = combined.sort_values(["signal_date", "model_name", "action", "ticker"]).reset_index(drop=True)
    combined.to_csv(REBALANCE_LEDGER_PATH, index=False)

    active = orders[orders["action"].isin(["BUY", "SELL"])].copy()

    lines = []
    lines.append("LTSAF Live Rebalance Summary")
    lines.append("===========================")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model: {summary['model_name']}")
    lines.append(f"Signal date: {summary['signal_date']}")
    lines.append(f"Price date: {summary['price_date']}")
    lines.append("")
    lines.append(f"Portfolio value: ${summary['portfolio_value']:,.2f}")
    lines.append(f"Current cash: ${summary['current_cash']:,.2f}")
    lines.append(f"Total buy value: ${summary['total_buy_value']:,.2f}")
    lines.append(f"Total sell value: ${summary['total_sell_value']:,.2f}")
    lines.append(f"Estimated cash after trades: ${summary['estimated_cash_after_trades']:,.2f}")
    lines.append("")
    lines.append(f"Buy orders: {summary['buy_order_count']}")
    lines.append(f"Sell orders: {summary['sell_order_count']}")
    lines.append(f"Hold count: {summary['hold_count']}")
    lines.append(f"Missing price count: {summary['missing_price_count']}")
    lines.append(f"Bad price count: {summary['bad_price_count']}")
    lines.append("")
    lines.append("Active orders:")
    if len(active) == 0:
        lines.append("No active buy/sell orders.")
    else:
        lines.append(active.to_string(index=False))
    lines.append("")
    lines.append("Full rebalance table:")
    lines.append(orders.to_string(index=False))

    with open(REBALANCE_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    config = load_config(CONFIG_PATH)
    ensure_output_dirs(config)

    holdings = load_live_holdings()
    target = load_latest_live_target()
    prices = load_monthly_prices(config)
    prices.index = pd.to_datetime(prices.index).normalize()

    orders, summary = build_live_rebalance_orders(
        holdings=holdings,
        target=target,
        prices=prices,
        config=config,
    )

    save_outputs(orders, summary)

    active = orders[orders["action"].isin(["BUY", "SELL"])].copy()

    print("")
    print("=" * 100)
    print("LTSAF LIVE REBALANCE ORDERS")
    print("=" * 100)
    print(f"Signal date: {summary['signal_date']}")
    print(f"Portfolio value: ${summary['portfolio_value']:,.2f}")
    print(f"Current cash: ${summary['current_cash']:,.2f}")
    print(f"Total buy value: ${summary['total_buy_value']:,.2f}")
    print(f"Total sell value: ${summary['total_sell_value']:,.2f}")
    print(f"Estimated cash after trades: ${summary['estimated_cash_after_trades']:,.2f}")
    print("")
    print(f"Buy orders: {summary['buy_order_count']}")
    print(f"Sell orders: {summary['sell_order_count']}")
    print(f"Hold count: {summary['hold_count']}")
    print(f"Missing price count: {summary['missing_price_count']}")
    print(f"Bad price count: {summary['bad_price_count']}")
    print("")
    print("ACTIVE ORDERS")
    if len(active) == 0:
        print("No active buy/sell orders.")
    else:
        print(active.to_string(index=False))
    print("")
    print("Saved latest rebalance:", LATEST_REBALANCE_PATH)
    print("Saved rebalance ledger:", REBALANCE_LEDGER_PATH)
    print("Saved summary:", REBALANCE_SUMMARY_PATH)


if __name__ == "__main__":
    main()