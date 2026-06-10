import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, ensure_output_dirs, check_required_files
from src.data.loaders import load_monthly_prices
from src.paper_trading.holdings import (
    load_current_holdings,
    mark_holdings_to_market,
    save_current_holdings,
)


CONFIG_PATH = "configs/final_model_config.yaml"


def get_paths(config: dict) -> dict:
    outputs_dir = Path(config["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "signals": outputs_dir / "paper_portfolio_signals.csv",
        "holdings": outputs_dir / "current_paper_holdings.csv",
        "rebalance_ledger": outputs_dir / "paper_rebalance_orders_ledger.csv",
        "latest_rebalance": outputs_dir / "latest_rebalance_orders.csv",
        "latest_summary": outputs_dir / "latest_rebalance_summary.txt",
    }


def load_latest_target_portfolio(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Signal ledger not found: {path}")

    signals = pd.read_csv(path)
    signals["signal_date"] = pd.to_datetime(
        signals["signal_date"],
        errors="coerce",
    ).dt.normalize()

    signals["ticker"] = signals["ticker"].astype(str).str.strip().str.upper()
    signals["final_weight"] = pd.to_numeric(
        signals["final_weight"],
        errors="coerce",
    ).fillna(0.0)

    if "model_name" in signals.columns:
        signals = signals[signals["model_name"] == model_name].copy()

    if len(signals) == 0:
        raise ValueError(f"No target signals found for model: {model_name}")

    latest_signal_date = signals["signal_date"].max()

    target = signals[signals["signal_date"] == latest_signal_date].copy()

    keep_cols = [
        "signal_date",
        "ticker",
        "final_weight",
        "branches",
        "best_rank",
        "avg_ranker_score",
        "regime_risk_on",
        "tech_drawdown",
    ]

    target = target[[c for c in keep_cols if c in target.columns]].copy()
    target = target.sort_values("final_weight", ascending=False).reset_index(drop=True)

    return target


def get_latest_prices(
    prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> tuple[pd.Series, pd.Timestamp]:
    as_of_date = pd.Timestamp(as_of_date).normalize()

    available = prices[prices.index <= as_of_date].copy()

    if len(available) == 0:
        raise ValueError(f"No prices available at or before {as_of_date}")

    price_date = pd.Timestamp(available.index.max()).normalize()
    latest_prices = available.loc[price_date]

    return latest_prices, price_date


def build_rebalance_orders(
    holdings: pd.DataFrame,
    target: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    model_name = config["final_model_name"]
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    signal_date = pd.Timestamp(target["signal_date"].iloc[0]).normalize()
    latest_prices, price_date = get_latest_prices(prices, signal_date)

    holdings = holdings.copy()
    holdings["ticker"] = holdings["ticker"].astype(str).str.strip().str.upper()
    holdings["shares"] = pd.to_numeric(holdings["shares"], errors="coerce").fillna(0.0)
    holdings["market_value"] = pd.to_numeric(
        holdings["market_value"],
        errors="coerce",
    ).fillna(0.0)

    target = target.copy()
    target["ticker"] = target["ticker"].astype(str).str.strip().str.upper()
    target["final_weight"] = pd.to_numeric(
        target["final_weight"],
        errors="coerce",
    ).fillna(0.0)

    portfolio_value = float(holdings["market_value"].sum())

    tickers = sorted(set(holdings["ticker"].tolist()) | set(target["ticker"].tolist()))

    rows = []

    for ticker in tickers:
        current_row = holdings[holdings["ticker"] == ticker]
        target_row = target[target["ticker"] == ticker]

        current_shares = float(current_row["shares"].sum()) if len(current_row) > 0 else 0.0
        current_value = float(current_row["market_value"].sum()) if len(current_row) > 0 else 0.0
        current_weight = current_value / portfolio_value if portfolio_value > 0 else 0.0

        target_weight = float(target_row["final_weight"].sum()) if len(target_row) > 0 else 0.0
        target_value = portfolio_value * target_weight

        if ticker == "CASH":
            latest_price = 1.0
            target_shares = target_value
            raw_trade_shares = target_value - current_shares
            rounded_trade_shares = raw_trade_shares
            trade_value = rounded_trade_shares
            action = "CASH_ADJUST"
        else:
            if ticker not in latest_prices.index:
                latest_price = np.nan
                target_shares = np.nan
                raw_trade_shares = np.nan
                rounded_trade_shares = np.nan
                trade_value = np.nan
                action = "MISSING_PRICE"
            else:
                latest_price = float(latest_prices[ticker])

                if latest_price <= 0 or pd.isna(latest_price):
                    target_shares = np.nan
                    raw_trade_shares = np.nan
                    rounded_trade_shares = np.nan
                    trade_value = np.nan
                    action = "BAD_PRICE"
                else:
                    target_shares = target_value / latest_price
                    raw_trade_shares = target_shares - current_shares

                    if raw_trade_shares > 0:
                        rounded_trade_shares = float(np.floor(raw_trade_shares))
                    else:
                        rounded_trade_shares = float(np.ceil(raw_trade_shares))

                    trade_value = rounded_trade_shares * latest_price

                    if rounded_trade_shares > 0:
                        action = "BUY"
                    elif rounded_trade_shares < 0:
                        action = "SELL"
                    else:
                        action = "HOLD"

        branches = ""
        best_rank = np.nan

        if len(target_row) > 0:
            branches = target_row["branches"].iloc[0] if "branches" in target_row.columns else ""
            best_rank = target_row["best_rank"].iloc[0] if "best_rank" in target_row.columns else np.nan

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
                "trade_shares": rounded_trade_shares,
                "latest_price": latest_price,
                "current_value": current_value,
                "target_value": target_value,
                "trade_value": trade_value,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "weight_diff": target_weight - current_weight,
                "branches": branches,
                "best_rank": best_rank,
            }
        )

    orders = pd.DataFrame(rows)

    # Estimate cash after trades. Buys have positive trade_value, sells negative.
    non_cash_orders = orders[orders["ticker"] != "CASH"].copy()
    cash_row = orders[orders["ticker"] == "CASH"].copy()

    current_cash = float(
        holdings.loc[holdings["ticker"] == "CASH", "market_value"].sum()
    )

    total_buy_value = float(
        non_cash_orders.loc[non_cash_orders["trade_value"] > 0, "trade_value"].sum(skipna=True)
    )
    total_sell_value = float(
        -non_cash_orders.loc[non_cash_orders["trade_value"] < 0, "trade_value"].sum(skipna=True)
    )

    estimated_cash_after_trades = current_cash + total_sell_value - total_buy_value

    orders["portfolio_value"] = portfolio_value
    orders["current_cash"] = current_cash
    orders["total_buy_value"] = total_buy_value
    orders["total_sell_value"] = total_sell_value
    orders["estimated_cash_after_trades"] = estimated_cash_after_trades

    # For display, keep active orders first.
    action_order = {
        "SELL": 0,
        "BUY": 1,
        "HOLD": 2,
        "CASH_ADJUST": 3,
        "MISSING_PRICE": 4,
        "BAD_PRICE": 5,
    }

    orders["action_sort"] = orders["action"].map(action_order).fillna(9)

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
    }

    return orders, summary


def save_rebalance_outputs(
    orders: pd.DataFrame,
    summary: dict,
    paths: dict,
) -> None:
    orders.to_csv(paths["latest_rebalance"], index=False)

    if paths["rebalance_ledger"].exists():
        existing = pd.read_csv(paths["rebalance_ledger"])
        existing["signal_date"] = pd.to_datetime(
            existing["signal_date"],
            errors="coerce",
        ).dt.normalize()

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

    combined["signal_date"] = pd.to_datetime(
        combined["signal_date"],
        errors="coerce",
    ).dt.normalize()

    combined = combined.sort_values(
        ["signal_date", "model_name", "action", "ticker"],
    ).reset_index(drop=True)

    combined.to_csv(paths["rebalance_ledger"], index=False)

    active_orders = orders[orders["action"].isin(["BUY", "SELL"])].copy()

    lines = []
    lines.append("Latent Market Twin Rebalance Summary")
    lines.append("===================================")
    lines.append("")
    lines.append(f"Run timestamp: {summary['run_timestamp']}")
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
    lines.append("")
    lines.append("Active rebalance orders:")
    if len(active_orders) == 0:
        lines.append("No active buy/sell orders. Portfolio is already close to target.")
    else:
        lines.append(active_orders.to_string(index=False))
    lines.append("")
    lines.append("Full rebalance table:")
    lines.append(orders.to_string(index=False))

    with open(paths["latest_summary"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    config = load_config(CONFIG_PATH)
    ensure_output_dirs(config)
    check_required_files(config)

    paths = get_paths(config)

    prices = load_monthly_prices(config)
    prices.index = pd.to_datetime(prices.index).normalize()

    holdings = load_current_holdings(config)

    # Mark holdings to latest available model signal/price date before rebalance.
    target = load_latest_target_portfolio(
        paths["signals"],
        model_name=config["final_model_name"],
    )

    signal_date = pd.Timestamp(target["signal_date"].iloc[0]).normalize()

    holdings = mark_holdings_to_market(
        holdings=holdings,
        prices=prices,
        as_of_date=signal_date,
    )

    save_current_holdings(holdings, config)

    orders, summary = build_rebalance_orders(
        holdings=holdings,
        target=target,
        prices=prices,
        config=config,
    )

    save_rebalance_outputs(
        orders=orders,
        summary=summary,
        paths=paths,
    )

    active_orders = orders[orders["action"].isin(["BUY", "SELL"])].copy()

    print("")
    print("=" * 100)
    print("REBALANCE ORDER GENERATOR")
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
    print("")
    print("ACTIVE REBALANCE ORDERS")
    if len(active_orders) == 0:
        print("No active buy/sell orders. Portfolio is already close to target.")
    else:
        print(active_orders.to_string(index=False))
    print("")
    print("Saved latest rebalance:", paths["latest_rebalance"])
    print("Saved rebalance ledger:", paths["rebalance_ledger"])
    print("Saved rebalance summary:", paths["latest_summary"])


if __name__ == "__main__":
    main()