import os
import sys
import glob
from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, ensure_output_dirs


CONFIG_PATH = "configs/live_model_config.yaml"


def get_paths(config: dict) -> dict:
    outputs_dir = PROJECT_ROOT / config["paths"]["outputs_dir"]
    outputs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "outputs_dir": outputs_dir,
        "live_order_ledger": outputs_dir / "live_order_ledger.csv",
        "current_live_holdings": outputs_dir / "current_live_holdings.csv",
        "live_holdings_summary": outputs_dir / "latest_live_holdings_summary.txt",
    }


def load_latest_live_orders(paths: dict) -> pd.DataFrame:
    ledger_path = paths["live_order_ledger"]

    if ledger_path.exists():
        orders = pd.read_csv(ledger_path)
    else:
        pattern = str(paths["outputs_dir"] / "live_paper_trade_orders_*.csv")
        files = sorted(glob.glob(pattern))

        if not files:
            raise FileNotFoundError(
                "No live order ledger or timestamped live order files found. "
                "Run scripts/run_live_final_pipeline.py first."
            )

        orders = pd.read_csv(files[-1])

    orders["ticker"] = orders["ticker"].astype(str).str.strip().str.upper()
    orders["signal_date"] = pd.to_datetime(orders["signal_date"], errors="coerce").dt.normalize()
    orders["price_date"] = pd.to_datetime(orders["price_date"], errors="coerce").dt.normalize()

    latest_signal_date = orders["signal_date"].max()
    latest = orders[orders["signal_date"] == latest_signal_date].copy()

    return latest


def initialize_holdings_from_live_orders(
    orders: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    starting_cash = float(config["paper_trading"]["starting_cash"])

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
                "last_updated": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "source": "live_initial_orders",
            }
        )

    holdings = pd.DataFrame(rows)

    invested = float(holdings["market_value"].sum()) if len(holdings) > 0 else 0.0

    explicit_cash = float(
        orders.loc[orders["ticker"] == "CASH", "target_dollars"].sum()
        if "target_dollars" in orders.columns
        else 0.0
    )

    leftover_cash = float(
        orders["leftover_cash_from_rounding"].iloc[0]
        if "leftover_cash_from_rounding" in orders.columns
        else max(0.0, starting_cash - invested)
    )

    cash = explicit_cash + leftover_cash

    if cash <= 0:
        cash = max(0.0, starting_cash - invested)

    cash_row = {
        "ticker": "CASH",
        "shares": cash,
        "avg_entry_price": 1.0,
        "last_price": 1.0,
        "market_value": cash,
        "last_signal_date": orders["signal_date"].iloc[0],
        "last_price_date": orders["price_date"].iloc[0],
        "last_updated": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "source": "live_initial_cash",
    }

    holdings = pd.concat([holdings, pd.DataFrame([cash_row])], ignore_index=True)

    total_value = float(holdings["market_value"].sum())

    if total_value > 0:
        holdings["current_weight"] = holdings["market_value"] / total_value
    else:
        holdings["current_weight"] = 0.0

    holdings = holdings.sort_values("market_value", ascending=False).reset_index(drop=True)

    return holdings


def write_summary(holdings: pd.DataFrame, path: Path) -> None:
    stock_holdings = holdings[holdings["ticker"] != "CASH"].copy()
    total_value = float(holdings["market_value"].sum())
    cash_value = float(holdings.loc[holdings["ticker"] == "CASH", "market_value"].sum())
    cash_weight = cash_value / total_value if total_value > 0 else 0.0

    top = stock_holdings.sort_values("market_value", ascending=False).head(10)

    lines = []
    lines.append("LTSAF Live Holdings Summary")
    lines.append("==========================")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total value: ${total_value:,.2f}")
    lines.append(f"Cash value: ${cash_value:,.2f}")
    lines.append(f"Cash weight: {cash_weight:.2%}")
    lines.append(f"Stock positions: {len(stock_holdings)}")
    lines.append("")
    lines.append("Top holdings:")
    lines.append(top.to_string(index=False))
    lines.append("")
    lines.append("Full holdings:")
    lines.append(holdings.to_string(index=False))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    config = load_config(CONFIG_PATH)
    ensure_output_dirs(config)

    paths = get_paths(config)

    if paths["current_live_holdings"].exists():
        print("")
        print("=" * 100)
        print("LIVE HOLDINGS ALREADY EXISTS")
        print("=" * 100)
        print("File:", paths["current_live_holdings"])
        print("")
        print("Not overwriting existing live holdings.")
        print("Delete the file manually if you intentionally want to reinitialize.")
        return

    orders = load_latest_live_orders(paths)
    holdings = initialize_holdings_from_live_orders(orders, config)

    holdings.to_csv(paths["current_live_holdings"], index=False)
    write_summary(holdings, paths["live_holdings_summary"])

    print("")
    print("=" * 100)
    print("LIVE HOLDINGS INITIALIZED")
    print("=" * 100)
    print("Saved:", paths["current_live_holdings"])
    print("Saved summary:", paths["live_holdings_summary"])
    print("")
    print(holdings.to_string(index=False))
    print("")
    print("Total value:", holdings["market_value"].sum())


if __name__ == "__main__":
    main()