import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


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


def get_output_paths(config: dict) -> dict:
    outputs_dir = Path(config["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    return {
        "valuation_ledger": outputs_dir / "paper_portfolio_value_ledger.csv",
        "latest_value_summary": outputs_dir / "latest_portfolio_value_summary.txt",
        "comparison_plot": figures_dir / "paper_portfolio_vs_spy.png",
    }


def compute_current_value_snapshot(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    latest_price_date = pd.Timestamp(prices.index.max()).normalize()

    marked = mark_holdings_to_market(
        holdings=holdings,
        prices=prices,
        as_of_date=latest_price_date,
    )

    total_value = float(marked["market_value"].sum())
    cash_value = float(
        marked.loc[marked["ticker"] == "CASH", "market_value"].sum()
    )
    stock_value = total_value - cash_value
    cash_weight = cash_value / total_value if total_value > 0 else 0.0

    stock_positions = marked[marked["ticker"] != "CASH"].copy()

    if len(stock_positions) > 0:
        top = stock_positions.sort_values("market_value", ascending=False).iloc[0]
        top_ticker = str(top["ticker"])
        top_value = float(top["market_value"])
        top_weight = top_value / total_value if total_value > 0 else 0.0
    else:
        top_ticker = "NONE"
        top_value = 0.0
        top_weight = 0.0

    starting_cash = float(config["paper_trading"]["starting_cash"])
    total_return = total_value / starting_cash - 1.0 if starting_cash > 0 else 0.0

    snapshot = {
        "run_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "valuation_date": latest_price_date,
        "starting_cash": starting_cash,
        "total_portfolio_value": total_value,
        "portfolio_total_return": total_return,
        "stock_value": stock_value,
        "cash_value": cash_value,
        "cash_weight": cash_weight,
        "stock_positions": len(stock_positions),
        "top_holding": top_ticker,
        "top_holding_value": top_value,
        "top_holding_weight": top_weight,
    }

    return marked, snapshot


def append_value_ledger(snapshot: dict, valuation_ledger_path: Path) -> pd.DataFrame:
    new_row = pd.DataFrame([snapshot])

    if valuation_ledger_path.exists():
        existing = pd.read_csv(valuation_ledger_path)

        if "valuation_date" in existing.columns:
            existing["valuation_date"] = pd.to_datetime(
                existing["valuation_date"],
                errors="coerce",
            ).dt.normalize()

            valuation_date = pd.Timestamp(snapshot["valuation_date"]).normalize()
            existing = existing[existing["valuation_date"] != valuation_date].copy()

        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row.copy()

    combined["valuation_date"] = pd.to_datetime(
        combined["valuation_date"],
        errors="coerce",
    ).dt.normalize()

    combined = combined.sort_values("valuation_date").reset_index(drop=True)

    return combined


def build_spy_benchmark(
    value_ledger: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    if "SPY" not in prices.columns:
        raise ValueError("SPY missing from monthly price data.")

    out = value_ledger.copy()
    out["valuation_date"] = pd.to_datetime(out["valuation_date"], errors="coerce").dt.normalize()

    for col in [
        "total_portfolio_value",
        "portfolio_total_return",
        "starting_cash",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    spy_prices = prices["SPY"].copy()
    spy_prices.index = pd.to_datetime(spy_prices.index).normalize()
    spy_prices = spy_prices.sort_index()

    starting_cash = float(config["paper_trading"]["starting_cash"])

    if len(out) == 0:
        return out

    start_date = out["valuation_date"].min()
    available_start_dates = spy_prices[spy_prices.index <= start_date]

    if len(available_start_dates) == 0:
        spy_start_date = spy_prices.index.min()
    else:
        spy_start_date = available_start_dates.index.max()

    spy_start_price = float(spy_prices.loc[spy_start_date])

    spy_values = []

    for date in out["valuation_date"]:
        available = spy_prices[spy_prices.index <= date]

        if len(available) == 0:
            spy_value = None
            spy_return = None
        else:
            price = float(available.iloc[-1])
            spy_return = price / spy_start_price - 1.0
            spy_value = starting_cash * (1.0 + spy_return)

        spy_values.append(
            {
                "valuation_date": date,
                "spy_value": spy_value,
                "spy_total_return": spy_return,
            }
        )

    spy_df = pd.DataFrame(spy_values)

    out = out.drop(
        columns=[
            c for c in [
                "spy_value",
                "spy_total_return",
                "excess_value_vs_spy",
                "excess_return_vs_spy",
            ]
            if c in out.columns
        ],
        errors="ignore",
    )

    merged = out.merge(spy_df, on="valuation_date", how="left")

    merged["spy_value"] = pd.to_numeric(merged["spy_value"], errors="coerce")
    merged["spy_total_return"] = pd.to_numeric(
        merged["spy_total_return"],
        errors="coerce",
    )

    merged["excess_value_vs_spy"] = (
        merged["total_portfolio_value"] - merged["spy_value"]
    )

    merged["excess_return_vs_spy"] = (
        merged["portfolio_total_return"] - merged["spy_total_return"]
    )

    return merged


def save_value_summary(
    marked_holdings: pd.DataFrame,
    comparison: pd.DataFrame,
    snapshot: dict,
    summary_path: Path,
) -> None:
    latest = comparison.sort_values("valuation_date").iloc[-1]

    spy_value = latest.get("spy_value", float("nan"))
    spy_total_return = latest.get("spy_total_return", float("nan"))
    excess_value_vs_spy = latest.get("excess_value_vs_spy", float("nan"))
    excess_return_vs_spy = latest.get("excess_return_vs_spy", float("nan"))

    lines = []
    lines.append("Latent Market Twin Current Portfolio Value")
    lines.append("=========================================")
    lines.append("")
    lines.append(f"Valuation date: {snapshot['valuation_date']}")
    lines.append(f"Total portfolio value: ${snapshot['total_portfolio_value']:,.2f}")
    lines.append(f"Starting cash: ${snapshot['starting_cash']:,.2f}")
    lines.append(f"Portfolio total return: {snapshot['portfolio_total_return']:.2%}")
    lines.append("")
    lines.append(f"SPY benchmark value: ${spy_value:,.2f}")
    lines.append(f"SPY total return: {spy_total_return:.2%}")
    lines.append(f"Excess value vs SPY: ${excess_value_vs_spy:,.2f}")
    lines.append(f"Excess return vs SPY: {excess_return_vs_spy:.2%}")
    lines.append("")
    lines.append(f"Cash value: ${snapshot['cash_value']:,.2f}")
    lines.append(f"Cash weight: {snapshot['cash_weight']:.2%}")
    lines.append(f"Stock positions: {snapshot['stock_positions']}")
    lines.append(f"Top holding: {snapshot['top_holding']}")
    lines.append(f"Top holding value: ${snapshot['top_holding_value']:,.2f}")
    lines.append(f"Top holding weight: {snapshot['top_holding_weight']:.2%}")
    lines.append("")
    lines.append("Current holdings:")
    lines.append(marked_holdings.to_string(index=False))
    lines.append("")
    lines.append("Portfolio value history:")
    lines.append(comparison.to_string(index=False))

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def make_portfolio_vs_spy_plot(
    comparison: pd.DataFrame,
    plot_path: Path,
) -> None:
    if len(comparison) == 0:
        raise ValueError("No valuation history available for plotting.")

    plot_df = comparison.copy()
    plot_df["valuation_date"] = pd.to_datetime(plot_df["valuation_date"])

    plt.figure(figsize=(10, 6))

    plt.plot(
        plot_df["valuation_date"],
        plot_df["total_portfolio_value"],
        marker="o",
        label="Latent Twin Portfolio",
    )

    if "spy_value" in plot_df.columns:
        plt.plot(
            plot_df["valuation_date"],
            plot_df["spy_value"],
            marker="o",
            label="SPY Benchmark",
        )

    plt.title("Paper Portfolio Value vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def main():
    config = load_config(CONFIG_PATH)
    ensure_output_dirs(config)
    check_required_files(config)

    paths = get_output_paths(config)

    prices = load_monthly_prices(config)
    prices.index = pd.to_datetime(prices.index).normalize()

    holdings = load_current_holdings(config)

    marked_holdings, snapshot = compute_current_value_snapshot(
        holdings=holdings,
        prices=prices,
        config=config,
    )

    save_current_holdings(marked_holdings, config)

    value_ledger = append_value_ledger(
        snapshot=snapshot,
        valuation_ledger_path=paths["valuation_ledger"],
    )

    comparison = build_spy_benchmark(
        value_ledger=value_ledger,
        prices=prices,
        config=config,
    )

    comparison.to_csv(paths["valuation_ledger"], index=False)

    save_value_summary(
        marked_holdings=marked_holdings,
        comparison=comparison,
        snapshot=snapshot,
        summary_path=paths["latest_value_summary"],
    )

    make_portfolio_vs_spy_plot(
        comparison=comparison,
        plot_path=paths["comparison_plot"],
    )

    latest = comparison.sort_values("valuation_date").iloc[-1]

    print("")
    print("=" * 100)
    print("CURRENT PAPER PORTFOLIO VALUE")
    print("=" * 100)
    print(f"Valuation date: {snapshot['valuation_date']}")
    print(f"Total portfolio value: ${snapshot['total_portfolio_value']:,.2f}")
    print(f"Portfolio total return: {snapshot['portfolio_total_return']:.2%}")
    print("")
    print(f"SPY benchmark value: ${latest['spy_value']:,.2f}")
    print(f"SPY total return: {latest['spy_total_return']:.2%}")
    print(f"Excess value vs SPY: ${latest['excess_value_vs_spy']:,.2f}")
    print(f"Excess return vs SPY: {latest['excess_return_vs_spy']:.2%}")
    print("")
    print(f"Cash value: ${snapshot['cash_value']:,.2f}")
    print(f"Cash weight: {snapshot['cash_weight']:.2%}")
    print(f"Stock positions: {snapshot['stock_positions']}")
    print(f"Top holding: {snapshot['top_holding']}")
    print(f"Top holding weight: {snapshot['top_holding_weight']:.2%}")
    print("")
    print("Saved value ledger:", paths["valuation_ledger"])
    print("Saved latest value summary:", paths["latest_value_summary"])
    print("Saved plot:", paths["comparison_plot"])
    print("")
    print("CURRENT HOLDINGS")
    print(marked_holdings.to_string(index=False))


if __name__ == "__main__":
    main()