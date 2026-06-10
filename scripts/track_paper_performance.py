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


CONFIG_PATH = "configs/final_model_config.yaml"


def get_paths(config: dict) -> dict:
    outputs_dir = Path(config["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "signals": outputs_dir / "paper_portfolio_signals.csv",
        "orders": outputs_dir / "paper_trade_orders_ledger.csv",
        "performance": outputs_dir / "paper_performance_ledger.csv",
        "latest_summary": outputs_dir / "latest_paper_performance_summary.txt",
    }


def load_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Signal ledger not found: {path}")

    signals = pd.read_csv(path)
    signals["signal_date"] = pd.to_datetime(signals["signal_date"], errors="coerce").dt.normalize()
    signals["ticker"] = signals["ticker"].astype(str).str.strip().str.upper()
    signals["final_weight"] = pd.to_numeric(signals["final_weight"], errors="coerce").fillna(0.0)

    if "model_name" not in signals.columns:
        signals["model_name"] = "unknown_model"

    return signals


def get_next_price_date(prices: pd.DataFrame, signal_date: pd.Timestamp) -> pd.Timestamp | None:
    future_dates = [pd.Timestamp(d).normalize() for d in prices.index if pd.Timestamp(d).normalize() > signal_date]

    if len(future_dates) == 0:
        return None

    return min(future_dates)


def evaluate_signal_month(
    signal_group: pd.DataFrame,
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    evaluation_date: pd.Timestamp,
    model_name: str,
) -> dict:
    signal_date = pd.Timestamp(signal_date).normalize()
    evaluation_date = pd.Timestamp(evaluation_date).normalize()

    price_index = pd.to_datetime(prices.index).normalize()
    prices = prices.copy()
    prices.index = price_index

    if signal_date not in prices.index:
        available = prices[prices.index <= signal_date]

        if len(available) == 0:
            raise ValueError(f"No price data at or before signal_date={signal_date}")

        signal_price_date = pd.Timestamp(available.index.max()).normalize()
    else:
        signal_price_date = signal_date

    if evaluation_date not in prices.index:
        available = prices[prices.index <= evaluation_date]

        if len(available) == 0:
            raise ValueError(f"No price data at or before evaluation_date={evaluation_date}")

        eval_price_date = pd.Timestamp(available.index.max()).normalize()
    else:
        eval_price_date = evaluation_date

    start_prices = prices.loc[signal_price_date]
    end_prices = prices.loc[eval_price_date]

    rows = []

    for _, row in signal_group.iterrows():
        ticker = row["ticker"]
        weight = float(row["final_weight"])

        if ticker == "CASH":
            holding_return = 0.0
            start_price = np.nan
            end_price = np.nan
        elif ticker not in start_prices.index or ticker not in end_prices.index:
            holding_return = np.nan
            start_price = np.nan
            end_price = np.nan
        else:
            start_price = float(start_prices[ticker])
            end_price = float(end_prices[ticker])

            if start_price <= 0 or pd.isna(start_price) or pd.isna(end_price):
                holding_return = np.nan
            else:
                holding_return = end_price / start_price - 1.0

        contribution = weight * holding_return if not pd.isna(holding_return) else np.nan

        rows.append(
            {
                "ticker": ticker,
                "final_weight": weight,
                "start_price": start_price,
                "end_price": end_price,
                "holding_return": holding_return,
                "contribution": contribution,
            }
        )

    holdings_eval = pd.DataFrame(rows)

    usable = holdings_eval.dropna(subset=["contribution"]).copy()
    portfolio_return = float(usable["contribution"].sum())

    missing_tickers = holdings_eval.loc[
        holdings_eval["holding_return"].isna(),
        "ticker",
    ].tolist()

    if "SPY" in start_prices.index and "SPY" in end_prices.index:
        spy_start = float(start_prices["SPY"])
        spy_end = float(end_prices["SPY"])
        spy_return = spy_end / spy_start - 1.0 if spy_start > 0 else np.nan
    else:
        spy_start = np.nan
        spy_end = np.nan
        spy_return = np.nan

    excess_return = portfolio_return - spy_return if not pd.isna(spy_return) else np.nan

    stock_rows = holdings_eval[holdings_eval["ticker"] != "CASH"].copy()

    if len(stock_rows.dropna(subset=["holding_return"])) > 0:
        best_row = stock_rows.sort_values("holding_return", ascending=False).iloc[0]
        worst_row = stock_rows.sort_values("holding_return", ascending=True).iloc[0]

        best_ticker = best_row["ticker"]
        best_return = float(best_row["holding_return"])
        worst_ticker = worst_row["ticker"]
        worst_return = float(worst_row["holding_return"])
    else:
        best_ticker = None
        best_return = np.nan
        worst_ticker = None
        worst_return = np.nan

    return {
        "model_name": model_name,
        "signal_date": signal_date,
        "signal_price_date": signal_price_date,
        "evaluation_date": evaluation_date,
        "evaluation_price_date": eval_price_date,
        "portfolio_return": portfolio_return,
        "spy_return": spy_return,
        "excess_return": excess_return,
        "beat_spy": bool(portfolio_return > spy_return) if not pd.isna(spy_return) else False,
        "holding_count": int(len(signal_group[signal_group["ticker"] != "CASH"])),
        "cash_weight": float(signal_group.loc[signal_group["ticker"] == "CASH", "final_weight"].sum()),
        "best_ticker": best_ticker,
        "best_holding_return": best_return,
        "worst_ticker": worst_ticker,
        "worst_holding_return": worst_return,
        "missing_tickers": ", ".join(missing_tickers),
        "run_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }


def add_cumulative_stats(performance: pd.DataFrame, starting_cash: float) -> pd.DataFrame:
    out = performance.copy()
    out = out.sort_values(["model_name", "signal_date"]).reset_index(drop=True)

    cumulative_rows = []

    for model_name, group in out.groupby("model_name"):
        group = group.sort_values("signal_date").copy()

        group["portfolio_cumulative_value"] = starting_cash * (1.0 + group["portfolio_return"]).cumprod()
        group["spy_cumulative_value"] = starting_cash * (1.0 + group["spy_return"]).cumprod()

        group["portfolio_cumulative_return"] = group["portfolio_cumulative_value"] / starting_cash - 1.0
        group["spy_cumulative_return"] = group["spy_cumulative_value"] / starting_cash - 1.0
        group["cumulative_excess_return"] = group["portfolio_cumulative_return"] - group["spy_cumulative_return"]

        running_max = group["portfolio_cumulative_value"].cummax()
        group["portfolio_drawdown"] = group["portfolio_cumulative_value"] / running_max - 1.0

        cumulative_rows.append(group)

    return pd.concat(cumulative_rows, ignore_index=True)


def update_performance_ledger(
    new_rows: list[dict],
    performance_path: Path,
    starting_cash: float,
) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)

    if performance_path.exists():
        existing = pd.read_csv(performance_path)
        existing["signal_date"] = pd.to_datetime(existing["signal_date"], errors="coerce").dt.normalize()
        existing["evaluation_date"] = pd.to_datetime(existing["evaluation_date"], errors="coerce").dt.normalize()

        new_df["signal_date"] = pd.to_datetime(new_df["signal_date"], errors="coerce").dt.normalize()
        new_df["evaluation_date"] = pd.to_datetime(new_df["evaluation_date"], errors="coerce").dt.normalize()

        existing_key = existing["model_name"].astype(str) + "|" + existing["signal_date"].astype(str)
        new_key = new_df["model_name"].astype(str) + "|" + new_df["signal_date"].astype(str)

        existing = existing[~existing_key.isin(set(new_key))].copy()

        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    combined["signal_date"] = pd.to_datetime(combined["signal_date"], errors="coerce").dt.normalize()
    combined["evaluation_date"] = pd.to_datetime(combined["evaluation_date"], errors="coerce").dt.normalize()

    combined = combined.sort_values(["model_name", "signal_date"]).reset_index(drop=True)
    combined = add_cumulative_stats(combined, starting_cash=starting_cash)

    combined.to_csv(performance_path, index=False)

    return combined


def write_summary(performance: pd.DataFrame, summary_path: Path) -> None:
    latest = performance.sort_values("evaluation_date").tail(1).iloc[0]

    lines = []
    lines.append("Latent Market Twin Paper Performance Summary")
    lines.append("===========================================")
    lines.append("")
    lines.append(f"Latest signal date: {latest['signal_date']}")
    lines.append(f"Latest evaluation date: {latest['evaluation_date']}")
    lines.append("")
    lines.append(f"Latest portfolio return: {latest['portfolio_return']:.2%}")
    lines.append(f"Latest SPY return: {latest['spy_return']:.2%}")
    lines.append(f"Latest excess return: {latest['excess_return']:.2%}")
    lines.append(f"Beat SPY: {latest['beat_spy']}")
    lines.append("")
    lines.append(f"Cumulative portfolio value: ${latest['portfolio_cumulative_value']:,.2f}")
    lines.append(f"Cumulative SPY value: ${latest['spy_cumulative_value']:,.2f}")
    lines.append(f"Cumulative portfolio return: {latest['portfolio_cumulative_return']:.2%}")
    lines.append(f"Cumulative SPY return: {latest['spy_cumulative_return']:.2%}")
    lines.append(f"Cumulative excess return: {latest['cumulative_excess_return']:.2%}")
    lines.append(f"Current drawdown: {latest['portfolio_drawdown']:.2%}")
    lines.append("")
    lines.append(f"Best holding: {latest['best_ticker']} ({latest['best_holding_return']:.2%})")
    lines.append(f"Worst holding: {latest['worst_ticker']} ({latest['worst_holding_return']:.2%})")
    lines.append("")
    lines.append("Full performance ledger:")
    lines.append(performance.to_string(index=False))

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    config = load_config(CONFIG_PATH)
    ensure_output_dirs(config)
    check_required_files(config)

    paths = get_paths(config)

    signals = load_signals(paths["signals"])
    prices = load_monthly_prices(config)
    prices.index = pd.to_datetime(prices.index).normalize()

    starting_cash = float(config["paper_trading"]["starting_cash"])

    new_rows = []

    for (model_name, signal_date), group in signals.groupby(["model_name", "signal_date"]):
        signal_date = pd.Timestamp(signal_date).normalize()
        evaluation_date = get_next_price_date(prices, signal_date)

        if evaluation_date is None:
            print(f"Skipping signal_date={signal_date.date()} because no later price date exists yet.")
            continue

        row = evaluate_signal_month(
            signal_group=group,
            prices=prices,
            signal_date=signal_date,
            evaluation_date=evaluation_date,
            model_name=model_name,
        )

        new_rows.append(row)

    if len(new_rows) == 0:
        print("")
        print("=" * 100)
        print("PAPER PERFORMANCE TRACKER")
        print("=" * 100)
        print("No completed holding periods available yet.")
        print("This is expected if the latest signal date is also the latest price date.")
        return

    performance = update_performance_ledger(
        new_rows=new_rows,
        performance_path=paths["performance"],
        starting_cash=starting_cash,
    )

    write_summary(performance, paths["latest_summary"])

    latest = performance.sort_values("evaluation_date").tail(1).iloc[0]

    print("")
    print("=" * 100)
    print("PAPER PERFORMANCE TRACKER")
    print("=" * 100)
    print(f"Latest signal date: {latest['signal_date']}")
    print(f"Latest evaluation date: {latest['evaluation_date']}")
    print("")
    print(f"Latest portfolio return: {latest['portfolio_return']:.2%}")
    print(f"Latest SPY return: {latest['spy_return']:.2%}")
    print(f"Latest excess return: {latest['excess_return']:.2%}")
    print(f"Beat SPY: {latest['beat_spy']}")
    print("")
    print(f"Cumulative portfolio value: ${latest['portfolio_cumulative_value']:,.2f}")
    print(f"Cumulative SPY value: ${latest['spy_cumulative_value']:,.2f}")
    print(f"Cumulative excess return: {latest['cumulative_excess_return']:.2%}")
    print(f"Current drawdown: {latest['portfolio_drawdown']:.2%}")
    print("")
    print(f"Best holding: {latest['best_ticker']} ({latest['best_holding_return']:.2%})")
    print(f"Worst holding: {latest['worst_ticker']} ({latest['worst_holding_return']:.2%})")
    print("")
    print("Saved performance ledger:", paths["performance"])
    print("Saved latest summary:", paths["latest_summary"])
    print("")
    print("FULL PERFORMANCE LEDGER")
    print(performance.to_string(index=False))


if __name__ == "__main__":
    main()