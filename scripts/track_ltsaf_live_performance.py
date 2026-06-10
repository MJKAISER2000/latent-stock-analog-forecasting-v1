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

LIVE_SIGNALS_PATH = LIVE_OUTPUT_DIR / "live_portfolio_signals.csv"
LIVE_PERFORMANCE_LEDGER_PATH = LIVE_OUTPUT_DIR / "live_performance_ledger.csv"
LIVE_PERFORMANCE_SUMMARY_PATH = LIVE_OUTPUT_DIR / "latest_live_performance_summary.txt"


def load_live_signals() -> pd.DataFrame:
    if not LIVE_SIGNALS_PATH.exists():
        raise FileNotFoundError(
            f"Live signal ledger not found: {LIVE_SIGNALS_PATH}. "
            "Run scripts/run_live_final_pipeline.py first."
        )

    signals = pd.read_csv(LIVE_SIGNALS_PATH)

    signals["signal_date"] = pd.to_datetime(
        signals["signal_date"],
        errors="coerce",
    ).dt.normalize()

    signals["ticker"] = signals["ticker"].astype(str).str.strip().str.upper()

    signals["final_weight"] = pd.to_numeric(
        signals["final_weight"],
        errors="coerce",
    ).fillna(0.0)

    if "model_name" not in signals.columns:
        signals["model_name"] = "LTSAF_live_v1"

    return signals


def get_next_price_date(
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> pd.Timestamp | None:
    signal_date = pd.Timestamp(signal_date).normalize()

    future_dates = [
        pd.Timestamp(d).normalize()
        for d in prices.index
        if pd.Timestamp(d).normalize() > signal_date
    ]

    if len(future_dates) == 0:
        return None

    return min(future_dates)


def evaluate_signal_period(
    signal_group: pd.DataFrame,
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    evaluation_date: pd.Timestamp,
    model_name: str,
) -> tuple[dict, pd.DataFrame]:
    signal_date = pd.Timestamp(signal_date).normalize()
    evaluation_date = pd.Timestamp(evaluation_date).normalize()

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).normalize()

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

        evaluation_price_date = pd.Timestamp(available.index.max()).normalize()
    else:
        evaluation_price_date = evaluation_date

    start_prices = prices.loc[signal_price_date]
    end_prices = prices.loc[evaluation_price_date]

    rows = []

    for _, row in signal_group.iterrows():
        ticker = row["ticker"]
        weight = float(row["final_weight"])

        if ticker == "CASH":
            start_price = np.nan
            end_price = np.nan
            holding_return = 0.0
        elif ticker not in start_prices.index or ticker not in end_prices.index:
            start_price = np.nan
            end_price = np.nan
            holding_return = np.nan
        else:
            start_price = float(start_prices[ticker])
            end_price = float(end_prices[ticker])

            if pd.isna(start_price) or pd.isna(end_price) or start_price <= 0:
                holding_return = np.nan
            else:
                holding_return = end_price / start_price - 1.0

        contribution = weight * holding_return if not pd.isna(holding_return) else np.nan

        rows.append(
            {
                "model_name": model_name,
                "signal_date": signal_date,
                "evaluation_date": evaluation_date,
                "ticker": ticker,
                "final_weight": weight,
                "start_price": start_price,
                "end_price": end_price,
                "holding_return": holding_return,
                "contribution": contribution,
            }
        )

    holding_detail = pd.DataFrame(rows)

    usable = holding_detail.dropna(subset=["contribution"]).copy()
    portfolio_return = float(usable["contribution"].sum())

    missing_tickers = holding_detail.loc[
        holding_detail["holding_return"].isna(),
        "ticker",
    ].tolist()

    missing_tickers = [t for t in missing_tickers if t != "CASH"]

    if "SPY" in start_prices.index and "SPY" in end_prices.index:
        spy_start = float(start_prices["SPY"])
        spy_end = float(end_prices["SPY"])

        if spy_start > 0:
            spy_return = spy_end / spy_start - 1.0
        else:
            spy_return = np.nan
    else:
        spy_start = np.nan
        spy_end = np.nan
        spy_return = np.nan

    excess_return = portfolio_return - spy_return if not pd.isna(spy_return) else np.nan

    stock_rows = holding_detail[
        (holding_detail["ticker"] != "CASH")
        & holding_detail["holding_return"].notna()
    ].copy()

    if len(stock_rows) > 0:
        best = stock_rows.sort_values("holding_return", ascending=False).iloc[0]
        worst = stock_rows.sort_values("holding_return", ascending=True).iloc[0]

        best_ticker = best["ticker"]
        best_holding_return = float(best["holding_return"])
        worst_ticker = worst["ticker"]
        worst_holding_return = float(worst["holding_return"])
    else:
        best_ticker = "NONE"
        best_holding_return = np.nan
        worst_ticker = "NONE"
        worst_holding_return = np.nan

    cash_weight = float(
        signal_group.loc[
            signal_group["ticker"] == "CASH",
            "final_weight",
        ].sum()
    )

    period_row = {
        "run_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_name": model_name,
        "signal_date": signal_date,
        "signal_price_date": signal_price_date,
        "evaluation_date": evaluation_date,
        "evaluation_price_date": evaluation_price_date,
        "portfolio_return": portfolio_return,
        "spy_return": spy_return,
        "excess_return": excess_return,
        "beat_spy": bool(portfolio_return > spy_return) if not pd.isna(spy_return) else False,
        "holding_count": int(len(signal_group[signal_group["ticker"] != "CASH"])),
        "cash_weight": cash_weight,
        "best_ticker": best_ticker,
        "best_holding_return": best_holding_return,
        "worst_ticker": worst_ticker,
        "worst_holding_return": worst_holding_return,
        "missing_tickers": ", ".join(missing_tickers),
    }

    return period_row, holding_detail


def add_cumulative_stats(
    performance: pd.DataFrame,
    starting_cash: float,
) -> pd.DataFrame:
    out = performance.copy()
    out = out.sort_values(["model_name", "signal_date"]).reset_index(drop=True)

    chunks = []

    for model_name, group in out.groupby("model_name"):
        group = group.sort_values("signal_date").copy()

        group["portfolio_cumulative_value"] = starting_cash * (
            1.0 + group["portfolio_return"]
        ).cumprod()

        group["spy_cumulative_value"] = starting_cash * (
            1.0 + group["spy_return"]
        ).cumprod()

        group["portfolio_cumulative_return"] = (
            group["portfolio_cumulative_value"] / starting_cash - 1.0
        )

        group["spy_cumulative_return"] = (
            group["spy_cumulative_value"] / starting_cash - 1.0
        )

        group["cumulative_excess_return"] = (
            group["portfolio_cumulative_return"] - group["spy_cumulative_return"]
        )

        running_max = group["portfolio_cumulative_value"].cummax()
        group["portfolio_drawdown"] = group["portfolio_cumulative_value"] / running_max - 1.0

        chunks.append(group)

    return pd.concat(chunks, ignore_index=True)


def update_performance_ledger(
    new_rows: list[dict],
    starting_cash: float,
) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)

    if LIVE_PERFORMANCE_LEDGER_PATH.exists():
        existing = pd.read_csv(LIVE_PERFORMANCE_LEDGER_PATH)

        existing["signal_date"] = pd.to_datetime(
            existing["signal_date"],
            errors="coerce",
        ).dt.normalize()

        new_df["signal_date"] = pd.to_datetime(
            new_df["signal_date"],
            errors="coerce",
        ).dt.normalize()

        existing_key = existing["model_name"].astype(str) + "|" + existing["signal_date"].astype(str)
        new_key = new_df["model_name"].astype(str) + "|" + new_df["signal_date"].astype(str)

        existing = existing[~existing_key.isin(set(new_key))].copy()
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    combined["signal_date"] = pd.to_datetime(
        combined["signal_date"],
        errors="coerce",
    ).dt.normalize()

    combined["evaluation_date"] = pd.to_datetime(
        combined["evaluation_date"],
        errors="coerce",
    ).dt.normalize()

    combined = combined.sort_values(["model_name", "signal_date"]).reset_index(drop=True)
    combined = add_cumulative_stats(combined, starting_cash=starting_cash)

    combined.to_csv(LIVE_PERFORMANCE_LEDGER_PATH, index=False)

    return combined


def write_summary(performance: pd.DataFrame) -> None:
    latest = performance.sort_values("evaluation_date").tail(1).iloc[0]

    lines = []
    lines.append("LTSAF Live Performance Summary")
    lines.append("=============================")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

    with open(LIVE_PERFORMANCE_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    config = load_config(CONFIG_PATH)
    ensure_output_dirs(config)

    LIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    signals = load_live_signals()
    prices = load_monthly_prices(config)
    prices.index = pd.to_datetime(prices.index).normalize()

    starting_cash = float(config["paper_trading"]["starting_cash"])

    new_rows = []

    for (model_name, signal_date), group in signals.groupby(["model_name", "signal_date"]):
        signal_date = pd.Timestamp(signal_date).normalize()
        evaluation_date = get_next_price_date(prices, signal_date)

        if evaluation_date is None:
            print(
                f"Skipping signal_date={signal_date.date()} because no later completed price date exists yet."
            )
            continue

        row, detail = evaluate_signal_period(
            signal_group=group,
            prices=prices,
            signal_date=signal_date,
            evaluation_date=evaluation_date,
            model_name=model_name,
        )

        detail_path = LIVE_OUTPUT_DIR / f"live_performance_detail_{signal_date.date()}.csv"
        detail.to_csv(detail_path, index=False)

        new_rows.append(row)

    if len(new_rows) == 0:
        print("")
        print("=" * 100)
        print("LTSAF LIVE PERFORMANCE TRACKER")
        print("=" * 100)
        print("No completed live holding periods available yet.")
        print("This is expected if the latest live signal is also the latest completed monthly price date.")
        return

    performance = update_performance_ledger(
        new_rows=new_rows,
        starting_cash=starting_cash,
    )

    write_summary(performance)

    latest = performance.sort_values("evaluation_date").tail(1).iloc[0]

    print("")
    print("=" * 100)
    print("LTSAF LIVE PERFORMANCE TRACKER")
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
    print("Saved performance ledger:", LIVE_PERFORMANCE_LEDGER_PATH)
    print("Saved summary:", LIVE_PERFORMANCE_SUMMARY_PATH)
    print("")
    print("FULL LIVE PERFORMANCE LEDGER")
    print(performance.to_string(index=False))


if __name__ == "__main__":
    main()