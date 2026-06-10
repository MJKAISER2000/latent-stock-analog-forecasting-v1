import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:
    import yfinance as yf
except ImportError as exc:
    raise ImportError(
        "yfinance is required for live portfolio valuation. "
        "Install it with: pip install yfinance"
    ) from exc


HOLDINGS_PATH = PROJECT_ROOT / "outputs" / "paper_trading" / "current_paper_holdings.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_trading"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

LIVE_SNAPSHOT_PATH = OUTPUT_DIR / "live_portfolio_value_snapshots.csv"
LATEST_SUMMARY_PATH = OUTPUT_DIR / "latest_live_portfolio_value_summary.txt"
LATEST_DETAIL_PATH = OUTPUT_DIR / "latest_live_portfolio_value_detail.csv"
LIVE_PLOT_PATH = FIGURES_DIR / "live_portfolio_vs_spy.png"


def load_holdings() -> pd.DataFrame:
    if not HOLDINGS_PATH.exists():
        raise FileNotFoundError(f"Holdings file not found: {HOLDINGS_PATH}")

    holdings = pd.read_csv(HOLDINGS_PATH)
    holdings["ticker"] = holdings["ticker"].astype(str).str.strip().str.upper()
    holdings["shares"] = pd.to_numeric(holdings["shares"], errors="coerce").fillna(0.0)

    return holdings


def get_quote_for_ticker(ticker: str) -> dict:
    if ticker == "CASH":
        return {
            "ticker": ticker,
            "current_price": 1.0,
            "previous_close": 1.0,
            "quote_status": "cash",
        }

    try:
        asset = yf.Ticker(ticker)
        fast = asset.fast_info

        current_price = None
        previous_close = None

        try:
            current_price = fast.get("last_price", None)
        except Exception:
            current_price = None

        try:
            previous_close = fast.get("previous_close", None)
        except Exception:
            previous_close = None

        if current_price is None or pd.isna(current_price) or previous_close is None or pd.isna(previous_close):
            hist = asset.history(period="5d", interval="1d", auto_adjust=False)

            if len(hist) >= 2:
                previous_close = float(hist["Close"].iloc[-2])
                current_price = float(hist["Close"].iloc[-1])
            elif len(hist) == 1:
                current_price = float(hist["Close"].iloc[-1])
                previous_close = current_price

        if current_price is None or previous_close is None:
            return {
                "ticker": ticker,
                "current_price": np.nan,
                "previous_close": np.nan,
                "quote_status": "missing_quote",
            }

        return {
            "ticker": ticker,
            "current_price": float(current_price),
            "previous_close": float(previous_close),
            "quote_status": "ok",
        }

    except Exception as exc:
        return {
            "ticker": ticker,
            "current_price": np.nan,
            "previous_close": np.nan,
            "quote_status": f"error: {exc}",
        }


def get_live_quotes(tickers: list[str]) -> pd.DataFrame:
    rows = []

    for ticker in tickers:
        print(f"Fetching quote: {ticker}")
        rows.append(get_quote_for_ticker(ticker))

    quotes = pd.DataFrame(rows)
    return quotes


def compute_live_portfolio_value(holdings: pd.DataFrame, quotes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    detail = holdings.merge(quotes, on="ticker", how="left")

    detail["previous_close"] = pd.to_numeric(detail["previous_close"], errors="coerce")
    detail["current_price"] = pd.to_numeric(detail["current_price"], errors="coerce")

    detail["previous_close_value"] = detail["shares"] * detail["previous_close"]
    detail["current_value"] = detail["shares"] * detail["current_price"]
    detail["day_pnl"] = detail["current_value"] - detail["previous_close_value"]

    detail["day_return"] = np.where(
        detail["previous_close_value"] > 0,
        detail["day_pnl"] / detail["previous_close_value"],
        0.0,
    )

    previous_close_value = float(detail["previous_close_value"].sum(skipna=True))
    current_value = float(detail["current_value"].sum(skipna=True))
    day_pnl = current_value - previous_close_value
    day_return = day_pnl / previous_close_value if previous_close_value > 0 else 0.0

    cash_value = float(detail.loc[detail["ticker"] == "CASH", "current_value"].sum(skipna=True))
    cash_weight = cash_value / current_value if current_value > 0 else 0.0

    missing_quotes = detail.loc[
        detail["quote_status"].astype(str) != "ok",
        "ticker",
    ].tolist()

    missing_quotes = [ticker for ticker in missing_quotes if ticker != "CASH"]

    stock_detail = detail[detail["ticker"] != "CASH"].copy()

    if len(stock_detail.dropna(subset=["day_pnl"])) > 0:
        best = stock_detail.sort_values("day_pnl", ascending=False).iloc[0]
        worst = stock_detail.sort_values("day_pnl", ascending=True).iloc[0]

        best_ticker = str(best["ticker"])
        best_day_pnl = float(best["day_pnl"])
        worst_ticker = str(worst["ticker"])
        worst_day_pnl = float(worst["day_pnl"])
    else:
        best_ticker = "NONE"
        best_day_pnl = 0.0
        worst_ticker = "NONE"
        worst_day_pnl = 0.0

    snapshot = {
        "run_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "valuation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "previous_close_value": previous_close_value,
        "current_value": current_value,
        "day_pnl": day_pnl,
        "day_return": day_return,
        "cash_value": cash_value,
        "cash_weight": cash_weight,
        "stock_positions": int(len(stock_detail)),
        "best_ticker_by_pnl": best_ticker,
        "best_day_pnl": best_day_pnl,
        "worst_ticker_by_pnl": worst_ticker,
        "worst_day_pnl": worst_day_pnl,
        "missing_quotes": ", ".join(missing_quotes),
    }

    return detail, snapshot


def get_spy_live_snapshot(previous_close_value: float) -> dict:
    quote = get_quote_for_ticker("SPY")

    spy_current = quote["current_price"]
    spy_previous = quote["previous_close"]

    if pd.isna(spy_current) or pd.isna(spy_previous) or spy_previous <= 0:
        return {
            "spy_previous_close": np.nan,
            "spy_current_price": np.nan,
            "spy_day_return": np.nan,
            "spy_equivalent_previous_value": previous_close_value,
            "spy_equivalent_current_value": np.nan,
            "excess_day_return_vs_spy": np.nan,
            "excess_day_pnl_vs_spy": np.nan,
        }

    spy_day_return = spy_current / spy_previous - 1.0
    spy_equivalent_current_value = previous_close_value * (1.0 + spy_day_return)

    return {
        "spy_previous_close": spy_previous,
        "spy_current_price": spy_current,
        "spy_day_return": spy_day_return,
        "spy_equivalent_previous_value": previous_close_value,
        "spy_equivalent_current_value": spy_equivalent_current_value,
    }


def append_live_snapshot(snapshot: dict) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([snapshot])

    if LIVE_SNAPSHOT_PATH.exists():
        existing = pd.read_csv(LIVE_SNAPSHOT_PATH)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row.copy()

    combined.to_csv(LIVE_SNAPSHOT_PATH, index=False)

    return combined


def make_live_plot(snapshots: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if len(snapshots) == 0:
        return

    plot_df = snapshots.copy()
    plot_df["valuation_time"] = pd.to_datetime(plot_df["valuation_time"], errors="coerce")

    plt.figure(figsize=(10, 6))
    plt.plot(
        plot_df["valuation_time"],
        plot_df["current_value"],
        marker="o",
        label="Latent Twin Portfolio",
    )

    if "spy_equivalent_current_value" in plot_df.columns:
        plt.plot(
            plot_df["valuation_time"],
            plot_df["spy_equivalent_current_value"],
            marker="o",
            label="SPY Same Starting Value",
        )

    plt.title("Live Paper Portfolio Value vs SPY")
    plt.xlabel("Time")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LIVE_PLOT_PATH, dpi=200)
    plt.close()


def save_summary(detail: pd.DataFrame, snapshot: dict) -> None:
    lines = []
    lines.append("Latent Market Twin Live Portfolio Value")
    lines.append("======================================")
    lines.append("")
    lines.append(f"Valuation time: {snapshot['valuation_time']}")
    lines.append("")
    lines.append(f"Yesterday close value: ${snapshot['previous_close_value']:,.2f}")
    lines.append(f"Current value: ${snapshot['current_value']:,.2f}")
    lines.append(f"Day P&L: ${snapshot['day_pnl']:,.2f}")
    lines.append(f"Day return: {snapshot['day_return']:.2%}")
    lines.append("")
    lines.append(f"SPY day return: {snapshot['spy_day_return']:.2%}")
    lines.append(f"SPY-equivalent current value: ${snapshot['spy_equivalent_current_value']:,.2f}")
    lines.append(f"Excess day return vs SPY: {snapshot['excess_day_return_vs_spy']:.2%}")
    lines.append(f"Excess day P&L vs SPY: ${snapshot['excess_day_pnl_vs_spy']:,.2f}")
    lines.append("")
    lines.append(f"Cash value: ${snapshot['cash_value']:,.2f}")
    lines.append(f"Cash weight: {snapshot['cash_weight']:.2%}")
    lines.append(f"Stock positions: {snapshot['stock_positions']}")
    lines.append("")
    lines.append(f"Best ticker by P&L: {snapshot['best_ticker_by_pnl']} (${snapshot['best_day_pnl']:,.2f})")
    lines.append(f"Worst ticker by P&L: {snapshot['worst_ticker_by_pnl']} (${snapshot['worst_day_pnl']:,.2f})")
    lines.append("")
    lines.append(f"Missing quotes: {snapshot['missing_quotes']}")
    lines.append("")
    lines.append("Holding detail:")
    lines.append(detail.to_string(index=False))

    with open(LATEST_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    holdings = load_holdings()

    tickers = sorted(holdings["ticker"].unique().tolist())
    quotes = get_live_quotes(tickers)

    detail, snapshot = compute_live_portfolio_value(
        holdings=holdings,
        quotes=quotes,
    )

    spy_snapshot = get_spy_live_snapshot(
        previous_close_value=snapshot["previous_close_value"],
    )

    snapshot.update(spy_snapshot)

    snapshot["excess_day_return_vs_spy"] = (
        snapshot["day_return"] - snapshot["spy_day_return"]
        if not pd.isna(snapshot["spy_day_return"])
        else np.nan
    )

    snapshot["excess_day_pnl_vs_spy"] = (
        snapshot["current_value"] - snapshot["spy_equivalent_current_value"]
        if not pd.isna(snapshot["spy_equivalent_current_value"])
        else np.nan
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    detail.to_csv(LATEST_DETAIL_PATH, index=False)

    snapshots = append_live_snapshot(snapshot)
    make_live_plot(snapshots)
    save_summary(detail, snapshot)

    print("")
    print("=" * 100)
    print("LIVE PAPER PORTFOLIO VALUE")
    print("=" * 100)
    print(f"Valuation time: {snapshot['valuation_time']}")
    print("")
    print(f"Yesterday close value: ${snapshot['previous_close_value']:,.2f}")
    print(f"Current value: ${snapshot['current_value']:,.2f}")
    print(f"Day P&L: ${snapshot['day_pnl']:,.2f}")
    print(f"Day return: {snapshot['day_return']:.2%}")
    print("")
    print(f"SPY day return: {snapshot['spy_day_return']:.2%}")
    print(f"SPY-equivalent current value: ${snapshot['spy_equivalent_current_value']:,.2f}")
    print(f"Excess day return vs SPY: {snapshot['excess_day_return_vs_spy']:.2%}")
    print(f"Excess day P&L vs SPY: ${snapshot['excess_day_pnl_vs_spy']:,.2f}")
    print("")
    print(f"Cash value: ${snapshot['cash_value']:,.2f}")
    print(f"Cash weight: {snapshot['cash_weight']:.2%}")
    print(f"Stock positions: {snapshot['stock_positions']}")
    print("")
    print(f"Best ticker by P&L: {snapshot['best_ticker_by_pnl']} (${snapshot['best_day_pnl']:,.2f})")
    print(f"Worst ticker by P&L: {snapshot['worst_ticker_by_pnl']} (${snapshot['worst_day_pnl']:,.2f})")
    print("")
    print(f"Missing quotes: {snapshot['missing_quotes']}")
    print("")
    print("Saved detail:", LATEST_DETAIL_PATH)
    print("Saved live snapshots:", LIVE_SNAPSHOT_PATH)
    print("Saved summary:", LATEST_SUMMARY_PATH)
    print("Saved plot:", LIVE_PLOT_PATH)
    print("")
    print("LIVE HOLDING DETAIL")
    print(detail.to_string(index=False))


if __name__ == "__main__":
    main()