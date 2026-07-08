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
        "yfinance is required for live valuation. Install it with: pip install yfinance"
    ) from exc


LIVE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_trading_live"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

LIVE_HOLDINGS_PATH = LIVE_OUTPUT_DIR / "current_live_holdings.csv"

LIVE_VALUE_SNAPSHOTS_PATH = LIVE_OUTPUT_DIR / "live_value_snapshots.csv"
LATEST_LIVE_VALUE_DETAIL_PATH = LIVE_OUTPUT_DIR / "latest_live_value_detail.csv"
LATEST_LIVE_VALUE_SUMMARY_PATH = LIVE_OUTPUT_DIR / "latest_live_value_summary.txt"
LIVE_VALUE_VS_SPY_PLOT_PATH = FIGURES_DIR / "ltsaf_live_value_vs_spy.png"

# Anchor for the SPY buy-and-hold benchmark (portfolio starting value + SPY price on the
# paper-trading inception date). Persisted once so the SPY gap is measured from inception
# and stays correct even when daily logging is skipped for a stretch of days.
SPY_BENCHMARK_ANCHOR_PATH = LIVE_OUTPUT_DIR / "spy_benchmark_anchor.csv"
CONFIG_PATH = "configs/live_model_config.yaml"
DEFAULT_STARTING_CASH = 100000.0


def load_live_holdings() -> pd.DataFrame:
    if not LIVE_HOLDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Live holdings file not found: {LIVE_HOLDINGS_PATH}. "
            "Run scripts/initialize_live_holdings.py first."
        )

    holdings = pd.read_csv(LIVE_HOLDINGS_PATH)
    holdings["ticker"] = holdings["ticker"].astype(str).str.strip().str.upper()
    holdings["shares"] = pd.to_numeric(holdings["shares"], errors="coerce").fillna(0.0)

    return holdings


def get_quote(ticker: str) -> dict:
    if ticker == "CASH":
        return {
            "ticker": "CASH",
            "current_price": 1.0,
            "previous_close": 1.0,
            "quote_status": "cash",
        }

    try:
        asset = yf.Ticker(ticker)

        current_price = np.nan
        previous_close = np.nan

        try:
            fast = asset.fast_info

            if "last_price" in fast:
                current_price = fast["last_price"]

            if "previous_close" in fast:
                previous_close = fast["previous_close"]

        except Exception:
            pass

        if pd.isna(current_price) or pd.isna(previous_close):
            hist = asset.history(period="5d", interval="1d", auto_adjust=False)

            if len(hist) >= 2:
                previous_close = float(hist["Close"].iloc[-2])
                current_price = float(hist["Close"].iloc[-1])
            elif len(hist) == 1:
                current_price = float(hist["Close"].iloc[-1])
                previous_close = current_price

        if pd.isna(current_price) or pd.isna(previous_close):
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


def get_quotes(tickers: list[str]) -> pd.DataFrame:
    rows = []

    for ticker in tickers:
        print(f"Fetching quote: {ticker}")
        rows.append(get_quote(ticker))

    return pd.DataFrame(rows)


def compute_live_value(holdings: pd.DataFrame, quotes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    detail = holdings.merge(quotes, on="ticker", how="left")

    detail["current_price"] = pd.to_numeric(detail["current_price"], errors="coerce")
    detail["previous_close"] = pd.to_numeric(detail["previous_close"], errors="coerce")

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

    missing_quotes = detail.loc[
        ~detail["quote_status"].isin(["ok", "cash"]),
        "ticker",
    ].tolist()

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


def get_spy_benchmark_anchor(spy_current_price: float, holdings: pd.DataFrame) -> tuple[float, float]:
    """Return (inception_value, spy_inception_price) for the SPY buy-and-hold benchmark.

    The anchor is the portfolio's starting value and SPY's closing price on the paper-trading
    inception date. It is computed once and persisted, so the SPY benchmark is a true
    buy-and-hold from inception rather than something re-anchored to the portfolio every day.
    This makes the SPY gap correct even if daily logging is skipped for a while, because the
    benchmark only depends on the inception price and the current price -- not on a continuous
    day-by-day chain of snapshots.
    """
    if SPY_BENCHMARK_ANCHOR_PATH.exists():
        anchor = pd.read_csv(SPY_BENCHMARK_ANCHOR_PATH)
        inception_value = float(anchor["inception_value"].iloc[0])
        spy_inception_price = float(anchor["spy_inception_price"].iloc[0])
        return inception_value, spy_inception_price

    # Inception value = configured starting cash (fall back to the known default).
    inception_value = DEFAULT_STARTING_CASH
    try:
        from src.utils.config import load_config

        config = load_config(CONFIG_PATH)
        inception_value = float(config["paper_trading"]["starting_cash"])
    except Exception:
        pass

    # Inception date = earliest signal date the live holdings were established on.
    inception_date = None
    try:
        if "last_signal_date" in holdings.columns:
            dates = pd.to_datetime(holdings["last_signal_date"], errors="coerce").dropna()
            if len(dates) > 0:
                inception_date = dates.min().normalize()
    except Exception:
        pass

    # SPY closing price on (or just before) the inception date.
    spy_inception_price = np.nan
    if inception_date is not None:
        try:
            hist = yf.Ticker("SPY").history(
                start=(inception_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                end=(inception_date + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
            )
            if len(hist) > 0:
                hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
                on_or_before = hist[hist.index <= inception_date]
                if len(on_or_before) > 0:
                    spy_inception_price = float(on_or_before["Close"].iloc[-1])
        except Exception:
            pass

    # Fallback: anchor to the current SPY price (benchmark simply starts flat from now).
    if pd.isna(spy_inception_price) or spy_inception_price <= 0:
        spy_inception_price = (
            float(spy_current_price) if not pd.isna(spy_current_price) else np.nan
        )

    if not pd.isna(spy_inception_price) and spy_inception_price > 0:
        LIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "inception_date": inception_date.date() if inception_date is not None else "",
                    "inception_value": inception_value,
                    "spy_inception_price": spy_inception_price,
                }
            ]
        ).to_csv(SPY_BENCHMARK_ANCHOR_PATH, index=False)

    return inception_value, spy_inception_price


def add_spy_comparison(
    snapshot: dict,
    spy_current: float,
    spy_previous: float,
    inception_value: float,
    spy_inception_price: float,
) -> dict:
    snapshot["spy_previous_close"] = spy_previous
    snapshot["spy_current_price"] = spy_current

    # --- Single-day comparison (always based on SPY's real prior trading-day close). ---
    if pd.isna(spy_current) or pd.isna(spy_previous) or spy_previous <= 0:
        snapshot["spy_day_return"] = np.nan
        snapshot["excess_day_return_vs_spy"] = np.nan
        snapshot["excess_day_pnl_vs_spy"] = np.nan
    else:
        spy_day_return = spy_current / spy_previous - 1.0
        snapshot["spy_day_return"] = spy_day_return
        snapshot["excess_day_return_vs_spy"] = snapshot["day_return"] - spy_day_return
        snapshot["excess_day_pnl_vs_spy"] = (
            snapshot["day_pnl"] - snapshot["previous_close_value"] * spy_day_return
        )

    # --- Cumulative comparison anchored to inception (this is the real SPY gap). ---
    if pd.isna(spy_current) or pd.isna(spy_inception_price) or spy_inception_price <= 0:
        snapshot["spy_equivalent_current_value"] = np.nan
        snapshot["spy_cumulative_return"] = np.nan
        snapshot["excess_cumulative_return_vs_spy"] = np.nan
        snapshot["excess_cumulative_pnl_vs_spy"] = np.nan
    else:
        spy_equivalent_current_value = inception_value * (spy_current / spy_inception_price)
        portfolio_cumulative_return = snapshot["current_value"] / inception_value - 1.0
        spy_cumulative_return = spy_current / spy_inception_price - 1.0

        snapshot["spy_equivalent_current_value"] = spy_equivalent_current_value
        snapshot["spy_cumulative_return"] = spy_cumulative_return
        snapshot["excess_cumulative_return_vs_spy"] = (
            portfolio_cumulative_return - spy_cumulative_return
        )
        snapshot["excess_cumulative_pnl_vs_spy"] = (
            snapshot["current_value"] - spy_equivalent_current_value
        )

    return snapshot


def recompute_spy_benchmark(
    snapshots: pd.DataFrame,
    inception_value: float,
    spy_inception_price: float,
) -> pd.DataFrame:
    """Recompute the cumulative SPY benchmark for every snapshot row from the inception anchor.

    This corrects historical rows that were logged with the old method (which re-anchored the
    SPY benchmark to the portfolio's own previous close each day) and guarantees the whole
    series is consistent, even across stretches where daily logging was skipped.
    """
    out = snapshots.copy()

    if (
        pd.isna(spy_inception_price)
        or spy_inception_price <= 0
        or "spy_current_price" not in out.columns
    ):
        return out

    spy_price = pd.to_numeric(out["spy_current_price"], errors="coerce")
    current_value = pd.to_numeric(out["current_value"], errors="coerce")

    out["spy_equivalent_current_value"] = inception_value * (spy_price / spy_inception_price)
    out["spy_cumulative_return"] = spy_price / spy_inception_price - 1.0
    out["excess_cumulative_return_vs_spy"] = (
        current_value / inception_value - 1.0
    ) - out["spy_cumulative_return"]
    out["excess_cumulative_pnl_vs_spy"] = current_value - out["spy_equivalent_current_value"]

    return out


def append_snapshot(snapshot: dict) -> pd.DataFrame:
    LIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([snapshot])

    if LIVE_VALUE_SNAPSHOTS_PATH.exists():
        existing = pd.read_csv(LIVE_VALUE_SNAPSHOTS_PATH)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row.copy()

    combined.to_csv(LIVE_VALUE_SNAPSHOTS_PATH, index=False)

    return combined


def make_plot(snapshots: pd.DataFrame) -> None:
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
        label="LTSAF live portfolio",
    )

    if "spy_equivalent_current_value" in plot_df.columns:
        plt.plot(
            plot_df["valuation_time"],
            plot_df["spy_equivalent_current_value"],
            marker="o",
            label="SPY same starting value",
        )

    plt.title("LTSAF Live Portfolio Value vs SPY")
    plt.xlabel("Time")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LIVE_VALUE_VS_SPY_PLOT_PATH, dpi=200)
    plt.close()


def save_summary(detail: pd.DataFrame, snapshot: dict) -> None:
    lines = []

    lines.append("LTSAF Live Portfolio Value Summary")
    lines.append("=================================")
    lines.append("")
    lines.append(f"Valuation time: {snapshot['valuation_time']}")
    lines.append("")
    lines.append(f"Yesterday close value: ${snapshot['previous_close_value']:,.2f}")
    lines.append(f"Current value: ${snapshot['current_value']:,.2f}")
    lines.append(f"Day P&L: ${snapshot['day_pnl']:,.2f}")
    lines.append(f"Day return: {snapshot['day_return']:.2%}")
    lines.append("")
    lines.append(f"SPY day return: {snapshot['spy_day_return']:.2%}")
    lines.append(f"Excess day return vs SPY: {snapshot['excess_day_return_vs_spy']:.2%}")
    lines.append(f"Excess day P&L vs SPY: ${snapshot['excess_day_pnl_vs_spy']:,.2f}")
    lines.append("")
    lines.append(f"SPY-equivalent value (buy & hold from inception): ${snapshot['spy_equivalent_current_value']:,.2f}")
    lines.append(f"SPY cumulative return since inception: {snapshot['spy_cumulative_return']:.2%}")
    lines.append(f"Cumulative excess return vs SPY: {snapshot['excess_cumulative_return_vs_spy']:.2%}")
    lines.append(f"Cumulative gap vs SPY: ${snapshot['excess_cumulative_pnl_vs_spy']:,.2f}")
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

    with open(LATEST_LIVE_VALUE_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    LIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    holdings = load_live_holdings()

    tickers = sorted(holdings["ticker"].unique().tolist())
    quotes = get_quotes(tickers)

    detail, snapshot = compute_live_value(holdings, quotes)

    # Guard against no-internet runs: if every stock quote failed, current_value would
    # collapse to just the cash position. Skip writing so the history is not polluted
    # with a misleading cash-only snapshot (these used to show up as a crash to ~$0).
    stock_detail = detail[detail["ticker"] != "CASH"]
    usable_quotes = int(pd.to_numeric(stock_detail["current_price"], errors="coerce").notna().sum())

    if len(stock_detail) > 0 and usable_quotes == 0:
        print("")
        print("=" * 100)
        print("LTSAF LIVE PORTFOLIO VALUE")
        print("=" * 100)
        print("No usable stock quotes were returned (likely no internet connection).")
        print("Skipping snapshot write so the value history is not corrupted with a cash-only row.")
        return

    spy_quote = get_quote("SPY")
    spy_current = pd.to_numeric(spy_quote["current_price"], errors="coerce")
    spy_previous = pd.to_numeric(spy_quote["previous_close"], errors="coerce")

    inception_value, spy_inception_price = get_spy_benchmark_anchor(spy_current, holdings)

    snapshot = add_spy_comparison(
        snapshot,
        spy_current=spy_current,
        spy_previous=spy_previous,
        inception_value=inception_value,
        spy_inception_price=spy_inception_price,
    )

    detail.to_csv(LATEST_LIVE_VALUE_DETAIL_PATH, index=False)

    snapshots = append_snapshot(snapshot)

    # Re-derive the cumulative SPY benchmark for the whole history from the inception anchor,
    # so the SPY gap is correct even where past days were not logged.
    snapshots = recompute_spy_benchmark(snapshots, inception_value, spy_inception_price)
    snapshots.to_csv(LIVE_VALUE_SNAPSHOTS_PATH, index=False)

    make_plot(snapshots)
    save_summary(detail, snapshot)

    print("")
    print("=" * 100)
    print("LTSAF LIVE PORTFOLIO VALUE")
    print("=" * 100)
    print(f"Valuation time: {snapshot['valuation_time']}")
    print("")
    print(f"Yesterday close value: ${snapshot['previous_close_value']:,.2f}")
    print(f"Current value: ${snapshot['current_value']:,.2f}")
    print(f"Day P&L: ${snapshot['day_pnl']:,.2f}")
    print(f"Day return: {snapshot['day_return']:.2%}")
    print("")
    print(f"SPY day return: {snapshot['spy_day_return']:.2%}")
    print(f"Excess day return vs SPY: {snapshot['excess_day_return_vs_spy']:.2%}")
    print(f"Excess day P&L vs SPY: ${snapshot['excess_day_pnl_vs_spy']:,.2f}")
    print("")
    print(f"SPY-equivalent value (buy & hold from inception): ${snapshot['spy_equivalent_current_value']:,.2f}")
    print(f"SPY cumulative return since inception: {snapshot['spy_cumulative_return']:.2%}")
    print(f"Cumulative excess return vs SPY: {snapshot['excess_cumulative_return_vs_spy']:.2%}")
    print(f"Cumulative gap vs SPY: ${snapshot['excess_cumulative_pnl_vs_spy']:,.2f}")
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
    print("Saved detail:", LATEST_LIVE_VALUE_DETAIL_PATH)
    print("Saved snapshots:", LIVE_VALUE_SNAPSHOTS_PATH)
    print("Saved summary:", LATEST_LIVE_VALUE_SUMMARY_PATH)
    print("Saved plot:", LIVE_VALUE_VS_SPY_PLOT_PATH)
    print("")
    print("LIVE HOLDING DETAIL")
    print(detail.to_string(index=False))


if __name__ == "__main__":
    main()