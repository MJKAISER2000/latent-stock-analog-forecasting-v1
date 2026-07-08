"""LTSAF v2 overlay backtest.

v2 keeps the v1 model's monthly out-of-sample signals exactly as the v1
walk-forward CV produced them (outputs/cv_backtest/cv_holdings.csv) and
changes only what happens *after* the model speaks: hedging, dip/extension
tilts, volatility targeting, and trade filtering. Because the v1 books were
generated strictly walk-forward and every overlay uses only information
available at each signal date, the v2 results are equally out-of-sample —
with zero retraining cost.

Variants evaluated:
    v1_baseline    — reproduce v1 net returns from its holdings (sanity check)
    dip_tilt       — buy dips / sell high tilt inside the sleeve
    trend_hedge    — SPY 10-month MA market hedge
    vol_target     — scale exposure to a target realized vol
    no_trade_band  — skip sub-band trades to cut costs
    v2_combined    — all enabled overlays together

Run (from the project root):
    .venv312\\Scripts\\python.exe v2\\run_v2_backtest.py

Requires a completed v1 CV backtest (backtests/run_cv_backtest.py).
"""

import os
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtests.cv_metrics import contiguous_folds, summarize_returns
from v2.overlays import (
    apply_no_trade_band,
    dip_tilt,
    spy_trend_is_on,
    trend_hedge,
    vol_target_exposure,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "v2_config.yaml")


def load_inputs(config: dict) -> tuple[dict, pd.Series, pd.Series, pd.DataFrame]:
    """Return (monthly stock books, returns lookup, dip feature lookup, prices)."""
    holdings = pd.read_csv(config["inputs"]["v1_holdings"], parse_dates=["signal_date"])
    stocks = holdings[holdings["ticker"] != "CASH"]

    books = {
        date: month.set_index("ticker")["final_weight"]
        for date, month in stocks.groupby("signal_date")
    }

    dataset = pd.read_parquet(
        config["inputs"]["dataset"],
        columns=["date", "ticker", "future_1m_return",
                 config["overlays"]["dip_tilt"]["feature"]],
    )
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset["ticker"] = dataset["ticker"].astype(str).str.strip().str.upper()
    indexed = dataset.set_index(["date", "ticker"])

    returns_lookup = indexed["future_1m_return"]
    dip_lookup = indexed[config["overlays"]["dip_tilt"]["feature"]]

    prices = pd.read_parquet(config["inputs"]["monthly_prices"])
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices.columns = [str(c).strip().upper() for c in prices.columns]

    return books, returns_lookup, dip_lookup, prices


def run_variant(
    name: str,
    flags: dict,
    books: dict,
    returns_lookup: pd.Series,
    dip_lookup: pd.Series,
    prices: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Sequential monthly simulation of one overlay combination."""
    overlay_cfg = config["overlays"]
    tc = float(config["transaction_cost"])

    rows = []
    past_net: list[float] = []
    prev_actual: pd.Series | None = None
    prev_returns: pd.Series | None = None
    prev_gross = 0.0

    for signal_date in sorted(books):
        target = books[signal_date].copy()

        if flags.get("dip"):
            cfg = overlay_cfg["dip_tilt"]
            ratios = dip_lookup.loc[signal_date] if signal_date in dip_lookup.index else pd.Series(dtype=float)
            target = dip_tilt(
                target, ratios,
                strength=float(cfg["strength"]),
                min_mult=float(cfg["min_mult"]),
                max_mult=float(cfg["max_mult"]),
            )

        trend_on = True
        if flags.get("trend"):
            cfg = overlay_cfg["trend_hedge"]
            trend_on = spy_trend_is_on(prices, signal_date, int(cfg["ma_months"]))
            target = trend_hedge(target, trend_on, float(cfg["exposure_when_off"]))

        exposure = 1.0
        if flags.get("vol"):
            cfg = overlay_cfg["vol_target"]
            exposure = vol_target_exposure(
                past_net,
                target_annual_vol=float(cfg["target_annual_vol"]),
                window_months=int(cfg["window_months"]),
                max_exposure=float(cfg["max_exposure"]),
            )
            target = target * exposure

        if prev_actual is None:
            actual = target
            turnover = float(target.abs().sum())
        else:
            drifted = prev_actual * (1.0 + prev_returns) / (1.0 + prev_gross)
            if flags.get("band"):
                actual = apply_no_trade_band(
                    target, drifted, float(overlay_cfg["no_trade_band"]["band"])
                )
            else:
                actual = target
            tickers = actual.index.union(drifted.index)
            turnover = float(
                (actual.reindex(tickers, fill_value=0.0)
                 - drifted.reindex(tickers, fill_value=0.0)).abs().sum()
            )

        month = returns_lookup.loc[signal_date] if signal_date in returns_lookup.index else pd.Series(dtype=float)
        realized = month.reindex(actual.index).fillna(0.0)

        gross = float((actual * realized).sum())
        cost = tc * turnover
        net = gross - cost
        past_net.append(net)

        rows.append({
            "signal_date": signal_date,
            "variant": name,
            "gross_return": gross,
            "net_return": net,
            "turnover": turnover,
            "cost": cost,
            "stock_exposure": float(actual.sum()),
            "n_positions": int(len(actual)),
            "trend_on": trend_on,
            "vol_exposure": exposure,
        })

        prev_actual = actual
        prev_returns = realized
        prev_gross = gross

    return pd.DataFrame(rows)


def fold_sharpes(net: pd.Series, n_folds: int) -> tuple[float, float]:
    sharpes = []
    for _, fold_dates in contiguous_folds(list(net.index), n_folds):
        sharpes.append(summarize_returns(net.loc[fold_dates])["sharpe_ratio"])
    sharpes = pd.Series(sharpes)
    return float(sharpes.mean()), float(sharpes.std(ddof=1))


def main() -> None:
    os.chdir(PROJECT_ROOT)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not os.path.exists(config["inputs"]["v1_holdings"]):
        raise FileNotFoundError(
            "v1 CV holdings not found — run backtests/run_cv_backtest.py first."
        )

    os.makedirs(config["outputs_dir"], exist_ok=True)

    books, returns_lookup, dip_lookup, prices = load_inputs(config)

    v1_results = pd.read_csv(
        "outputs/cv_backtest/cv_monthly_results.csv", parse_dates=["signal_date"]
    )
    spy = pd.Series(
        v1_results["spy_return"].values, index=pd.DatetimeIndex(v1_results["signal_date"])
    ).sort_index()

    enabled = {k: v.get("enabled", False) for k, v in config["overlays"].items()}
    variants = {
        "v1_baseline": {},
        "dip_tilt": {"dip": True},
        "trend_hedge": {"trend": True},
        "vol_target": {"vol": True},
        "no_trade_band": {"band": True},
        "v2_combined": {
            "dip": enabled["dip_tilt"],
            "trend": enabled["trend_hedge"],
            "vol": enabled["vol_target"],
            "band": enabled["no_trade_band"],
        },
    }

    n_folds = int(config["folds"])
    all_monthly = []
    summary_rows = []

    for name, flags in variants.items():
        monthly = run_variant(
            name, flags, books, returns_lookup, dip_lookup, prices, config
        )
        all_monthly.append(monthly)

        net = pd.Series(
            monthly["net_return"].values, index=pd.DatetimeIndex(monthly["signal_date"])
        )
        row = summarize_returns(net, spy, label=name)
        row["avg_turnover"] = float(monthly["turnover"].mean())
        row["avg_cost_annualized"] = float(monthly["cost"].mean() * 12)
        row["fold_sharpe_mean"], row["fold_sharpe_std"] = fold_sharpes(net, n_folds)
        summary_rows.append(row)
        print(f"{name}: ann={row['annualized_return']:.2%} sharpe={row['sharpe_ratio']:.2f} "
              f"maxDD={row['max_drawdown']:.2%} turnover={row['avg_turnover']:.2f}", flush=True)

    spy_row = summarize_returns(spy, label="spy_buy_and_hold")
    spy_row["fold_sharpe_mean"], spy_row["fold_sharpe_std"] = fold_sharpes(spy, n_folds)
    summary_rows.append(spy_row)

    summary = pd.DataFrame(summary_rows)
    monthly_all = pd.concat(all_monthly, ignore_index=True)

    summary.to_csv(os.path.join(config["outputs_dir"], "v2_variant_comparison.csv"), index=False)
    monthly_all.to_csv(os.path.join(config["outputs_dir"], "v2_monthly_returns.csv"), index=False)

    cols = ["label", "annualized_return", "annualized_volatility", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "total_return",
            "information_ratio_vs_benchmark", "avg_turnover", "avg_cost_annualized",
            "fold_sharpe_mean", "fold_sharpe_std"]

    report_path = os.path.join(config["outputs_dir"], "v2_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("LTSAF v2 Overlay Backtest — variant comparison\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Months: {len(spy)} ({spy.index.min().date()} to {spy.index.max().date()}), "
                f"net of {config['transaction_cost']:.4f} costs, rf=0\n")
        f.write("Signals identical to v1 walk-forward CV; overlays applied per month "
                "using only same-date information.\n\n")
        f.write(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
        f.write("\n")

    with open(report_path, encoding="utf-8") as f:
        print("\n" + f.read())
    print("Outputs in:", os.path.abspath(config["outputs_dir"]))


if __name__ == "__main__":
    main()
