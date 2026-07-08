"""Walk-forward cross-validation backtest for the LTSAF live strategy.

For every signal month t in the test window this script re-runs the exact live
pipeline as it would have run at t:

1. Take only data with date <= t (no future rows exist at training time).
2. Impute features using medians computed on that visible slice only.
3. Train the two LightGBM LambdaRank branches on dates < t with
   src.models.rankers.train_predict_latest (same early-stopping / 80-20 date
   split the live system uses) and score month t.
4. Build the portfolio with src.paper_trading.portfolio.build_final_portfolio
   (70% original top-20 + 30% latent-neighbor top-10, inverse-vol weighting,
   tech-drawdown regime filter) — the same code the live system calls.
5. Realize the next-month return from future_1m_return and log holdings.

Because every month is scored strictly out-of-sample and the model is retrained
each step, this is expanding-window time-series cross-validation. The test
months are then also split into K contiguous folds so you get per-fold Sharpe /
annualized return and a dispersion estimate, plus per-calendar-year stats.

Transaction costs: config portfolio.transaction_cost (default 0.001 = 10 bps)
charged per dollar traded, on two-way turnover vs. the drifted prior-month
portfolio. Cash earns 0%. Sharpe uses rf = 0.

Results are checkpointed per month to outputs/cv_backtest/, so an interrupted
run resumes where it left off. Delete the output dir (or pass --fresh) to
recompute from scratch.

Usage (from the project root):
    .venv312\\Scripts\\python.exe backtests\\run_cv_backtest.py
    .venv312\\Scripts\\python.exe backtests\\run_cv_backtest.py --limit-months 3   # smoke test
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_config
from src.data.loaders import load_neighbor_dataset, load_monthly_prices, load_universe
from src.features.feature_sets import get_feature_columns, validate_no_leakage
from src.models.rankers import train_predict_latest
from src.paper_trading.portfolio import build_final_portfolio

from backtests.cv_metrics import contiguous_folds, summarize_returns

DEFAULT_CONFIG = "configs/live_model_config.yaml"
DEFAULT_OUTPUT_DIR = "outputs/cv_backtest"

MONTHLY_RESULTS_FILE = "cv_monthly_results.csv"
HOLDINGS_FILE = "cv_holdings.csv"
OVERALL_SUMMARY_FILE = "cv_overall_summary.csv"
FOLD_SUMMARY_FILE = "cv_fold_summary.csv"
YEARLY_SUMMARY_FILE = "cv_yearly_summary.csv"
REPORT_FILE = "cv_backtest_report.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward CV backtest for LTSAF_live_v1")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-train-months", type=int, default=36,
                        help="Months of history required before the first test month")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of contiguous CV folds for the fold summary")
    parser.add_argument("--limit-months", type=int, default=0,
                        help="If > 0, only run this many test months (smoke test)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing checkpoints and recompute everything")
    return parser.parse_args()


def point_in_time_impute(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Clean the feature matrix using only information in the given slice.

    Same cleaning as src.features.feature_sets.prepare_feature_matrix, but the
    caller is responsible for passing only rows with date <= signal date, so
    the imputation medians never see the future.
    """
    working = df.copy()
    X = working[feature_cols].copy()

    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], pd.NA)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0.0)

    for col in feature_cols:
        working[col] = X[col]

    return working


def get_test_dates(dataset: pd.DataFrame, min_train_months: int) -> list[pd.Timestamp]:
    """All month-ends that have realized next-month returns and enough history."""
    label_counts = dataset.groupby("date")["future_1m_return"].count()
    all_dates = sorted(pd.to_datetime(d) for d in dataset["date"].unique())
    scoreable = [d for d in all_dates if label_counts.get(d, 0) > 0]
    return [d for d in scoreable if all_dates.index(d) >= min_train_months]


def spy_return_for_date(dataset: pd.DataFrame, signal_date: pd.Timestamp) -> float:
    rows = dataset.loc[dataset["date"] == signal_date, "future_1m_spy_return"].dropna()
    return float(rows.iloc[0]) if len(rows) else np.nan


def run_one_month(
    dataset: pd.DataFrame,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    signal_date: pd.Timestamp,
    original_features: list[str],
    neighbor_features: list[str],
    config: dict,
) -> tuple[dict, pd.DataFrame]:
    """Train, build the portfolio, and realize the return for one signal month."""
    visible = dataset[dataset["date"] <= signal_date].copy()

    visible_original = point_in_time_impute(visible, original_features)
    visible_neighbor = point_in_time_impute(visible, neighbor_features)

    _, original_predictions = train_predict_latest(
        df=visible_original,
        feature_cols=original_features,
        config=config,
        signal_date=signal_date,
    )

    _, neighbor_predictions = train_predict_latest(
        df=visible_neighbor,
        feature_cols=neighbor_features,
        config=config,
        signal_date=signal_date,
    )

    final_portfolio, _, regime_status = build_final_portfolio(
        original_predictions=original_predictions,
        neighbor_predictions=neighbor_predictions,
        dataset=visible,
        prices=prices,
        universe=universe,
        signal_date=signal_date,
        config=config,
    )

    realized = dataset.loc[
        dataset["date"] == signal_date, ["ticker", "future_1m_return"]
    ].copy()

    holdings = final_portfolio[["ticker", "final_weight"]].merge(
        realized, on="ticker", how="left"
    )
    holdings.loc[holdings["ticker"] == "CASH", "future_1m_return"] = 0.0

    missing_mask = holdings["future_1m_return"].isna()
    missing_weight = float(holdings.loc[missing_mask, "final_weight"].sum())
    holdings["future_1m_return"] = holdings["future_1m_return"].fillna(0.0)

    gross_return = float(
        (holdings["final_weight"] * holdings["future_1m_return"]).sum()
    )

    cash_weight = float(
        holdings.loc[holdings["ticker"] == "CASH", "final_weight"].sum()
    )

    result = {
        "signal_date": signal_date,
        "gross_return": gross_return,
        "spy_return": spy_return_for_date(dataset, signal_date),
        "n_positions": int((holdings["ticker"] != "CASH").sum()),
        "cash_weight": cash_weight,
        "risk_on": bool(regime_status["risk_on"]),
        "tech_drawdown": float(regime_status["tech_drawdown"]),
        "missing_return_weight": missing_weight,
    }

    holdings.insert(0, "signal_date", signal_date)
    return result, holdings


def compute_net_returns(
    results: pd.DataFrame,
    holdings: pd.DataFrame,
    transaction_cost: float,
) -> pd.DataFrame:
    """Add turnover and net-of-cost returns to the monthly results table.

    Turnover is two-way: sum of |target weight - drifted prior weight| across
    all stock tickers (cash rebalancing itself is free; the stock trades that
    produce it are what get charged). The first month pays for the initial buy.
    """
    results = results.sort_values("signal_date").reset_index(drop=True)
    holdings = holdings.copy()
    holdings["signal_date"] = pd.to_datetime(holdings["signal_date"])

    prev_weights: pd.Series | None = None
    turnovers = []

    for signal_date in results["signal_date"]:
        month = holdings[holdings["signal_date"] == signal_date]
        target = month.set_index("ticker")["final_weight"]

        if prev_weights is None:
            turnover = float(target.drop(index="CASH", errors="ignore").abs().sum())
        else:
            tickers = target.index.union(prev_weights.index).drop("CASH", errors="ignore")
            turnover = float(
                (target.reindex(tickers, fill_value=0.0)
                 - prev_weights.reindex(tickers, fill_value=0.0)).abs().sum()
            )
        turnovers.append(turnover)

        # Drift this month's book by its realized returns to get next month's
        # starting weights.
        grown = target * (1.0 + month.set_index("ticker")["future_1m_return"])
        total = grown.sum()
        prev_weights = grown / total if total > 0 else target

    results["turnover"] = turnovers
    results["cost"] = results["turnover"] * transaction_cost
    results["net_return"] = results["gross_return"] - results["cost"]
    return results


def build_summaries(
    results: pd.DataFrame,
    n_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Overall / per-fold / per-year summary tables from the monthly results."""
    results = results.sort_values("signal_date").reset_index(drop=True)
    idx = pd.DatetimeIndex(results["signal_date"])
    net = pd.Series(results["net_return"].values, index=idx)
    gross = pd.Series(results["gross_return"].values, index=idx)
    spy = pd.Series(results["spy_return"].values, index=idx)

    overall = pd.DataFrame([
        summarize_returns(net, spy, label="strategy_net"),
        summarize_returns(gross, spy, label="strategy_gross"),
        summarize_returns(spy, label="spy_buy_and_hold"),
    ])

    fold_rows = []
    for fold_number, fold_dates in contiguous_folds(list(idx), n_folds):
        fold_net = net.loc[fold_dates]
        fold_spy = spy.loc[fold_dates]
        row = summarize_returns(fold_net, fold_spy, label=f"fold_{fold_number}")
        row["fold"] = fold_number
        row["spy_annualized_return"] = summarize_returns(fold_spy)["annualized_return"]
        row["spy_sharpe_ratio"] = summarize_returns(fold_spy)["sharpe_ratio"]
        fold_rows.append(row)
    folds = pd.DataFrame(fold_rows)

    yearly_rows = []
    for year, year_net in net.groupby(net.index.year):
        year_spy = spy[spy.index.year == year]
        row = summarize_returns(year_net, year_spy, label=str(year))
        row["spy_total_return"] = summarize_returns(year_spy)["total_return"]
        yearly_rows.append(row)
    yearly = pd.DataFrame(yearly_rows)

    return overall, folds, yearly


def write_report(
    path: str,
    config: dict,
    results: pd.DataFrame,
    overall: pd.DataFrame,
    folds: pd.DataFrame,
    yearly: pd.DataFrame,
    n_folds: int,
) -> None:
    net_row = overall[overall["label"] == "strategy_net"].iloc[0]
    spy_row = overall[overall["label"] == "spy_buy_and_hold"].iloc[0]

    fmt_cols = [
        "label", "months", "total_return", "annualized_return",
        "annualized_volatility", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "pct_positive_months",
    ]

    lines = []
    lines.append("LTSAF_live_v1 Walk-Forward Cross-Validation Backtest")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Model: {config['final_model_name']}")
    lines.append(f"Test window: {results['signal_date'].min().date()} to "
                 f"{results['signal_date'].max().date()} "
                 f"({len(results)} monthly out-of-sample signals)")
    lines.append(f"Transaction cost: {config['portfolio']['transaction_cost']:.4f} per dollar traded")
    lines.append(f"Avg two-way turnover: {results['turnover'].mean():.2f}/month")
    lines.append(f"Regime filter risk-off months: {int((~results['risk_on']).sum())}")
    lines.append("")
    lines.append("HEADLINE (net of costs, rf=0)")
    lines.append(f"  Annualized return: {net_row['annualized_return']:.2%}  "
                 f"(SPY: {spy_row['annualized_return']:.2%})")
    lines.append(f"  Sharpe ratio:      {net_row['sharpe_ratio']:.2f}  "
                 f"(SPY: {spy_row['sharpe_ratio']:.2f})")
    lines.append(f"  Max drawdown:      {net_row['max_drawdown']:.2%}  "
                 f"(SPY: {spy_row['max_drawdown']:.2%})")
    lines.append(f"  Fold dispersion:   Sharpe {folds['sharpe_ratio'].mean():.2f} "
                 f"+/- {folds['sharpe_ratio'].std(ddof=1):.2f} across {n_folds} folds")
    lines.append("")
    lines.append("OVERALL")
    lines.append(overall[fmt_cols + ["information_ratio_vs_benchmark", "hit_rate_vs_benchmark"]]
                 .to_string(index=False))
    lines.append("")
    lines.append(f"CONTIGUOUS FOLDS (n={n_folds})")
    lines.append(folds[["fold", "start", "end"] + fmt_cols[1:]
                       + ["spy_annualized_return", "spy_sharpe_ratio"]].to_string(index=False))
    lines.append("")
    lines.append("BY CALENDAR YEAR (net)")
    lines.append(yearly[fmt_cols + ["spy_total_return"]].to_string(index=False))
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    started = time.time()

    os.chdir(PROJECT_ROOT)
    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(args.output_dir, MONTHLY_RESULTS_FILE)
    holdings_path = os.path.join(args.output_dir, HOLDINGS_FILE)

    if args.fresh:
        for path in (results_path, holdings_path):
            if os.path.exists(path):
                os.remove(path)

    dataset = load_neighbor_dataset(config)
    prices = load_monthly_prices(config)
    universe = load_universe(config)

    original_features = get_feature_columns(
        dataset, config["portfolio"]["original_branch"]["feature_set"]
    )
    neighbor_features = get_feature_columns(
        dataset, config["portfolio"]["latent_neighbor_branch"]["feature_set"]
    )
    validate_no_leakage(original_features)
    validate_no_leakage(neighbor_features)

    test_dates = get_test_dates(dataset, args.min_train_months)
    if args.limit_months > 0:
        test_dates = test_dates[: args.limit_months]

    done_dates: set[pd.Timestamp] = set()
    if os.path.exists(results_path):
        existing = pd.read_csv(results_path, parse_dates=["signal_date"])
        done_dates = set(pd.to_datetime(existing["signal_date"]))

    pending = [d for d in test_dates if d not in done_dates]

    print(f"Test months: {len(test_dates)} "
          f"({test_dates[0].date()} to {test_dates[-1].date()}) | "
          f"already done: {len(done_dates & set(test_dates))} | to run: {len(pending)}")
    print(f"Features: original={len(original_features)}, neighbor={len(neighbor_features)}")

    for i, signal_date in enumerate(pending, start=1):
        month_started = time.time()
        result, holdings = run_one_month(
            dataset=dataset,
            prices=prices,
            universe=universe,
            signal_date=signal_date,
            original_features=original_features,
            neighbor_features=neighbor_features,
            config=config,
        )

        pd.DataFrame([result]).to_csv(
            results_path, mode="a", header=not os.path.exists(results_path), index=False
        )
        holdings.to_csv(
            holdings_path, mode="a", header=not os.path.exists(holdings_path), index=False
        )

        elapsed = time.time() - month_started
        print(f"[{i}/{len(pending)}] {signal_date.date()} "
              f"gross={result['gross_return']:+.2%} spy={result['spy_return']:+.2%} "
              f"risk_on={result['risk_on']} ({elapsed:.0f}s)", flush=True)

    # Rebuild summaries from the full checkpoint every run, so a resumed or
    # partial run still produces consistent outputs.
    results = pd.read_csv(results_path, parse_dates=["signal_date"])
    results = results[results["signal_date"].isin(test_dates)]
    holdings = pd.read_csv(holdings_path, parse_dates=["signal_date"])

    results = compute_net_returns(
        results, holdings, float(config["portfolio"]["transaction_cost"])
    )
    results.to_csv(results_path, index=False)

    overall, folds, yearly = build_summaries(results, args.folds)
    overall.to_csv(os.path.join(args.output_dir, OVERALL_SUMMARY_FILE), index=False)
    folds.to_csv(os.path.join(args.output_dir, FOLD_SUMMARY_FILE), index=False)
    yearly.to_csv(os.path.join(args.output_dir, YEARLY_SUMMARY_FILE), index=False)

    report_path = os.path.join(args.output_dir, REPORT_FILE)
    write_report(report_path, config, results, overall, folds, yearly, args.folds)

    print("")
    with open(report_path, encoding="utf-8") as f:
        print(f.read())
    print(f"Total runtime: {(time.time() - started) / 60:.1f} min")
    print(f"Outputs in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
