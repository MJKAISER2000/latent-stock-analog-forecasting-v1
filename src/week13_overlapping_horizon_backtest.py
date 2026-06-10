import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


HORIZONS = [1, 3, 6, 12, 24, 36]
TOP_NS = [5, 10, 25, 50]


def performance_stats(monthly_returns: pd.Series) -> dict:
    monthly_returns = monthly_returns.dropna()

    if len(monthly_returns) == 0:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_no_risk_free": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
        }

    total_return = (1 + monthly_returns).prod() - 1
    annualized_return = (1 + total_return) ** (12 / len(monthly_returns)) - 1
    annualized_volatility = monthly_returns.std() * np.sqrt(12)

    sharpe = np.nan
    if annualized_volatility != 0:
        sharpe = annualized_return / annualized_volatility

    cumulative = (1 + monthly_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = drawdown.min()

    win_rate = (monthly_returns > 0).mean()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_no_risk_free": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }


def load_monthly_returns() -> pd.DataFrame:
    """
    Uses expanded monthly prices to compute actual one-month returns.
    These are used to value overlapping portfolios month by month.
    """
    prices_path = "data/processed/expanded_monthly_prices.parquet"

    prices = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    monthly_returns = prices / prices.shift(1) - 1

    rows = []

    for ticker in monthly_returns.columns:
        for date in monthly_returns.index:
            ret = monthly_returns.loc[date, ticker]

            if pd.isna(ret):
                continue

            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "monthly_return": ret,
                }
            )

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out


def build_signal_baskets(predictions: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    For each signal date, select top N tickers.
    """
    rows = []

    for date, group in predictions.groupby("date"):
        top = group.sort_values("predicted_prob_outperform_h", ascending=False).head(top_n)

        rows.append(
            {
                "signal_date": date,
                "top_n": top_n,
                "tickers": top["ticker"].tolist(),
            }
        )

    return pd.DataFrame(rows)


def build_overlapping_returns(
    baskets: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    horizon_months: int,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """
    Overlapping holding-period backtest.

    Each month:
    - a new basket is opened
    - each basket stays active for horizon_months
    - portfolio return is the average return of all active baskets
    """

    all_dates = sorted(monthly_returns["date"].unique())
    all_dates = pd.to_datetime(all_dates)

    basket_rows = []

    for _, row in baskets.iterrows():
        signal_date = pd.to_datetime(row["signal_date"])
        tickers = row["tickers"]

        # First investable month is the month after signal date.
        future_dates = [d for d in all_dates if d > signal_date]
        holding_dates = future_dates[:horizon_months]

        for holding_idx, hold_date in enumerate(holding_dates, start=1):
            month_rets = monthly_returns[
                (monthly_returns["date"] == hold_date)
                & (monthly_returns["ticker"].isin(tickers))
            ]

            if len(month_rets) == 0:
                continue

            basket_return = month_rets["monthly_return"].mean()

            # Charge transaction cost only at basket opening.
            if holding_idx == 1:
                basket_return = basket_return - transaction_cost

            basket_rows.append(
                {
                    "date": hold_date,
                    "signal_date": signal_date,
                    "basket_return": basket_return,
                    "active_tickers": ", ".join(tickers),
                }
            )

    basket_returns = pd.DataFrame(basket_rows)

    if basket_returns.empty:
        return pd.DataFrame(columns=["date", "strategy_monthly_return", "active_basket_count"])

    strategy = (
        basket_returns.groupby("date")
        .agg(
            strategy_monthly_return=("basket_return", "mean"),
            active_basket_count=("basket_return", "count"),
        )
        .reset_index()
    )

    strategy["date"] = pd.to_datetime(strategy["date"])
    strategy = strategy.sort_values("date").reset_index(drop=True)

    return strategy


def build_equal_weight_benchmark(monthly_returns: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(dates)

    eq = (
        monthly_returns[monthly_returns["date"].isin(dates)]
        .groupby("date")["monthly_return"]
        .mean()
        .reset_index()
        .rename(columns={"monthly_return": "equal_weight_monthly_return"})
    )

    return eq


def build_spy_benchmark(monthly_returns: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(dates)

    spy = monthly_returns[
        (monthly_returns["ticker"] == "SPY")
        & (monthly_returns["date"].isin(dates))
    ].copy()

    spy = spy[["date", "monthly_return"]].rename(
        columns={"monthly_return": "spy_monthly_return"}
    )

    return spy


def add_cumulative(df: pd.DataFrame, return_col: str, cumulative_col: str) -> pd.DataFrame:
    df = df.copy()
    df[cumulative_col] = (1 + df[return_col]).cumprod()
    return df


def plot_best_curves(curves: pd.DataFrame, output_path: str):
    plt.figure(figsize=(11, 7))

    for col in curves.columns:
        if col.endswith("_cumulative") and col not in [
            "equal_weight_cumulative",
            "spy_cumulative",
        ]:
            plt.plot(curves["date"], curves[col], label=col.replace("_cumulative", ""))

    if "equal_weight_cumulative" in curves.columns:
        plt.plot(curves["date"], curves["equal_weight_cumulative"], label="Equal Weight")

    if "spy_cumulative" in curves.columns:
        plt.plot(curves["date"], curves["spy_cumulative"], label="SPY")

    plt.title("Week 13 Overlapping Horizon Backtest")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print("Saved plot:", output_path)


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    monthly_returns = load_monthly_returns()

    all_stats = []
    all_curves = None

    transaction_cost = 0.001

    for horizon in HORIZONS:
        pred_path = f"outputs/tables/week13_predictions_horizon_{horizon}m.csv"

        print("")
        print("=" * 80)
        print(f"Backtesting horizon={horizon} months")
        print("Loading:", pred_path)

        predictions = pd.read_csv(pred_path)
        predictions["date"] = pd.to_datetime(predictions["date"])

        for top_n in TOP_NS:
            print(f"Top {top_n}")

            baskets = build_signal_baskets(predictions, top_n=top_n)

            strategy = build_overlapping_returns(
                baskets=baskets,
                monthly_returns=monthly_returns,
                horizon_months=horizon,
                transaction_cost=transaction_cost,
            )

            if strategy.empty:
                continue

            strategy_name = f"h{horizon}_top{top_n}"

            stats = {
                "horizon_months": horizon,
                "top_n": top_n,
                "strategy": strategy_name,
                **performance_stats(strategy["strategy_monthly_return"]),
                "avg_active_basket_count": strategy["active_basket_count"].mean(),
            }

            all_stats.append(stats)

            curve = strategy[["date", "strategy_monthly_return"]].rename(
                columns={"strategy_monthly_return": f"{strategy_name}_return"}
            )

            curve = add_cumulative(
                curve,
                f"{strategy_name}_return",
                f"{strategy_name}_cumulative",
            )

            if all_curves is None:
                all_curves = curve
            else:
                all_curves = all_curves.merge(curve, on="date", how="outer")

    all_curves = all_curves.sort_values("date").reset_index(drop=True)

    eq = build_equal_weight_benchmark(monthly_returns, all_curves["date"])
    spy = build_spy_benchmark(monthly_returns, all_curves["date"])

    all_curves = all_curves.merge(eq, on="date", how="left")
    all_curves = all_curves.merge(spy, on="date", how="left")

    all_curves = add_cumulative(
        all_curves,
        "equal_weight_monthly_return",
        "equal_weight_cumulative",
    )

    all_curves = add_cumulative(
        all_curves,
        "spy_monthly_return",
        "spy_cumulative",
    )

    benchmark_stats = [
        {
            "horizon_months": "benchmark",
            "top_n": "all",
            "strategy": "equal_weight",
            **performance_stats(all_curves["equal_weight_monthly_return"]),
            "avg_active_basket_count": np.nan,
        },
        {
            "horizon_months": "benchmark",
            "top_n": "spy",
            "strategy": "spy",
            **performance_stats(all_curves["spy_monthly_return"]),
            "avg_active_basket_count": np.nan,
        },
    ]

    stats_df = pd.DataFrame(all_stats + benchmark_stats)

    stats_path = "outputs/tables/week13_overlapping_horizon_backtest_stats.csv"
    curves_path = "outputs/tables/week13_overlapping_horizon_backtest_curves.csv"
    figure_path = "outputs/figures/week13_overlapping_horizon_equity_curves.png"
    report_path = "outputs/reports/week13_overlapping_horizon_backtest_summary.txt"

    stats_df.to_csv(stats_path, index=False)
    all_curves.to_csv(curves_path, index=False)

    # Plot only a manageable subset: top5 and top10 for all horizons + benchmarks.
    plot_cols = ["date", "equal_weight_cumulative", "spy_cumulative"]

    for horizon in HORIZONS:
        for top_n in [5, 10]:
            col = f"h{horizon}_top{top_n}_cumulative"
            if col in all_curves.columns:
                plot_cols.append(col)

    plot_best_curves(all_curves[plot_cols], figure_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 13 Overlapping Horizon Backtest Summary\n")
        f.write("===========================================\n\n")
        f.write("Goal:\n")
        f.write(
            "Test multiple prediction horizons using matching overlapping holding-period portfolios.\n\n"
        )
        f.write("Horizons tested:\n")
        f.write(str(HORIZONS))
        f.write("\n\n")
        f.write("Top-N portfolios tested:\n")
        f.write(str(TOP_NS))
        f.write("\n\n")
        f.write("Transaction cost per newly opened basket:\n")
        f.write(f"{transaction_cost:.3%}")
        f.write("\n\n")
        f.write("Performance stats:\n")
        f.write(stats_df.to_string(index=False))
        f.write("\n")

    print("")
    print("Saved:", stats_path)
    print("Saved:", curves_path)
    print("Saved:", figure_path)
    print("Saved:", report_path)
    print("")
    print("Top strategies by annualized return:")
    print(
        stats_df[
            ~stats_df["strategy"].isin(["equal_weight", "spy"])
        ].sort_values("annualized_return", ascending=False).head(15)
    )


if __name__ == "__main__":
    main()