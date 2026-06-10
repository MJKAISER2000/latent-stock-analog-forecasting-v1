import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


STRUCTURAL_POOL_SIZES = [15, 20, 25, 30]
FINAL_TOP_NS = [5, 10]


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


def build_gated_signal_baskets(
    predictions: pd.DataFrame,
    structural_pool_size: int,
    final_top_n: int,
) -> pd.DataFrame:
    rows = []

    for date, group in predictions.groupby("date"):
        structural_pool = (
            group.sort_values("score_36m", ascending=False)
            .head(structural_pool_size)
            .copy()
        )

        final_selection = (
            structural_pool.sort_values("score_1m", ascending=False)
            .head(final_top_n)
            .copy()
        )

        rows.append(
            {
                "signal_date": date,
                "structural_pool_size": structural_pool_size,
                "final_top_n": final_top_n,
                "tickers": final_selection["ticker"].tolist(),
                "structural_candidates": ", ".join(structural_pool["ticker"].tolist()),
                "selected_tickers": ", ".join(final_selection["ticker"].tolist()),
            }
        )

    return pd.DataFrame(rows)


def build_pure_signal_baskets(
    predictions: pd.DataFrame,
    top_n: int,
    score_col: str,
    label: str,
) -> pd.DataFrame:
    rows = []

    for date, group in predictions.groupby("date"):
        selected = group.sort_values(score_col, ascending=False).head(top_n)

        rows.append(
            {
                "signal_date": date,
                "structural_pool_size": label,
                "final_top_n": top_n,
                "tickers": selected["ticker"].tolist(),
                "structural_candidates": "",
                "selected_tickers": ", ".join(selected["ticker"].tolist()),
            }
        )

    return pd.DataFrame(rows)


def build_overlapping_returns(
    baskets: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    horizon_months: int,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    all_dates = sorted(pd.to_datetime(monthly_returns["date"].unique()))

    basket_rows = []

    for _, row in baskets.iterrows():
        signal_date = pd.to_datetime(row["signal_date"])
        tickers = row["tickers"]

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
        return pd.DataFrame(
            columns=[
                "date",
                "strategy_monthly_return",
                "active_basket_count",
            ]
        )

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

    return (
        monthly_returns[monthly_returns["date"].isin(dates)]
        .groupby("date")["monthly_return"]
        .mean()
        .reset_index()
        .rename(columns={"monthly_return": "equal_weight_monthly_return"})
    )


def build_spy_benchmark(monthly_returns: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(dates)

    spy = monthly_returns[
        (monthly_returns["ticker"] == "SPY")
        & (monthly_returns["date"].isin(dates))
    ].copy()

    return spy[["date", "monthly_return"]].rename(
        columns={"monthly_return": "spy_monthly_return"}
    )


def add_cumulative(df: pd.DataFrame, return_col: str, cumulative_col: str) -> pd.DataFrame:
    df = df.copy()
    df[cumulative_col] = (1 + df[return_col]).cumprod()
    return df


def plot_curves(curves: pd.DataFrame, output_path: str):
    plt.figure(figsize=(11, 7))

    for col in curves.columns:
        if col == "date":
            continue

        if col.endswith("_cumulative"):
            plt.plot(
                curves["date"],
                curves[col],
                label=col.replace("_cumulative", ""),
            )

    plt.title("Week 13 Gated Two-System Model Backtest")
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

    h1_path = "outputs/tables/week13_predictions_horizon_1m.csv"
    h36_path = "outputs/tables/week13_predictions_horizon_36m.csv"

    print("Loading 1-month tactical predictions...")
    h1 = pd.read_csv(h1_path)
    h1.columns = h1.columns.str.strip()
    h1["date"] = pd.to_datetime(h1["date"])
    h1["ticker"] = h1["ticker"].astype(str).str.strip()

    print("Loading 36-month structural predictions...")
    h36 = pd.read_csv(h36_path)
    h36.columns = h36.columns.str.strip()
    h36["date"] = pd.to_datetime(h36["date"])
    h36["ticker"] = h36["ticker"].astype(str).str.strip()

    h1 = h1[["date", "ticker", "predicted_prob_outperform_h"]].rename(
        columns={"predicted_prob_outperform_h": "score_1m"}
    )

    h36 = h36[["date", "ticker", "predicted_prob_outperform_h"]].rename(
        columns={"predicted_prob_outperform_h": "score_36m"}
    )

    combined = h36.merge(h1, on=["date", "ticker"], how="inner")
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)

    print("Combined prediction shape:", combined.shape)
    print("Date range:", combined["date"].min(), "to", combined["date"].max())
    print("Ticker count:", combined["ticker"].nunique())

    if combined.empty:
        raise ValueError("Combined prediction dataframe is empty.")

    monthly_returns = load_monthly_returns()
    transaction_cost = 0.001
    holding_horizon_months = 36

    all_stats = []
    all_curves = None
    holdings_rows = []

    strategy_configs = []

    # Baselines
    for top_n in FINAL_TOP_NS:
        strategy_configs.append(
            {
                "strategy_name": f"pure_36m_top{top_n}",
                "baskets": build_pure_signal_baskets(
                    predictions=combined,
                    top_n=top_n,
                    score_col="score_36m",
                    label="pure_36m",
                ),
            }
        )

        strategy_configs.append(
            {
                "strategy_name": f"pure_1m_top{top_n}",
                "baskets": build_pure_signal_baskets(
                    predictions=combined,
                    top_n=top_n,
                    score_col="score_1m",
                    label="pure_1m",
                ),
            }
        )

    # Gated versions
    for pool_size in STRUCTURAL_POOL_SIZES:
        for final_top_n in FINAL_TOP_NS:
            strategy_configs.append(
                {
                    "strategy_name": f"gated_pool{pool_size}_top{final_top_n}",
                    "baskets": build_gated_signal_baskets(
                        predictions=combined,
                        structural_pool_size=pool_size,
                        final_top_n=final_top_n,
                    ),
                }
            )

    for config in strategy_configs:
        strategy_name = config["strategy_name"]
        baskets = config["baskets"]

        print(f"Backtesting {strategy_name}")

        strategy = build_overlapping_returns(
            baskets=baskets,
            monthly_returns=monthly_returns,
            horizon_months=holding_horizon_months,
            transaction_cost=transaction_cost,
        )

        if strategy.empty:
            continue

        stats = {
            "strategy": strategy_name,
            "holding_horizon_months": holding_horizon_months,
            **performance_stats(strategy["strategy_monthly_return"]),
            "avg_active_basket_count": strategy["active_basket_count"].mean(),
        }

        all_stats.append(stats)

        curve = strategy[["date", "strategy_monthly_return"]].rename(
            columns={
                "strategy_monthly_return": f"{strategy_name}_return",
            }
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

        for _, row in baskets.iterrows():
            holdings_rows.append(
                {
                    "strategy": strategy_name,
                    "signal_date": row["signal_date"],
                    "selected_tickers": row["selected_tickers"],
                    "structural_candidates": row["structural_candidates"],
                }
            )

    if all_curves is None:
        raise ValueError("No strategy curves were created.")

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

    benchmark_rows = [
        {
            "strategy": "equal_weight",
            "holding_horizon_months": "benchmark",
            **performance_stats(all_curves["equal_weight_monthly_return"]),
            "avg_active_basket_count": np.nan,
        },
        {
            "strategy": "spy",
            "holding_horizon_months": "benchmark",
            **performance_stats(all_curves["spy_monthly_return"]),
            "avg_active_basket_count": np.nan,
        },
    ]

    stats_df = pd.DataFrame(all_stats + benchmark_rows)
    holdings_df = pd.DataFrame(holdings_rows)

    stats_path = "outputs/tables/week13_gated_two_system_backtest_stats.csv"
    curves_path = "outputs/tables/week13_gated_two_system_backtest_curves.csv"
    holdings_path = "outputs/tables/week13_gated_two_system_holdings.csv"
    figure_path = "outputs/figures/week13_gated_two_system_equity_curves.png"
    report_path = "outputs/reports/week13_gated_two_system_model_summary.txt"

    stats_df.to_csv(stats_path, index=False)
    all_curves.to_csv(curves_path, index=False)
    holdings_df.to_csv(holdings_path, index=False)

    plot_cols = ["date"]

    preferred = [
        "pure_36m_top10_cumulative",
        "pure_1m_top10_cumulative",
        "gated_pool15_top10_cumulative",
        "gated_pool20_top10_cumulative",
        "gated_pool25_top10_cumulative",
        "gated_pool30_top10_cumulative",
        "equal_weight_cumulative",
        "spy_cumulative",
    ]

    for col in preferred:
        if col in all_curves.columns:
            plot_cols.append(col)

    plot_curves(all_curves[plot_cols], figure_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 13 Gated Two-System Model Summary\n")
        f.write("=====================================\n\n")
        f.write("Goal:\n")
        f.write(
            "Test whether the 1-month tactical model improves entry selection inside the 36-month structural candidate pool.\n\n"
        )
        f.write("Architecture:\n")
        f.write("1. Rank all stocks by 36-month structural score.\n")
        f.write("2. Keep top K structural candidates.\n")
        f.write("3. Rank those candidates by 1-month tactical score.\n")
        f.write("4. Select final top N.\n")
        f.write("5. Hold using 36-month overlapping portfolios.\n\n")
        f.write("Structural pool sizes tested:\n")
        f.write(str(STRUCTURAL_POOL_SIZES))
        f.write("\n\n")
        f.write("Final top-N values tested:\n")
        f.write(str(FINAL_TOP_NS))
        f.write("\n\n")
        f.write("Performance stats:\n")
        f.write(
            stats_df.sort_values("annualized_return", ascending=False)
            .to_string(index=False)
        )
        f.write("\n")

    print("")
    print("Saved:", stats_path)
    print("Saved:", curves_path)
    print("Saved:", holdings_path)
    print("Saved:", figure_path)
    print("Saved:", report_path)
    print("")
    print("Top strategies:")
    print(
        stats_df.sort_values("annualized_return", ascending=False)
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()