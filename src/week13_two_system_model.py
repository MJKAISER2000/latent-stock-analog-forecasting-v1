import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


TOP_NS = [5, 10, 25]
WEIGHTS = [
    (0.90, 0.10),
    (0.80, 0.20),
    (0.70, 0.30),
    (0.60, 0.40),
    (0.50, 0.50),
]


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

    if annualized_volatility == 0:
        sharpe = np.nan
    else:
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


def minmax_by_date(df: pd.DataFrame, score_col: str, output_col: str) -> pd.DataFrame:
    """
    Safer date-wise min-max scaling.

    This avoids groupby.apply because that can sometimes mess with the index/columns
    depending on pandas version.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    group_min = df.groupby("date")[score_col].transform("min")
    group_max = df.groupby("date")[score_col].transform("max")
    denom = group_max - group_min

    df[output_col] = np.where(
        denom == 0,
        0.5,
        (df[score_col] - group_min) / denom,
    )

    df[output_col] = df[output_col].replace([np.inf, -np.inf], np.nan).fillna(0.5)

    return df


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


def build_signal_baskets(predictions: pd.DataFrame, top_n: int, score_col: str) -> pd.DataFrame:
    rows = []

    for date, group in predictions.groupby("date"):
        top = group.sort_values(score_col, ascending=False).head(top_n)

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

            # Charge transaction cost only when a new basket opens.
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

    plt.title("Week 13 Two-System Model Backtest")
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
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)

    print("Combined columns:", combined.columns.tolist())
    print("Combined prediction shape:", combined.shape)

    if combined.empty:
        raise ValueError(
            "The combined dataframe is empty. "
            "Check whether 1m and 36m prediction files share overlapping dates/tickers."
        )

    combined = minmax_by_date(combined, "score_36m", "score_36m_scaled")
    combined = minmax_by_date(combined, "score_1m", "score_1m_scaled")

    print("Date range:", combined["date"].min(), "to", combined["date"].max())
    print("Ticker count:", combined["ticker"].nunique())

    monthly_returns = load_monthly_returns()
    transaction_cost = 0.001

    all_stats = []
    all_curves = None
    holdings_rows = []

    combined["pure_36m_score"] = combined["score_36m_scaled"]
    combined["pure_1m_score"] = combined["score_1m_scaled"]

    score_configs = [
        ("pure_36m", "pure_36m_score"),
        ("pure_1m", "pure_1m_score"),
    ]

    for w36, w1 in WEIGHTS:
        name = f"combo_{int(w36 * 100)}struct_{int(w1 * 100)}tact"
        col = f"{name}_score"

        combined[col] = (
            w36 * combined["score_36m_scaled"]
            + w1 * combined["score_1m_scaled"]
        )

        score_configs.append((name, col))

    for strategy_name, score_col in score_configs:
        for top_n in TOP_NS:
            print(f"Backtesting {strategy_name}, top {top_n}")

            baskets = build_signal_baskets(
                predictions=combined,
                top_n=top_n,
                score_col=score_col,
            )

            # The combined model is fundamentally a long-horizon core model,
            # so use 36-month overlapping holding period.
            strategy = build_overlapping_returns(
                baskets=baskets,
                monthly_returns=monthly_returns,
                horizon_months=36,
                transaction_cost=transaction_cost,
            )

            if strategy.empty:
                continue

            full_strategy_name = f"{strategy_name}_top{top_n}"

            stats = {
                "strategy": full_strategy_name,
                "top_n": top_n,
                "holding_horizon_months": 36,
                **performance_stats(strategy["strategy_monthly_return"]),
                "avg_active_basket_count": strategy["active_basket_count"].mean(),
            }

            all_stats.append(stats)

            curve = strategy[["date", "strategy_monthly_return"]].rename(
                columns={
                    "strategy_monthly_return": f"{full_strategy_name}_return"
                }
            )

            curve = add_cumulative(
                curve,
                f"{full_strategy_name}_return",
                f"{full_strategy_name}_cumulative",
            )

            if all_curves is None:
                all_curves = curve
            else:
                all_curves = all_curves.merge(curve, on="date", how="outer")

            for _, row in baskets.iterrows():
                holdings_rows.append(
                    {
                        "strategy": full_strategy_name,
                        "signal_date": row["signal_date"],
                        "top_n": top_n,
                        "tickers": ", ".join(row["tickers"]),
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
            "top_n": "all",
            "holding_horizon_months": "benchmark",
            **performance_stats(all_curves["equal_weight_monthly_return"]),
            "avg_active_basket_count": np.nan,
        },
        {
            "strategy": "spy",
            "top_n": "spy",
            "holding_horizon_months": "benchmark",
            **performance_stats(all_curves["spy_monthly_return"]),
            "avg_active_basket_count": np.nan,
        },
    ]

    stats_df = pd.DataFrame(all_stats + benchmark_rows)
    holdings_df = pd.DataFrame(holdings_rows)

    plot_cols = ["date"]

    preferred = [
        "pure_36m_top10_cumulative",
        "pure_1m_top10_cumulative",
        "combo_90struct_10tact_top10_cumulative",
        "combo_80struct_20tact_top10_cumulative",
        "combo_70struct_30tact_top10_cumulative",
        "combo_60struct_40tact_top10_cumulative",
        "combo_50struct_50tact_top10_cumulative",
        "equal_weight_cumulative",
        "spy_cumulative",
    ]

    for col in preferred:
        if col in all_curves.columns and col not in plot_cols:
            plot_cols.append(col)

    stats_path = "outputs/tables/week13_two_system_backtest_stats.csv"
    curves_path = "outputs/tables/week13_two_system_backtest_curves.csv"
    holdings_path = "outputs/tables/week13_two_system_holdings.csv"
    combined_scores_path = "outputs/tables/week13_two_system_combined_scores.csv"
    figure_path = "outputs/figures/week13_two_system_equity_curves.png"
    report_path = "outputs/reports/week13_two_system_model_summary.txt"

    stats_df.to_csv(stats_path, index=False)
    all_curves.to_csv(curves_path, index=False)
    holdings_df.to_csv(holdings_path, index=False)
    combined.to_csv(combined_scores_path, index=False)

    plot_curves(all_curves[plot_cols], figure_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 13 Two-System Model Summary\n")
        f.write("================================\n\n")
        f.write("Goal:\n")
        f.write(
            "Combine the 36-month structural model with the 1-month tactical model.\n\n"
        )
        f.write("Architecture:\n")
        f.write("- 36-month model = structural ownership signal\n")
        f.write("- 1-month model = tactical timing signal\n")
        f.write(
            "- combined_score = w_structural * score_36m + w_tactical * score_1m\n\n"
        )
        f.write("Weights tested:\n")
        f.write(str(WEIGHTS))
        f.write("\n\n")
        f.write("Holding period:\n")
        f.write("36-month overlapping portfolios\n\n")
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
    print("Saved:", combined_scores_path)
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