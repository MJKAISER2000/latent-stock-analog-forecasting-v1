import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "rank_by_date"]).reset_index(drop=True)
    return df


def compute_monthly_strategy_returns(
    predictions: pd.DataFrame,
    top_n: int,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """
    Builds a simple monthly-rebalanced strategy.

    At each month:
    - choose top N stocks by predicted probability
    - portfolio return = average future 12-month return of selected stocks
    - convert approximate 12-month return into approximate monthly equivalent
    - subtract transaction cost per rebalance

    Note:
    This is still a simplified backtest. A more realistic version later will use
    realized next-month returns instead of 12-month forward returns.
    """

    rows = []

    for date, group in predictions.groupby("date"):
        top = group.sort_values("predicted_prob_outperform", ascending=False).head(top_n)

        avg_future_12m_return = top["future_12m_return"].mean()
        avg_spy_future_12m_return = top["future_12m_spy_return"].mean()

        # Approximate monthly equivalent from 12-month forward return
        strategy_monthly_return = (1 + avg_future_12m_return) ** (1 / 12) - 1
        spy_monthly_return = (1 + avg_spy_future_12m_return) ** (1 / 12) - 1

        # Transaction cost approximation
        strategy_monthly_return_after_cost = strategy_monthly_return - transaction_cost

        rows.append(
            {
                "date": date,
                "top_n": top_n,
                "avg_future_12m_return": avg_future_12m_return,
                "strategy_monthly_return": strategy_monthly_return,
                "strategy_monthly_return_after_cost": strategy_monthly_return_after_cost,
                "spy_monthly_return": spy_monthly_return,
                "selected_tickers": ", ".join(top["ticker"].tolist()),
            }
        )

    return pd.DataFrame(rows)


def compute_equal_weight_all_stock_benchmark(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Equal-weight benchmark across all available stocks in the test universe.
    """
    rows = []

    for date, group in predictions.groupby("date"):
        avg_future_12m_return = group["future_12m_return"].mean()
        avg_spy_future_12m_return = group["future_12m_spy_return"].mean()

        equal_weight_monthly_return = (1 + avg_future_12m_return) ** (1 / 12) - 1
        spy_monthly_return = (1 + avg_spy_future_12m_return) ** (1 / 12) - 1

        rows.append(
            {
                "date": date,
                "equal_weight_monthly_return": equal_weight_monthly_return,
                "spy_monthly_return": spy_monthly_return,
            }
        )

    return pd.DataFrame(rows)


def add_cumulative_returns(df: pd.DataFrame, return_col: str, output_col: str) -> pd.DataFrame:
    df = df.copy()
    df[output_col] = (1 + df[return_col]).cumprod()
    return df


def performance_stats(monthly_returns: pd.Series) -> dict:
    monthly_returns = monthly_returns.dropna()

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


def plot_equity_curves(curves: pd.DataFrame, output_path: str):
    plt.figure(figsize=(11, 7))

    plt.plot(curves["date"], curves["top5_after_cost_cumulative"], label="Latent Top 5")
    plt.plot(curves["date"], curves["top10_after_cost_cumulative"], label="Latent Top 10")
    plt.plot(curves["date"], curves["equal_weight_cumulative"], label="Equal Weight All Stocks")
    plt.plot(curves["date"], curves["spy_cumulative"], label="SPY")

    plt.title("Week 6 Backtest Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved equity curve plot to {output_path}")


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    prediction_path = "outputs/tables/week4_predictions_latent20_gradient_boosting.csv"

    print("Loading predictions from:", prediction_path)
    predictions = load_predictions(prediction_path)

    print("Prediction date range:")
    print(predictions["date"].min(), "to", predictions["date"].max())

    top5 = compute_monthly_strategy_returns(
        predictions,
        top_n=5,
        transaction_cost=0.001,
    )

    top10 = compute_monthly_strategy_returns(
        predictions,
        top_n=10,
        transaction_cost=0.001,
    )

    equal_weight = compute_equal_weight_all_stock_benchmark(predictions)

    top5 = add_cumulative_returns(
        top5,
        "strategy_monthly_return_after_cost",
        "top5_after_cost_cumulative",
    )

    top10 = add_cumulative_returns(
        top10,
        "strategy_monthly_return_after_cost",
        "top10_after_cost_cumulative",
    )

    equal_weight = add_cumulative_returns(
        equal_weight,
        "equal_weight_monthly_return",
        "equal_weight_cumulative",
    )

    equal_weight = add_cumulative_returns(
        equal_weight,
        "spy_monthly_return",
        "spy_cumulative",
    )

    curves = top5[
        [
            "date",
            "strategy_monthly_return_after_cost",
            "top5_after_cost_cumulative",
            "selected_tickers",
        ]
    ].rename(
        columns={
            "strategy_monthly_return_after_cost": "top5_monthly_return_after_cost",
            "selected_tickers": "top5_selected_tickers",
        }
    )

    curves = curves.merge(
        top10[
            [
                "date",
                "strategy_monthly_return_after_cost",
                "top10_after_cost_cumulative",
                "selected_tickers",
            ]
        ].rename(
            columns={
                "strategy_monthly_return_after_cost": "top10_monthly_return_after_cost",
                "selected_tickers": "top10_selected_tickers",
            }
        ),
        on="date",
        how="left",
    )

    curves = curves.merge(
        equal_weight[
            [
                "date",
                "equal_weight_monthly_return",
                "spy_monthly_return",
                "equal_weight_cumulative",
                "spy_cumulative",
            ]
        ],
        on="date",
        how="left",
    )

    stats_rows = []

    stats_rows.append(
        {
            "strategy": "latent_top5_after_cost",
            **performance_stats(curves["top5_monthly_return_after_cost"]),
        }
    )

    stats_rows.append(
        {
            "strategy": "latent_top10_after_cost",
            **performance_stats(curves["top10_monthly_return_after_cost"]),
        }
    )

    stats_rows.append(
        {
            "strategy": "equal_weight_all_stocks",
            **performance_stats(curves["equal_weight_monthly_return"]),
        }
    )

    stats_rows.append(
        {
            "strategy": "spy",
            **performance_stats(curves["spy_monthly_return"]),
        }
    )

    stats = pd.DataFrame(stats_rows)

    curves_path = "outputs/tables/week6_backtest_curves.csv"
    stats_path = "outputs/tables/week6_backtest_stats.csv"
    top5_path = "outputs/tables/week6_top5_holdings_by_month.csv"
    top10_path = "outputs/tables/week6_top10_holdings_by_month.csv"

    curves.to_csv(curves_path, index=False)
    stats.to_csv(stats_path, index=False)
    top5.to_csv(top5_path, index=False)
    top10.to_csv(top10_path, index=False)

    plot_equity_curves(curves, "outputs/figures/week6_backtest_equity_curves.png")

    report_path = "outputs/reports/week6_backtest_summary.txt"

    with open(report_path, "w") as f:
        f.write("Week 6 Backtest Summary\n")
        f.write("=======================\n\n")
        f.write("Model used:\n")
        f.write("latent_dim=20 autoencoder + gradient boosting classifier\n\n")
        f.write("Backtest setup:\n")
        f.write("- monthly rebalance\n")
        f.write("- top 5 and top 10 model-ranked stocks\n")
        f.write("- 0.1% transaction cost approximation per rebalance\n")
        f.write("- compared against SPY and equal-weight all-stock benchmark\n\n")
        f.write("Important limitation:\n")
        f.write(
            "This is a simplified backtest that converts 12-month forward returns "
            "into approximate monthly returns. A later version should use actual "
            "next-month realized returns for a fully realistic backtest.\n\n"
        )
        f.write("Performance stats:\n")
        f.write(stats.to_string(index=False))
        f.write("\n\n")
        f.write("First few equity curve rows:\n")
        f.write(curves.head().to_string(index=False))
        f.write("\n")

    print("")
    print("Saved curves to:", curves_path)
    print("Saved stats to:", stats_path)
    print("Saved holdings to:", top5_path, "and", top10_path)
    print("Saved report to:", report_path)
    print("")
    print("Backtest stats:")
    print(stats)


if __name__ == "__main__":
    main()