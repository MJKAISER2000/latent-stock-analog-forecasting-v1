import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def build_top_n_returns(predictions: pd.DataFrame, top_n: int, transaction_cost: float) -> pd.DataFrame:
    rows = []

    for date, group in predictions.groupby("date"):
        group = group.dropna(subset=["realized_next_1m_return"])

        if len(group) == 0:
            continue

        selected = group.sort_values("predicted_prob_outperform", ascending=False).head(top_n)

        gross_return = selected["realized_next_1m_return"].mean()
        net_return = gross_return - transaction_cost

        rows.append(
            {
                "date": date,
                "top_n": top_n,
                "gross_monthly_return": gross_return,
                "net_monthly_return": net_return,
                "selected_tickers": ", ".join(selected["ticker"].tolist()),
            }
        )

    return pd.DataFrame(rows)


def build_equal_weight_returns(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for date, group in predictions.groupby("date"):
        group = group.dropna(subset=["realized_next_1m_return"])

        if len(group) == 0:
            continue

        rows.append(
            {
                "date": date,
                "equal_weight_monthly_return": group["realized_next_1m_return"].mean(),
            }
        )

    return pd.DataFrame(rows)


def build_spy_returns(realized: pd.DataFrame) -> pd.DataFrame:
    spy = realized[realized["ticker"] == "SPY"].copy()
    spy = spy.rename(columns={"realized_next_1m_return": "spy_monthly_return"})
    return spy[["date", "spy_monthly_return"]]


def add_cumulative(df: pd.DataFrame, return_col: str, cumulative_col: str) -> pd.DataFrame:
    df = df.copy()
    df[cumulative_col] = (1 + df[return_col]).cumprod()
    return df


def plot_curves(curves: pd.DataFrame, output_path: str):
    plt.figure(figsize=(11, 7))

    plt.plot(curves["date"], curves["top5_cumulative"], label="Top 5")
    plt.plot(curves["date"], curves["top10_cumulative"], label="Top 10")
    plt.plot(curves["date"], curves["top25_cumulative"], label="Top 25")
    plt.plot(curves["date"], curves["top50_cumulative"], label="Top 50")
    plt.plot(curves["date"], curves["equal_weight_cumulative"], label="Equal Weight Universe")
    plt.plot(curves["date"], curves["spy_cumulative"], label="SPY")

    plt.title("Week 11 True Next-Month Realized-Return Backtest")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print("Saved plot:", output_path)


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    predictions_path = "outputs/tables/week11_expanded_predictions_latent20_gb.csv"
    realized_path = "data/processed/expanded_realized_monthly_returns.parquet"

    print("Loading predictions...")
    pred = pd.read_csv(predictions_path)
    pred["date"] = pd.to_datetime(pred["date"])

    print("Loading realized monthly returns...")
    realized = pd.read_parquet(realized_path)
    realized["date"] = pd.to_datetime(realized["date"])

    print("Merging predictions with realized next-month returns...")
    merged = pred.merge(
        realized,
        on=["date", "ticker"],
        how="left",
    )

    print("Merged shape:", merged.shape)
    print("Missing realized returns:", merged["realized_next_1m_return"].isna().sum())

    transaction_cost = 0.001

    top5 = build_top_n_returns(merged, top_n=5, transaction_cost=transaction_cost)
    top10 = build_top_n_returns(merged, top_n=10, transaction_cost=transaction_cost)
    top25 = build_top_n_returns(merged, top_n=25, transaction_cost=transaction_cost)
    top50 = build_top_n_returns(merged, top_n=50, transaction_cost=transaction_cost)

    equal_weight = build_equal_weight_returns(merged)
    spy = build_spy_returns(realized)

    curves = top5[["date", "net_monthly_return", "selected_tickers"]].rename(
        columns={
            "net_monthly_return": "top5_return",
            "selected_tickers": "top5_selected_tickers",
        }
    )

    curves = curves.merge(
        top10[["date", "net_monthly_return", "selected_tickers"]].rename(
            columns={
                "net_monthly_return": "top10_return",
                "selected_tickers": "top10_selected_tickers",
            }
        ),
        on="date",
        how="left",
    )

    curves = curves.merge(
        top25[["date", "net_monthly_return", "selected_tickers"]].rename(
            columns={
                "net_monthly_return": "top25_return",
                "selected_tickers": "top25_selected_tickers",
            }
        ),
        on="date",
        how="left",
    )

    curves = curves.merge(
        top50[["date", "net_monthly_return", "selected_tickers"]].rename(
            columns={
                "net_monthly_return": "top50_return",
                "selected_tickers": "top50_selected_tickers",
            }
        ),
        on="date",
        how="left",
    )

    curves = curves.merge(equal_weight, on="date", how="left")
    curves = curves.merge(spy, on="date", how="left")

    curves = curves.sort_values("date").reset_index(drop=True)

    curves = add_cumulative(curves, "top5_return", "top5_cumulative")
    curves = add_cumulative(curves, "top10_return", "top10_cumulative")
    curves = add_cumulative(curves, "top25_return", "top25_cumulative")
    curves = add_cumulative(curves, "top50_return", "top50_cumulative")
    curves = add_cumulative(curves, "equal_weight_monthly_return", "equal_weight_cumulative")
    curves = add_cumulative(curves, "spy_monthly_return", "spy_cumulative")

    stats_rows = [
        {"strategy": "top5_net", **performance_stats(curves["top5_return"])},
        {"strategy": "top10_net", **performance_stats(curves["top10_return"])},
        {"strategy": "top25_net", **performance_stats(curves["top25_return"])},
        {"strategy": "top50_net", **performance_stats(curves["top50_return"])},
        {"strategy": "equal_weight_universe", **performance_stats(curves["equal_weight_monthly_return"])},
        {"strategy": "spy", **performance_stats(curves["spy_monthly_return"])},
    ]

    stats = pd.DataFrame(stats_rows)

    curves_path = "outputs/tables/week11_realized_backtest_curves.csv"
    stats_path = "outputs/tables/week11_realized_backtest_stats.csv"
    merged_path = "outputs/tables/week11_predictions_with_realized_returns.csv"
    figure_path = "outputs/figures/week11_realized_backtest_equity_curves.png"
    report_path = "outputs/reports/week11_realized_backtest_summary.txt"

    curves.to_csv(curves_path, index=False)
    stats.to_csv(stats_path, index=False)
    merged.to_csv(merged_path, index=False)

    plot_curves(curves, figure_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 11 True Realized-Return Backtest Summary\n")
        f.write("============================================\n\n")
        f.write("Model:\n")
        f.write("Expanded autoencoder latent_dim=20 + gradient boosting\n\n")
        f.write("Backtest method:\n")
        f.write("At each month t, rank stocks by predicted probability of 12-month SPY outperformance, buy top N, hold for one month, and measure actual realized next-month return.\n\n")
        f.write(f"Transaction cost per rebalance: {transaction_cost:.3%}\n\n")
        f.write("Performance stats:\n")
        f.write(stats.to_string(index=False))
        f.write("\n\n")
        f.write("Important interpretation:\n")
        f.write("This is the first more realistic backtest because it uses actual next-month realized returns rather than converted 12-month forward returns.\n")

    print("")
    print("Saved:", curves_path)
    print("Saved:", stats_path)
    print("Saved:", merged_path)
    print("Saved:", figure_path)
    print("Saved:", report_path)
    print("")
    print("Backtest stats:")
    print(stats)


if __name__ == "__main__":
    main()