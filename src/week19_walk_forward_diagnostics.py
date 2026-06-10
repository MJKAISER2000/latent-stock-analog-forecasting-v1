import os
import pandas as pd
import numpy as np


TOP_NS = [5, 10, 20]


def performance_stats(monthly_returns: pd.Series) -> dict:
    monthly_returns = monthly_returns.dropna()

    if len(monthly_returns) == 0:
        return {
            "months": 0,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_no_risk_free": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
            "return_over_abs_drawdown": np.nan,
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

    return_over_abs_drawdown = np.nan
    if max_drawdown != 0 and not pd.isna(max_drawdown):
        return_over_abs_drawdown = annualized_return / abs(max_drawdown)

    return {
        "months": len(monthly_returns),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_no_risk_free": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "return_over_abs_drawdown": return_over_abs_drawdown,
    }


def explode_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in holdings.iterrows():
        tickers = str(row["selected_tickers"]).split(",")

        for ticker in tickers:
            ticker = ticker.strip().upper()

            if ticker:
                rows.append(
                    {
                        "strategy": row["strategy"],
                        "signal_date": row["signal_date"],
                        "ticker": ticker,
                    }
                )

    return pd.DataFrame(rows)


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    curves_path = "outputs/tables/week19_walk_forward_ranker_curves.csv"
    holdings_path = "outputs/tables/week19_walk_forward_ranker_holdings.csv"
    predictions_path = "outputs/tables/week19_walk_forward_ranker_predictions.csv"

    curves = pd.read_csv(curves_path)
    holdings = pd.read_csv(holdings_path)
    predictions = pd.read_csv(predictions_path)

    curves["date"] = pd.to_datetime(curves["date"])
    holdings["signal_date"] = pd.to_datetime(holdings["signal_date"])
    predictions["date"] = pd.to_datetime(predictions["date"])

    curves["year"] = curves["date"].dt.year

    yearly_rows = []

    for top_n in TOP_NS:
        return_col = f"top{top_n}_return"

        if return_col not in curves.columns:
            continue

        for year, group in curves.groupby("year"):
            stats = performance_stats(group[return_col])

            yearly_rows.append(
                {
                    "top_n": top_n,
                    "year": year,
                    **stats,
                }
            )

    yearly_stats = pd.DataFrame(yearly_rows)

    monthly_rows = []

    for top_n in TOP_NS:
        return_col = f"top{top_n}_return"

        if return_col not in curves.columns:
            continue

        temp = curves[["date", "year", return_col]].copy()
        temp = temp.rename(columns={return_col: "monthly_return"})
        temp["top_n"] = top_n
        monthly_rows.append(temp)

    monthly_returns = pd.concat(monthly_rows, ignore_index=True)
    best_months = monthly_returns.sort_values("monthly_return", ascending=False).head(20)
    worst_months = monthly_returns.sort_values("monthly_return", ascending=True).head(20)

    # Analyze selected ticker frequency by top-N strategy.
    exploded = explode_holdings(holdings)

    ticker_freq = (
        exploded.groupby(["strategy", "ticker"])
        .agg(
            times_selected=("ticker", "count"),
            first_selected=("signal_date", "min"),
            last_selected=("signal_date", "max"),
        )
        .reset_index()
        .sort_values(["strategy", "times_selected"], ascending=[True, False])
    )

    # Rank quality by year: average future return of top deciles.
    predictions["year"] = predictions["date"].dt.year

    rank_quality_rows = []

    for year, group_year in predictions.groupby("year"):
        for cutoff in [5, 10, 20, 50]:
            selected = group_year[group_year["rank_by_date"] <= cutoff]

            rank_quality_rows.append(
                {
                    "year": year,
                    "rank_cutoff": cutoff,
                    "avg_future_1m_return": selected["future_1m_return"].mean(),
                    "avg_future_1m_excess_return": selected["future_1m_excess_return"].mean(),
                    "avg_spy_future_1m_return": selected["future_1m_spy_return"].mean(),
                    "selected_count": len(selected),
                }
            )

    rank_quality = pd.DataFrame(rank_quality_rows)

    yearly_path = "outputs/tables/week19_walk_forward_yearly_stats.csv"
    monthly_path = "outputs/tables/week19_walk_forward_monthly_returns_long.csv"
    ticker_path = "outputs/tables/week19_walk_forward_ticker_frequency.csv"
    rank_quality_path = "outputs/tables/week19_walk_forward_rank_quality_by_year.csv"
    report_path = "outputs/reports/week19_walk_forward_diagnostics_summary.txt"

    yearly_stats.to_csv(yearly_path, index=False)
    monthly_returns.to_csv(monthly_path, index=False)
    ticker_freq.to_csv(ticker_path, index=False)
    rank_quality.to_csv(rank_quality_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 19 Walk-Forward Diagnostics Summary\n")
        f.write("=======================================\n\n")

        f.write("Year-by-year performance:\n")
        f.write(yearly_stats.to_string(index=False))
        f.write("\n\n")

        f.write("Rank quality by year:\n")
        f.write(rank_quality.to_string(index=False))
        f.write("\n\n")

        f.write("Best months:\n")
        f.write(best_months.to_string(index=False))
        f.write("\n\n")

        f.write("Worst months:\n")
        f.write(worst_months.to_string(index=False))
        f.write("\n\n")

        f.write("Top selected tickers by strategy:\n")
        for strategy, group in ticker_freq.groupby("strategy"):
            f.write(f"\n{strategy}\n")
            f.write(group.head(20).to_string(index=False))
            f.write("\n")

    print("")
    print("Saved:", yearly_path)
    print("Saved:", monthly_path)
    print("Saved:", ticker_path)
    print("Saved:", rank_quality_path)
    print("Saved:", report_path)

    print("")
    print("YEAR-BY-YEAR WALK-FORWARD STATS")
    print(yearly_stats.sort_values(["top_n", "year"]).to_string(index=False))

    print("")
    print("RANK QUALITY BY YEAR")
    print(rank_quality.sort_values(["year", "rank_cutoff"]).to_string(index=False))

    print("")
    print("BEST MONTHS")
    print(best_months.to_string(index=False))

    print("")
    print("WORST MONTHS")
    print(worst_months.to_string(index=False))

    print("")
    print("TOP SELECTED TICKERS")
    for strategy, group in ticker_freq.groupby("strategy"):
        print("")
        print("=" * 80)
        print(strategy)
        print(group.head(20).to_string(index=False))


if __name__ == "__main__":
    main()