import os
import pandas as pd
import numpy as np


TARGET_STRATEGY = "week15_full500_lgbm_ranker_h1_top5_inverse_vol"


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


def explode_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in holdings.iterrows():
        tickers = str(row["selected_tickers"]).split(",")

        for ticker in tickers:
            ticker = ticker.strip()

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

    curves_path = "outputs/tables/week17_lgbm_ranker_backtest_curves.csv"
    holdings_path = "outputs/tables/week17_lgbm_ranker_backtest_holdings.csv"

    curves = pd.read_csv(curves_path)
    holdings = pd.read_csv(holdings_path)

    curves["date"] = pd.to_datetime(curves["date"])
    holdings["signal_date"] = pd.to_datetime(holdings["signal_date"])

    return_col = f"{TARGET_STRATEGY}_return"
    cumulative_col = f"{TARGET_STRATEGY}_cumulative"

    if return_col not in curves.columns:
        raise ValueError(f"Missing return column: {return_col}")

    strategy_returns = curves[["date", return_col, cumulative_col]].copy()
    strategy_returns = strategy_returns.rename(
        columns={
            return_col: "monthly_return",
            cumulative_col: "cumulative_growth",
        }
    )

    strategy_returns["year"] = strategy_returns["date"].dt.year
    strategy_returns["month"] = strategy_returns["date"].dt.to_period("M").astype(str)

    year_stats = []

    for year, group in strategy_returns.groupby("year"):
        stats = performance_stats(group["monthly_return"])

        year_stats.append(
            {
                "year": year,
                "months": len(group),
                **stats,
            }
        )

    year_stats = pd.DataFrame(year_stats)

    monthly_summary = strategy_returns.copy()
    monthly_summary["rank_best"] = monthly_summary["monthly_return"].rank(ascending=False)
    monthly_summary["rank_worst"] = monthly_summary["monthly_return"].rank(ascending=True)

    best_months = monthly_summary.sort_values("monthly_return", ascending=False).head(15)
    worst_months = monthly_summary.sort_values("monthly_return", ascending=True).head(15)

    target_holdings = holdings[holdings["strategy"] == TARGET_STRATEGY].copy()

    exploded = explode_holdings(target_holdings)

    ticker_counts = (
        exploded.groupby("ticker")
        .agg(
            times_selected=("ticker", "count"),
            first_selected=("signal_date", "min"),
            last_selected=("signal_date", "max"),
        )
        .reset_index()
        .sort_values("times_selected", ascending=False)
    )

    # Approx concentration statistics
    total_signal_months = target_holdings["signal_date"].nunique()
    ticker_counts["selection_rate"] = ticker_counts["times_selected"] / total_signal_months

    # Check last 18 months contribution
    cutoff = strategy_returns["date"].max() - pd.DateOffset(months=18)

    recent = strategy_returns[strategy_returns["date"] >= cutoff]
    before_recent = strategy_returns[strategy_returns["date"] < cutoff]

    recent_stats = performance_stats(recent["monthly_return"])
    before_recent_stats = performance_stats(before_recent["monthly_return"])

    output_year_path = "outputs/tables/week17_ranker_diagnostics_year_stats.csv"
    output_months_path = "outputs/tables/week17_ranker_diagnostics_monthly_returns.csv"
    output_tickers_path = "outputs/tables/week17_ranker_diagnostics_ticker_frequency.csv"
    report_path = "outputs/reports/week17_ranker_diagnostics_summary.txt"

    year_stats.to_csv(output_year_path, index=False)
    monthly_summary.to_csv(output_months_path, index=False)
    ticker_counts.to_csv(output_tickers_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 17 Ranker Diagnostics Summary\n")
        f.write("==================================\n\n")
        f.write(f"Target strategy: {TARGET_STRATEGY}\n\n")

        f.write("Overall stats:\n")
        f.write(str(performance_stats(strategy_returns["monthly_return"])))
        f.write("\n\n")

        f.write("Year-by-year stats:\n")
        f.write(year_stats.to_string(index=False))
        f.write("\n\n")

        f.write("Best months:\n")
        f.write(best_months[["date", "monthly_return", "cumulative_growth"]].to_string(index=False))
        f.write("\n\n")

        f.write("Worst months:\n")
        f.write(worst_months[["date", "monthly_return", "cumulative_growth"]].to_string(index=False))
        f.write("\n\n")

        f.write("Top selected tickers:\n")
        f.write(ticker_counts.head(30).to_string(index=False))
        f.write("\n\n")

        f.write("Pre-recent-period stats:\n")
        f.write(str(before_recent_stats))
        f.write("\n\n")

        f.write("Recent-period stats:")
        f.write(str(recent_stats))
        f.write("\n")

    print("")
    print("Saved:", output_year_path)
    print("Saved:", output_months_path)
    print("Saved:", output_tickers_path)
    print("Saved:", report_path)

    print("")
    print("YEAR STATS")
    print(year_stats.to_string(index=False))

    print("")
    print("TOP SELECTED TICKERS")
    print(ticker_counts.head(30).to_string(index=False))

    print("")
    print("BEST MONTHS")
    print(best_months[["date", "monthly_return", "cumulative_growth"]].to_string(index=False))

    print("")
    print("WORST MONTHS")
    print(worst_months[["date", "monthly_return", "cumulative_growth"]].to_string(index=False))

    print("")
    print("BEFORE RECENT PERIOD")
    print(before_recent_stats)

    print("")
    print("RECENT PERIOD")
    print(recent_stats)


if __name__ == "__main__":
    main()