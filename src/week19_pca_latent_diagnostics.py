import os
import pandas as pd
import numpy as np


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

    nonlatent_curves_path = "outputs/tables/week19_walk_forward_ranker_curves.csv"
    pca_curves_path = "outputs/tables/week19_walk_forward_pca_latent_ranker_curves.csv"
    pca_regime_curves_path = "outputs/tables/week19_pca_latent_regime_filter_curves.csv"

    nonlatent_holdings_path = "outputs/tables/week19_walk_forward_ranker_holdings.csv"
    pca_holdings_path = "outputs/tables/week19_walk_forward_pca_latent_ranker_holdings.csv"

    nonlatent = pd.read_csv(nonlatent_curves_path)
    pca = pd.read_csv(pca_curves_path)
    pca_regime = pd.read_csv(pca_regime_curves_path)

    nonlatent["date"] = pd.to_datetime(nonlatent["date"])
    pca["date"] = pd.to_datetime(pca["date"])
    pca_regime["date"] = pd.to_datetime(pca_regime["date"])

    # Best regime-filtered PCA model from current results.
    regime_col = "top20_tech_drawdown_20_100cash_return"

    if regime_col not in pca_regime.columns:
        raise ValueError(f"Missing regime return column: {regime_col}")

    comparison = nonlatent[["date", "top20_return"]].rename(
        columns={"top20_return": "nonlatent_top20_return"}
    )

    comparison = comparison.merge(
        pca[["date", "top20_return"]].rename(
            columns={"top20_return": "pca_latent_top20_return"}
        ),
        on="date",
        how="inner",
    )

    comparison = comparison.merge(
        pca_regime[["date", regime_col]].rename(
            columns={regime_col: "pca_latent_top20_regime_return"}
        ),
        on="date",
        how="inner",
    )

    comparison["year"] = comparison["date"].dt.year

    yearly_rows = []

    for strategy_col in [
        "nonlatent_top20_return",
        "pca_latent_top20_return",
        "pca_latent_top20_regime_return",
    ]:
        for year, group in comparison.groupby("year"):
            stats = performance_stats(group[strategy_col])

            yearly_rows.append(
                {
                    "strategy": strategy_col.replace("_return", ""),
                    "year": year,
                    **stats,
                }
            )

    yearly_stats = pd.DataFrame(yearly_rows)

    full_rows = []

    for strategy_col in [
        "nonlatent_top20_return",
        "pca_latent_top20_return",
        "pca_latent_top20_regime_return",
    ]:
        stats = performance_stats(comparison[strategy_col])

        full_rows.append(
            {
                "strategy": strategy_col.replace("_return", ""),
                **stats,
            }
        )

    full_stats = pd.DataFrame(full_rows)

    # Monthly best/worst comparison.
    monthly_long = []

    for strategy_col in [
        "nonlatent_top20_return",
        "pca_latent_top20_return",
        "pca_latent_top20_regime_return",
    ]:
        temp = comparison[["date", "year", strategy_col]].copy()
        temp = temp.rename(columns={strategy_col: "monthly_return"})
        temp["strategy"] = strategy_col.replace("_return", "")
        monthly_long.append(temp)

    monthly_long = pd.concat(monthly_long, ignore_index=True)

    best_months = monthly_long.sort_values("monthly_return", ascending=False).head(25)
    worst_months = monthly_long.sort_values("monthly_return", ascending=True).head(25)

    # Holdings diagnostics.
    nonlatent_holdings = pd.read_csv(nonlatent_holdings_path)
    pca_holdings = pd.read_csv(pca_holdings_path)

    nonlatent_holdings["signal_date"] = pd.to_datetime(nonlatent_holdings["signal_date"])
    pca_holdings["signal_date"] = pd.to_datetime(pca_holdings["signal_date"])

    nonlatent_top20_holdings = nonlatent_holdings[
        nonlatent_holdings["strategy"] == "walk_forward_top20"
    ].copy()

    pca_top20_holdings = pca_holdings[
        pca_holdings["strategy"] == "walk_forward_pca_latent_top20"
    ].copy()

    nonlatent_exploded = explode_holdings(nonlatent_top20_holdings)
    pca_exploded = explode_holdings(pca_top20_holdings)

    nonlatent_exploded["model"] = "nonlatent_top20"
    pca_exploded["model"] = "pca_latent_top20"

    holdings_exploded = pd.concat([nonlatent_exploded, pca_exploded], ignore_index=True)

    ticker_freq = (
        holdings_exploded.groupby(["model", "ticker"])
        .agg(
            times_selected=("ticker", "count"),
            first_selected=("signal_date", "min"),
            last_selected=("signal_date", "max"),
        )
        .reset_index()
        .sort_values(["model", "times_selected"], ascending=[True, False])
    )

    # Save.
    full_path = "outputs/tables/week19_pca_latent_diagnostics_full_stats.csv"
    yearly_path = "outputs/tables/week19_pca_latent_diagnostics_yearly_stats.csv"
    monthly_path = "outputs/tables/week19_pca_latent_diagnostics_monthly_returns.csv"
    ticker_path = "outputs/tables/week19_pca_latent_diagnostics_ticker_frequency.csv"
    report_path = "outputs/reports/week19_pca_latent_diagnostics_summary.txt"

    full_stats.to_csv(full_path, index=False)
    yearly_stats.to_csv(yearly_path, index=False)
    monthly_long.to_csv(monthly_path, index=False)
    ticker_freq.to_csv(ticker_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 19 PCA Latent Diagnostics Summary\n")
        f.write("=====================================\n\n")
        f.write("Goal:\n")
        f.write("Diagnose whether PCA market-state latent features improve walk-forward performance year by year.\n\n")

        f.write("Full-period stats:\n")
        f.write(full_stats.to_string(index=False))
        f.write("\n\n")

        f.write("Yearly stats:\n")
        f.write(yearly_stats.sort_values(["strategy", "year"]).to_string(index=False))
        f.write("\n\n")

        f.write("Worst months:\n")
        f.write(worst_months.to_string(index=False))
        f.write("\n\n")

        f.write("Best months:\n")
        f.write(best_months.to_string(index=False))
        f.write("\n\n")

        f.write("Top selected tickers:\n")
        for model, group in ticker_freq.groupby("model"):
            f.write(f"\n{model}\n")
            f.write(group.head(25).to_string(index=False))
            f.write("\n")

    print("")
    print("Saved:", full_path)
    print("Saved:", yearly_path)
    print("Saved:", monthly_path)
    print("Saved:", ticker_path)
    print("Saved:", report_path)

    print("")
    print("FULL-PERIOD STATS")
    print(full_stats.to_string(index=False))

    print("")
    print("YEARLY STATS")
    print(yearly_stats.sort_values(["strategy", "year"]).to_string(index=False))

    print("")
    print("WORST MONTHS")
    print(worst_months.to_string(index=False))

    print("")
    print("TOP SELECTED TICKERS")
    for model, group in ticker_freq.groupby("model"):
        print("")
        print("=" * 80)
        print(model)
        print(group.head(25).to_string(index=False))


if __name__ == "__main__":
    main()