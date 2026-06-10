import os
import pandas as pd
import numpy as np


TARGET_STRATEGY = "week15_full500_lgbm_ranker_h1_top5_inverse_vol"


def load_monthly_returns() -> pd.DataFrame:
    prices_path = "data/processed/week15_500_monthly_prices.parquet"

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
                    "ticker": str(ticker).strip().upper(),
                    "monthly_return": ret,
                }
            )

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year

    return out


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

    universe_path = "data/external/week15_500_stock_universe.csv"
    holdings_path = "outputs/tables/week17_lgbm_ranker_backtest_holdings.csv"

    universe = pd.read_csv(universe_path)
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()

    holdings = pd.read_csv(holdings_path)
    holdings["signal_date"] = pd.to_datetime(holdings["signal_date"])

    monthly_returns = load_monthly_returns()

    # Add sector metadata to all returns.
    ret_meta = monthly_returns.merge(
        universe[["ticker", "company", "sector", "industry"]],
        on="ticker",
        how="left",
    )

    ret_meta["sector"] = ret_meta["sector"].fillna("Unknown")
    ret_meta["industry"] = ret_meta["industry"].fillna("Unknown")

    # Sector returns by year.
    # Sector returns by year.
# First average stock returns within each sector-month,
# then compound those monthly sector returns within each year.
    sector_month_returns = (
    ret_meta.groupby(["year", "date", "sector"])["monthly_return"]
    .mean()
    .reset_index()
    .rename(columns={"monthly_return": "sector_monthly_return"})
)

    sector_year_returns = (
    sector_month_returns.groupby(["year", "sector"])["sector_monthly_return"]
    .apply(lambda x: (1 + x).prod() - 1)
    .reset_index()
    .rename(columns={"sector_monthly_return": "sector_total_return"})
)

    sector_year_returns["sector_rank_in_year"] = sector_year_returns.groupby("year")[
        "sector_total_return"
    ].rank(ascending=False, method="first")

    sector_year_returns = sector_year_returns.sort_values(
        ["year", "sector_rank_in_year"]
    )

    # Ranker selected sectors by year.
    target_holdings = holdings[holdings["strategy"] == TARGET_STRATEGY].copy()
    exploded = explode_holdings(target_holdings)
    exploded["signal_date"] = pd.to_datetime(exploded["signal_date"])
    exploded["year"] = exploded["signal_date"].dt.year

    selected_meta = exploded.merge(
        universe[["ticker", "company", "sector", "industry"]],
        on="ticker",
        how="left",
    )

    selected_meta["sector"] = selected_meta["sector"].fillna("Unknown")
    selected_meta["industry"] = selected_meta["industry"].fillna("Unknown")

    selected_sector_counts = (
        selected_meta.groupby(["year", "sector"])
        .agg(
            selected_count=("ticker", "count"),
            unique_tickers=("ticker", "nunique"),
        )
        .reset_index()
    )

    selected_sector_counts["selection_share"] = selected_sector_counts.groupby("year")[
        "selected_count"
    ].transform(lambda x: x / x.sum())

    selected_sector_counts = selected_sector_counts.sort_values(
        ["year", "selected_count"], ascending=[True, False]
    )

    # Combine selected sectors with their return rank that year.
    selected_vs_leaders = selected_sector_counts.merge(
        sector_year_returns,
        on=["year", "sector"],
        how="left",
    )

    selected_vs_leaders = selected_vs_leaders.sort_values(
        ["year", "selected_count"], ascending=[True, False]
    )

    # Top industries selected by year.
    selected_industry_counts = (
        selected_meta.groupby(["year", "industry"])
        .agg(
            selected_count=("ticker", "count"),
            unique_tickers=("ticker", "nunique"),
        )
        .reset_index()
    )

    selected_industry_counts["selection_share"] = selected_industry_counts.groupby("year")[
        "selected_count"
    ].transform(lambda x: x / x.sum())

    selected_industry_counts = selected_industry_counts.sort_values(
        ["year", "selected_count"], ascending=[True, False]
    )

    # Save outputs.
    sector_returns_path = "outputs/tables/week17_sector_regime_yearly_sector_returns.csv"
    selected_sector_path = "outputs/tables/week17_sector_regime_selected_sectors.csv"
    selected_vs_leaders_path = "outputs/tables/week17_sector_regime_selected_vs_leaders.csv"
    selected_industry_path = "outputs/tables/week17_sector_regime_selected_industries.csv"
    report_path = "outputs/reports/week17_sector_regime_diagnostic_summary.txt"

    sector_year_returns.to_csv(sector_returns_path, index=False)
    selected_sector_counts.to_csv(selected_sector_path, index=False)
    selected_vs_leaders.to_csv(selected_vs_leaders_path, index=False)
    selected_industry_counts.to_csv(selected_industry_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 17 Sector Regime Diagnostic\n")
        f.write("================================\n\n")
        f.write(f"Target strategy: {TARGET_STRATEGY}\n\n")

        f.write("Top yearly sector returns:\n")
        for year, group in sector_year_returns.groupby("year"):
            f.write(f"\nYear {year}\n")
            f.write(
                group[["sector", "sector_total_return", "sector_rank_in_year"]]
                .head(8)
                .to_string(index=False)
            )
            f.write("\n")

        f.write("\n\nRanker selected sectors by year:\n")
        for year, group in selected_sector_counts.groupby("year"):
            f.write(f"\nYear {year}\n")
            f.write(
                group[["sector", "selected_count", "unique_tickers", "selection_share"]]
                .head(8)
                .to_string(index=False)
            )
            f.write("\n")

        f.write("\n\nSelected sectors with sector return ranks:\n")
        for year, group in selected_vs_leaders.groupby("year"):
            f.write(f"\nYear {year}\n")
            f.write(
                group[
                    [
                        "sector",
                        "selected_count",
                        "selection_share",
                        "sector_total_return",
                        "sector_rank_in_year",
                    ]
                ]
                .head(8)
                .to_string(index=False)
            )
            f.write("\n")

    print("")
    print("Saved:", sector_returns_path)
    print("Saved:", selected_sector_path)
    print("Saved:", selected_vs_leaders_path)
    print("Saved:", selected_industry_path)
    print("Saved:", report_path)

    print("")
    print("SELECTED SECTORS VS YEARLY SECTOR LEADERS")
    for year, group in selected_vs_leaders.groupby("year"):
        print("")
        print("=" * 80)
        print("YEAR", year)
        print(
            group[
                [
                    "sector",
                    "selected_count",
                    "selection_share",
                    "sector_total_return",
                    "sector_rank_in_year",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print("")
    print("TOP SECTOR RETURNS BY YEAR")
    for year, group in sector_year_returns.groupby("year"):
        print("")
        print("=" * 80)
        print("YEAR", year)
        print(
            group[["sector", "sector_total_return", "sector_rank_in_year"]]
            .head(8)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()