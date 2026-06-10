import os
import pandas as pd
import numpy as np


def compute_volatility_table(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    monthly_returns = monthly_prices / monthly_prices.shift(1) - 1

    rows = []

    for ticker in monthly_returns.columns:
        if ticker == "SPY":
            continue

        r = monthly_returns[ticker].dropna()

        if len(r) < 36:
            continue

        trailing_12m_vol = r.tail(12).std() * np.sqrt(12)
        full_sample_vol = r.std() * np.sqrt(12)
        avg_abs_monthly_return = r.abs().mean()
        total_return = monthly_prices[ticker].dropna().iloc[-1] / monthly_prices[ticker].dropna().iloc[0] - 1

        cumulative = (1 + r).cumprod()
        running_max = cumulative.cummax()
        max_drawdown = (cumulative / running_max - 1).min()

        rows.append(
            {
                "ticker": ticker,
                "monthly_obs": len(r),
                "trailing_12m_vol": trailing_12m_vol,
                "full_sample_vol": full_sample_vol,
                "avg_abs_monthly_return": avg_abs_monthly_return,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
            }
        )

    vol = pd.DataFrame(rows)
    vol = vol.sort_values("full_sample_vol", ascending=False).reset_index(drop=True)
    vol["vol_rank"] = np.arange(1, len(vol) + 1)

    return vol


def main():
    os.makedirs("data/external", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    universe_path = "data/external/week15_500_stock_universe.csv"
    prices_path = "data/processed/week15_500_monthly_prices.parquet"

    vol_table_path = "outputs/tables/week15_500_volatility_rankings.csv"
    high100_path = "data/external/week15_high_vol_100_universe.csv"
    high200_path = "data/external/week15_high_vol_200_universe.csv"
    report_path = "outputs/reports/week15_volatility_subset_summary.txt"

    print("Loading 500-stock universe...")
    universe = pd.read_csv(universe_path)
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()

    print("Loading monthly prices...")
    prices = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index)

    print("Computing volatility rankings...")
    vol = compute_volatility_table(prices)

    enriched = vol.merge(universe, on="ticker", how="left")

    high100 = enriched.head(100).copy()
    high200 = enriched.head(200).copy()

    enriched.to_csv(vol_table_path, index=False)
    high100.to_csv(high100_path, index=False)
    high200.to_csv(high200_path, index=False)

    lines = []
    lines.append("Week 15 Volatility Subset Summary")
    lines.append("=================================")
    lines.append("")
    lines.append("Goal:")
    lines.append("Rank the broad 500-stock universe by realized volatility and create high-volatility subsets.")
    lines.append("")
    lines.append(f"Full ranked volatility table shape: {enriched.shape}")
    lines.append(f"High-vol 100 shape: {high100.shape}")
    lines.append(f"High-vol 200 shape: {high200.shape}")
    lines.append("")
    lines.append("Top 30 most volatile tickers:")
    lines.append(enriched[["vol_rank", "ticker", "company", "sector", "full_sample_vol", "max_drawdown"]].head(30).to_string(index=False))
    lines.append("")
    lines.append("High-vol 100 sector counts:")
    lines.append(str(high100["sector"].value_counts()))
    lines.append("")
    lines.append("High-vol 200 sector counts:")
    lines.append(str(high200["sector"].value_counts()))
    lines.append("")
    lines.append("Output files:")
    lines.append(vol_table_path)
    lines.append(high100_path)
    lines.append(high200_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("")
    print("Saved:", vol_table_path)
    print("Saved:", high100_path)
    print("Saved:", high200_path)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()