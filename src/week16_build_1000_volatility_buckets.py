import os
import pandas as pd
import numpy as np


def compute_volatility_table(monthly_prices: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    monthly_returns = monthly_prices / monthly_prices.shift(1) - 1

    rows = []

    for ticker in monthly_returns.columns:
        if ticker == "SPY":
            continue

        r = monthly_returns[ticker].dropna()
        p = monthly_prices[ticker].dropna()

        if len(r) < 36 or len(p) < 37:
            continue

        trailing_12m_vol = r.tail(12).std() * np.sqrt(12)
        full_sample_vol = r.std() * np.sqrt(12)
        avg_abs_monthly_return = r.abs().mean()
        total_return = p.iloc[-1] / p.iloc[0] - 1

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

    universe = universe.copy()
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()

    vol = vol.merge(universe, on="ticker", how="left")

    for col in ["company", "exchange", "sector", "industry"]:
        if col not in vol.columns:
            vol[col] = "Unknown"
        vol[col] = vol[col].fillna("Unknown")

    vol = vol.sort_values("full_sample_vol", ascending=True).reset_index(drop=True)
    vol["vol_rank_low_to_high"] = np.arange(1, len(vol) + 1)
    vol["vol_rank_high_to_low"] = len(vol) - vol["vol_rank_low_to_high"] + 1

    return vol


def make_bucket_sets(vol: pd.DataFrame, bucket_size: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(vol)

    low = vol.head(bucket_size).copy()
    high = vol.tail(bucket_size).copy().sort_values("full_sample_vol", ascending=False).reset_index(drop=True)

    mid_start = max((n // 2) - (bucket_size // 2), 0)
    mid = vol.iloc[mid_start:mid_start + bucket_size].copy().reset_index(drop=True)

    low["vol_bucket"] = "low"
    mid["vol_bucket"] = "mid"
    high["vol_bucket"] = "high"

    balanced = pd.concat([low, mid, high], ignore_index=True)
    balanced = balanced.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    return low, mid, high, balanced


def save_universe(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


def main():
    os.makedirs("data/external", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    universe_path = "data/external/week16_1000_stock_universe.csv"
    prices_path = "data/processed/week16_1000_monthly_prices.parquet"

    vol_rankings_path = "outputs/tables/week16_1000_volatility_rankings.csv"

    low300_path = "data/external/week16_lowvol300_universe.csv"
    mid300_path = "data/external/week16_midvol300_universe.csv"
    high300_path = "data/external/week16_highvol300_universe.csv"
    balanced900_path = "data/external/week16_balanced900_universe.csv"

    low150_path = "data/external/week16_lowvol150_universe.csv"
    mid150_path = "data/external/week16_midvol150_universe.csv"
    high150_path = "data/external/week16_highvol150_universe.csv"
    balanced450_path = "data/external/week16_balanced450_universe.csv"

    report_path = "outputs/reports/week16_1000_volatility_bucket_summary.txt"

    print("Loading 1000+ universe...")
    universe = pd.read_csv(universe_path)

    print("Loading 1000+ monthly prices...")
    prices = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index)

    print("Computing volatility rankings...")
    vol = compute_volatility_table(prices, universe)
    vol.to_csv(vol_rankings_path, index=False)

    print("Creating 300-stock buckets...")
    low300, mid300, high300, balanced900 = make_bucket_sets(vol, bucket_size=300)

    print("Creating 150-stock buckets...")
    low150, mid150, high150, balanced450 = make_bucket_sets(vol, bucket_size=150)

    save_universe(low300, low300_path)
    save_universe(mid300, mid300_path)
    save_universe(high300, high300_path)
    save_universe(balanced900, balanced900_path)

    save_universe(low150, low150_path)
    save_universe(mid150, mid150_path)
    save_universe(high150, high150_path)
    save_universe(balanced450, balanced450_path)

    lines = []
    lines.append("Week 16 1000+ Volatility Bucket Summary")
    lines.append("=======================================")
    lines.append("")
    lines.append("Goal:")
    lines.append("Create large low-volatility, mid-volatility, high-volatility, and balanced universes from the 1000+ ticker base.")
    lines.append("")
    lines.append(f"Usable volatility-ranked ticker count: {len(vol)}")
    lines.append("")
    lines.append("Created universes:")
    lines.append(f"lowvol300: {low300.shape}")
    lines.append(f"midvol300: {mid300.shape}")
    lines.append(f"highvol300: {high300.shape}")
    lines.append(f"balanced900: {balanced900.shape}")
    lines.append(f"lowvol150: {low150.shape}")
    lines.append(f"midvol150: {mid150.shape}")
    lines.append(f"highvol150: {high150.shape}")
    lines.append(f"balanced450: {balanced450.shape}")
    lines.append("")
    lines.append("Balanced900 bucket counts:")
    lines.append(str(balanced900["vol_bucket"].value_counts()))
    lines.append("")
    lines.append("Balanced450 bucket counts:")
    lines.append(str(balanced450["vol_bucket"].value_counts()))
    lines.append("")
    lines.append("Lowest-volatility sample:")
    lines.append(low300[["ticker", "company", "exchange", "full_sample_vol", "max_drawdown"]].head(25).to_string(index=False))
    lines.append("")
    lines.append("Middle-volatility sample:")
    lines.append(mid300[["ticker", "company", "exchange", "full_sample_vol", "max_drawdown"]].head(25).to_string(index=False))
    lines.append("")
    lines.append("Highest-volatility sample:")
    lines.append(high300[["ticker", "company", "exchange", "full_sample_vol", "max_drawdown"]].head(25).to_string(index=False))
    lines.append("")
    lines.append("Output files:")
    lines.append(vol_rankings_path)
    lines.append(low300_path)
    lines.append(mid300_path)
    lines.append(high300_path)
    lines.append(balanced900_path)
    lines.append(low150_path)
    lines.append(mid150_path)
    lines.append(high150_path)
    lines.append(balanced450_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("")
    print("Saved:", vol_rankings_path)
    print("Saved:", balanced900_path)
    print("Saved:", balanced450_path)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()