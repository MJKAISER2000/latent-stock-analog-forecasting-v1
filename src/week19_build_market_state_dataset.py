import os
import pandas as pd
import numpy as np


PRICE_PATH = "data/processed/week15_500_monthly_prices.parquet"
UNIVERSE_PATH = "data/external/week15_500_stock_universe.csv"

OUTPUT_PATH = "data/processed/week19_market_state_dataset.parquet"
REPORT_PATH = "outputs/reports/week19_market_state_dataset_summary.txt"


MOM_WINDOWS = [1, 3, 6, 12]


def trailing_return(series: pd.Series, months: int) -> pd.Series:
    return series / series.shift(months) - 1


def rolling_vol(returns: pd.Series, months: int) -> pd.Series:
    return returns.rolling(months).std() * np.sqrt(12)


def drawdown_from_index(index: pd.Series) -> pd.Series:
    return index / index.cummax() - 1


def safe_mean(df: pd.DataFrame) -> pd.Series:
    return df.mean(axis=1, skipna=True)


def safe_std(df: pd.DataFrame) -> pd.Series:
    return df.std(axis=1, skipna=True)


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    print("Loading prices...")
    prices = pd.read_parquet(PRICE_PATH)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    print("Loading universe metadata...")
    universe = pd.read_csv(UNIVERSE_PATH)
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()
    universe["sector"] = universe["sector"].fillna("Unknown").astype(str)

    prices.columns = [str(c).strip().upper() for c in prices.columns]

    monthly_returns = prices / prices.shift(1) - 1

    if "SPY" not in prices.columns:
        raise ValueError("SPY is missing from price data.")

    stock_cols = [c for c in prices.columns if c != "SPY"]

    market = pd.DataFrame(index=prices.index)
    market["date"] = market.index

    # SPY state.
    spy_price = prices["SPY"]
    spy_ret = monthly_returns["SPY"]
    spy_index = (1 + spy_ret.fillna(0)).cumprod()

    market["spy_ret_1m"] = spy_ret
    market["spy_drawdown"] = drawdown_from_index(spy_index)

    for w in MOM_WINDOWS:
        market[f"spy_ret_{w}m"] = trailing_return(spy_price, w)
        market[f"spy_vol_{w}m"] = rolling_vol(spy_ret, w)

    # Equal-weight market state.
    ew_ret = monthly_returns[stock_cols].mean(axis=1, skipna=True)
    ew_index = (1 + ew_ret.fillna(0)).cumprod()

    market["equal_weight_ret_1m"] = ew_ret
    market["equal_weight_drawdown"] = drawdown_from_index(ew_index)

    for w in MOM_WINDOWS:
        market[f"equal_weight_ret_{w}m"] = (1 + ew_ret).rolling(w).apply(np.prod, raw=True) - 1
        market[f"equal_weight_vol_{w}m"] = rolling_vol(ew_ret, w)
        market[f"equal_weight_minus_spy_{w}m"] = market[f"equal_weight_ret_{w}m"] - market[f"spy_ret_{w}m"]

    # Cross-sectional market state.
    for w in MOM_WINDOWS:
        stock_trailing = prices[stock_cols] / prices[stock_cols].shift(w) - 1

        market[f"avg_stock_ret_{w}m"] = safe_mean(stock_trailing)
        market[f"median_stock_ret_{w}m"] = stock_trailing.median(axis=1, skipna=True)
        market[f"cross_sectional_ret_std_{w}m"] = safe_std(stock_trailing)
        market[f"pct_stocks_positive_ret_{w}m"] = (stock_trailing > 0).mean(axis=1)

    # Stock-level volatility and drawdown aggregates.
    stock_vol_12m = monthly_returns[stock_cols].rolling(12).std() * np.sqrt(12)
    market["avg_stock_vol_12m"] = stock_vol_12m.mean(axis=1, skipna=True)
    market["median_stock_vol_12m"] = stock_vol_12m.median(axis=1, skipna=True)
    market["cross_sectional_vol_std_12m"] = stock_vol_12m.std(axis=1, skipna=True)

    stock_index = (1 + monthly_returns[stock_cols].fillna(0)).cumprod()
    stock_drawdown = stock_index / stock_index.cummax() - 1

    market["avg_stock_drawdown"] = stock_drawdown.mean(axis=1, skipna=True)
    market["median_stock_drawdown"] = stock_drawdown.median(axis=1, skipna=True)
    market["pct_stocks_drawdown_worse_20"] = (stock_drawdown < -0.20).mean(axis=1)
    market["pct_stocks_drawdown_worse_40"] = (stock_drawdown < -0.40).mean(axis=1)

    # Breadth versus moving averages.
    for w in [3, 6, 12]:
        ma = prices[stock_cols].rolling(w).mean()
        market[f"pct_stocks_above_ma_{w}m"] = (prices[stock_cols] > ma).mean(axis=1)

    # Sector returns and sector dispersion.
    available_meta = universe[universe["ticker"].isin(stock_cols)].copy()
    sectors = sorted(available_meta["sector"].dropna().unique().tolist())

    sector_return_cols = []

    for sector in sectors:
        tickers = available_meta.loc[available_meta["sector"] == sector, "ticker"].tolist()
        tickers = [t for t in tickers if t in monthly_returns.columns]

        if len(tickers) == 0:
            continue

        safe_sector_name = (
            sector.lower()
            .replace(" ", "_")
            .replace("&", "and")
            .replace("/", "_")
            .replace("-", "_")
        )

        sector_ret = monthly_returns[tickers].mean(axis=1, skipna=True)
        sector_index = (1 + sector_ret.fillna(0)).cumprod()

        market[f"sector_{safe_sector_name}_ret_1m"] = sector_ret
        market[f"sector_{safe_sector_name}_drawdown"] = drawdown_from_index(sector_index)

        for w in [3, 6, 12]:
            col = f"sector_{safe_sector_name}_ret_{w}m"
            market[col] = (1 + sector_ret).rolling(w).apply(np.prod, raw=True) - 1
            market[f"sector_{safe_sector_name}_minus_spy_{w}m"] = market[col] - market[f"spy_ret_{w}m"]

        sector_return_cols.append(f"sector_{safe_sector_name}_ret_1m")

    if len(sector_return_cols) > 0:
        sector_ret_frame = market[sector_return_cols]
        market["sector_return_dispersion_1m"] = sector_ret_frame.std(axis=1, skipna=True)
        market["best_sector_ret_1m"] = sector_ret_frame.max(axis=1, skipna=True)
        market["worst_sector_ret_1m"] = sector_ret_frame.min(axis=1, skipna=True)
        market["best_minus_worst_sector_ret_1m"] = (
            market["best_sector_ret_1m"] - market["worst_sector_ret_1m"]
        )

    # Clean.
    market = market.replace([np.inf, -np.inf], np.nan)
    market = market.sort_values("date").reset_index(drop=True)

    # Fill missing with expanding/median fallback.
    feature_cols = [c for c in market.columns if c != "date"]

    for col in feature_cols:
        market[col] = pd.to_numeric(market[col], errors="coerce")
        market[col] = market[col].ffill()
        market[col] = market[col].fillna(market[col].median())

    market.to_parquet(OUTPUT_PATH)

    lines = []
    lines.append("Week 19 Market-State Dataset Summary")
    lines.append("====================================")
    lines.append("")
    lines.append("Goal:")
    lines.append("Build one row per month describing the full market state for latent market twin modeling.")
    lines.append("")
    lines.append(f"Output shape: {market.shape}")
    lines.append(f"Date range: {market['date'].min()} to {market['date'].max()}")
    lines.append(f"Feature count: {len(feature_cols)}")
    lines.append("")
    lines.append("Example columns:")
    lines.append(", ".join(market.columns[:80]))
    lines.append("")
    lines.append(f"Saved to: {OUTPUT_PATH}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("")
    print("Saved:", OUTPUT_PATH)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()