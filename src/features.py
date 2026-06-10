import os
import yaml
import pandas as pd
import numpy as np


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes trailing return features from monthly prices.
    """
    features = []

    horizons = {
        "ret_1m": 1,
        "ret_3m": 3,
        "ret_6m": 6,
        "ret_12m": 12,
    }

    for name, months in horizons.items():
        ret = prices / prices.shift(months) - 1
        ret = ret.stack().reset_index()
        ret.columns = ["date", "ticker", name]
        features.append(ret)

    out = features[0]
    for f in features[1:]:
        out = out.merge(f, on=["date", "ticker"], how="outer")

    return out


def compute_moving_average_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes price divided by moving average features.
    Uses monthly moving averages for now.
    """
    features = []

    windows = {
        "price_to_ma_3m": 3,
        "price_to_ma_6m": 6,
        "price_to_ma_12m": 12,
    }

    for name, window in windows.items():
        ma = prices.rolling(window=window).mean()
        ratio = prices / ma
        ratio = ratio.stack().reset_index()
        ratio.columns = ["date", "ticker", name]
        features.append(ratio)

    out = features[0]
    for f in features[1:]:
        out = out.merge(f, on=["date", "ticker"], how="outer")

    return out


def compute_volatility_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes trailing volatility using monthly returns.
    """
    monthly_returns = prices.pct_change()

    features = []

    windows = {
        "vol_3m": 3,
        "vol_6m": 6,
        "vol_12m": 12,
    }

    for name, window in windows.items():
        vol = monthly_returns.rolling(window=window).std()
        vol = vol.stack().reset_index()
        vol.columns = ["date", "ticker", name]
        features.append(vol)

    out = features[0]
    for f in features[1:]:
        out = out.merge(f, on=["date", "ticker"], how="outer")

    return out


def compute_drawdown_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes stock-level drawdown from each stock's previous peak.
    """
    rolling_peak = prices.cummax()
    drawdown = prices / rolling_peak - 1

    out = drawdown.stack().reset_index()
    out.columns = ["date", "ticker", "stock_drawdown"]

    return out


def build_price_features(monthly_prices: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    """
    Builds stock-level price features.
    Removes benchmark rows from final stock feature table.
    """
    returns = compute_returns(monthly_prices)
    ma_features = compute_moving_average_features(monthly_prices)
    vol_features = compute_volatility_features(monthly_prices)
    drawdown_features = compute_drawdown_features(monthly_prices)

    features = returns
    for f in [ma_features, vol_features, drawdown_features]:
        features = features.merge(f, on=["date", "ticker"], how="outer")

    features = features[features["ticker"] != benchmark].copy()

    return features


def build_market_features(monthly_prices: pd.DataFrame, macro: pd.DataFrame, regimes: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    """
    Builds market-wide features that will be merged onto every stock-date.
    """
    spy = monthly_prices[benchmark].copy()

    market = pd.DataFrame(index=monthly_prices.index)
    market.index.name = "date"

    market["spy_ret_1m"] = spy / spy.shift(1) - 1
    market["spy_ret_3m"] = spy / spy.shift(3) - 1
    market["spy_ret_6m"] = spy / spy.shift(6) - 1
    market["spy_ret_12m"] = spy / spy.shift(12) - 1

    market["spy_vol_3m"] = market["spy_ret_1m"].rolling(3).std()
    market["spy_vol_6m"] = market["spy_ret_1m"].rolling(6).std()
    market["spy_vol_12m"] = market["spy_ret_1m"].rolling(12).std()

    regimes = regimes.copy()
    regimes.index = pd.to_datetime(regimes.index)

    macro = macro.copy()
    macro.index = pd.to_datetime(macro.index)

    market = market.merge(regimes, left_index=True, right_index=True, how="left")
    market = market.merge(macro, left_index=True, right_index=True, how="left")

    if "ten_year" in market.columns and "two_year" in market.columns:
        market["yield_curve_10y_2y"] = market["ten_year"] - market["two_year"]

    if "cpi" in market.columns:
        market["cpi_yoy_change"] = market["cpi"] / market["cpi"].shift(12) - 1

    market = market.reset_index()

    return market


def add_sector_labels(features: pd.DataFrame) -> pd.DataFrame:
    """
    Adds simple sector labels manually for starter tickers.
    Later we can replace this with automatic sector metadata.
    """

    sector_map = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "NVDA": "Technology",
        "GOOGL": "Communication Services",
        "AMZN": "Consumer Discretionary",
        "META": "Communication Services",
        "JPM": "Financials",
        "XOM": "Energy",
        "UNH": "Healthcare",
        "WMT": "Consumer Staples",
        "PG": "Consumer Staples",
        "KO": "Consumer Staples",
        "HD": "Consumer Discretionary",
        "COST": "Consumer Staples",
        "AVGO": "Technology",
        "LLY": "Healthcare",
        "TSLA": "Consumer Discretionary",
        "V": "Financials",
        "MA": "Financials",
        "NFLX": "Communication Services",
    }

    features["sector"] = features["ticker"].map(sector_map)
    features["sector"] = features["sector"].fillna("Unknown")

    return features


def build_features():
    config = load_config("configs/experiment_01.yaml")

    processed_dir = config["processed_data_dir"]
    benchmark = config["benchmark"]

    monthly_prices = pd.read_parquet(os.path.join(processed_dir, "monthly_prices.parquet"))
    macro = pd.read_parquet(os.path.join(processed_dir, "macro_monthly.parquet"))
    regimes = pd.read_parquet(os.path.join(processed_dir, "market_regimes.parquet"))
    targets = pd.read_parquet(os.path.join(processed_dir, "targets.parquet"))

    monthly_prices.index = pd.to_datetime(monthly_prices.index)
    macro.index = pd.to_datetime(macro.index)
    regimes.index = pd.to_datetime(regimes.index)
    targets["date"] = pd.to_datetime(targets["date"])

    print("Building stock-level price features...")
    price_features = build_price_features(monthly_prices, benchmark=benchmark)

    print("Building market-level features...")
    market_features = build_market_features(monthly_prices, macro, regimes, benchmark=benchmark)

    print("Merging stock features with market features...")
    features = price_features.merge(market_features, on="date", how="left")

    print("Adding sector labels...")
    features = add_sector_labels(features)

    print("Merging features with targets...")
    dataset = features.merge(targets, on=["date", "ticker"], how="inner")

    dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)

    output_path = os.path.join(processed_dir, "features.parquet")
    dataset.to_parquet(output_path)

    print(f"Saved feature dataset to {output_path}")
    print("Dataset shape:", dataset.shape)
    print("Columns:")
    print(dataset.columns.tolist())


if __name__ == "__main__":
    build_features()