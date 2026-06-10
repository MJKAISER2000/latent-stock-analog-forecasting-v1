import os
import pandas as pd
import numpy as np


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
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
    rolling_peak = prices.cummax()
    drawdown = prices / rolling_peak - 1

    out = drawdown.stack().reset_index()
    out.columns = ["date", "ticker", "stock_drawdown"]

    return out


def build_price_features(monthly_prices: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    returns = compute_returns(monthly_prices)
    ma_features = compute_moving_average_features(monthly_prices)
    vol_features = compute_volatility_features(monthly_prices)
    drawdown_features = compute_drawdown_features(monthly_prices)

    features = returns

    for f in [ma_features, vol_features, drawdown_features]:
        features = features.merge(f, on=["date", "ticker"], how="outer")

    features = features[features["ticker"] != benchmark].copy()

    return features


def build_market_features(
    monthly_prices: pd.DataFrame,
    macro: pd.DataFrame,
    regimes: pd.DataFrame,
    benchmark: str = "SPY",
) -> pd.DataFrame:
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


def clean_expanded_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    non_feature_cols = [
        "date",
        "ticker",
        "future_12m_return",
        "future_12m_spy_return",
        "target_abs_direction",
        "target_outperform_spy",
    ]

    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Drop features that are too missing
    missing_fraction = df[feature_cols].isna().mean()
    keep_feature_cols = missing_fraction[missing_fraction < 0.40].index.tolist()

    dropped = sorted(set(feature_cols) - set(keep_feature_cols))
    if dropped:
        print("Dropped mostly-missing columns:")
        print(dropped)

    feature_cols = keep_feature_cols

    for col in feature_cols:
        df[col] = df[col].fillna(df[col].median())

    final_cols = non_feature_cols + feature_cols
    df = df[final_cols]

    return df


def main():
    processed_dir = "data/processed"

    monthly_prices_path = os.path.join(processed_dir, "expanded_monthly_prices.parquet")
    targets_path = os.path.join(processed_dir, "expanded_targets.parquet")
    regimes_path = os.path.join(processed_dir, "expanded_market_regimes.parquet")
    macro_path = os.path.join(processed_dir, "macro_monthly.parquet")

    print("Loading expanded monthly prices...")
    monthly_prices = pd.read_parquet(monthly_prices_path)
    monthly_prices.index = pd.to_datetime(monthly_prices.index)

    print("Loading expanded targets...")
    targets = pd.read_parquet(targets_path)
    targets["date"] = pd.to_datetime(targets["date"])

    print("Loading market regimes and macro data...")
    regimes = pd.read_parquet(regimes_path)
    macro = pd.read_parquet(macro_path)

    print("Building expanded stock-level features...")
    price_features = build_price_features(monthly_prices, benchmark="SPY")

    print("Building expanded market-level features...")
    market_features = build_market_features(
        monthly_prices=monthly_prices,
        macro=macro,
        regimes=regimes,
        benchmark="SPY",
    )

    print("Merging features...")
    features = price_features.merge(market_features, on="date", how="left")

    print("Merging targets...")
    dataset = features.merge(targets, on=["date", "ticker"], how="inner")

    raw_output = os.path.join(processed_dir, "expanded_features.parquet")
    dataset.to_parquet(raw_output)
    print("Saved raw expanded features to:", raw_output)
    print("Raw expanded shape:", dataset.shape)

    print("Cleaning expanded dataset...")
    clean = clean_expanded_dataset(dataset)

    output_path = os.path.join(processed_dir, "expanded_modeling_dataset.parquet")
    clean.to_parquet(output_path)

    print("Saved expanded modeling dataset to:", output_path)
    print("Clean shape:", clean.shape)
    print("Missing values:", clean.isna().sum().sum())
    print("Ticker count:", clean["ticker"].nunique())
    print("Date range:", clean["date"].min(), "to", clean["date"].max())
    print("Target balance:")
    print(clean["target_outperform_spy"].value_counts(normalize=True))


if __name__ == "__main__":
    main()