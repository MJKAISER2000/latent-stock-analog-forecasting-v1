import os
import pandas as pd
import numpy as np


HORIZONS = [1, 36]

UNIVERSE_FILES = {
    "balanced450": "data/external/week16_balanced450_universe.csv",
    "balanced900": "data/external/week16_balanced900_universe.csv",
    "lowvol300": "data/external/week16_lowvol300_universe.csv",
    "midvol300": "data/external/week16_midvol300_universe.csv",
    "highvol300": "data/external/week16_highvol300_universe.csv",
}


def compute_drawdown(price_series: pd.Series) -> pd.Series:
    running_max = price_series.cummax()
    return price_series / running_max - 1


def add_spy_features(features: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()

    spy = prices["SPY"].copy()
    spy_ret = spy / spy.shift(1) - 1

    spy_df = pd.DataFrame(index=prices.index)
    spy_df["spy_ret_1m"] = spy_ret
    spy_df["spy_ret_3m"] = spy / spy.shift(3) - 1
    spy_df["spy_ret_6m"] = spy / spy.shift(6) - 1
    spy_df["spy_ret_12m"] = spy / spy.shift(12) - 1
    spy_df["spy_vol_3m"] = spy_ret.rolling(3).std() * np.sqrt(12)
    spy_df["spy_vol_6m"] = spy_ret.rolling(6).std() * np.sqrt(12)
    spy_df["spy_vol_12m"] = spy_ret.rolling(12).std() * np.sqrt(12)
    spy_df["spy_drawdown"] = compute_drawdown(spy)

    spy_df = spy_df.reset_index()
    spy_df = spy_df.rename(columns={spy_df.columns[0]: "date"})
    spy_df["date"] = pd.to_datetime(spy_df["date"])

    features = features.merge(spy_df, on="date", how="left")

    features["bear_regime"] = (features["spy_drawdown"] <= -0.20).astype(int)
    features["correction_regime"] = (features["spy_drawdown"] <= -0.10).astype(int)
    features["crash_regime"] = (features["spy_drawdown"] <= -0.30).astype(int)

    return features


def build_features_from_prices(prices: pd.DataFrame, universe_tickers: list[str]) -> pd.DataFrame:
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    monthly_returns = prices / prices.shift(1) - 1

    rows = []

    for ticker in universe_tickers:
        if ticker not in prices.columns or ticker == "SPY":
            continue

        p = prices[ticker]
        r = monthly_returns[ticker]
        dd = compute_drawdown(p)

        ticker_df = pd.DataFrame(index=prices.index)
        ticker_df["date"] = prices.index
        ticker_df["ticker"] = ticker

        ticker_df["ret_1m"] = p / p.shift(1) - 1
        ticker_df["ret_3m"] = p / p.shift(3) - 1
        ticker_df["ret_6m"] = p / p.shift(6) - 1
        ticker_df["ret_12m"] = p / p.shift(12) - 1

        ticker_df["vol_3m"] = r.rolling(3).std() * np.sqrt(12)
        ticker_df["vol_6m"] = r.rolling(6).std() * np.sqrt(12)
        ticker_df["vol_12m"] = r.rolling(12).std() * np.sqrt(12)

        ticker_df["stock_drawdown"] = dd

        ticker_df["price_to_ma_3m"] = p / p.rolling(3).mean()
        ticker_df["price_to_ma_6m"] = p / p.rolling(6).mean()
        ticker_df["price_to_ma_12m"] = p / p.rolling(12).mean()

        ticker_df["return_to_vol_12m"] = ticker_df["ret_12m"] / ticker_df["vol_12m"].replace(0, np.nan)

        rows.append(ticker_df.reset_index(drop=True))

    features = pd.concat(rows, ignore_index=True)
    features["date"] = pd.to_datetime(features["date"])

    features = add_spy_features(features, prices)

    return features


def add_metadata(features: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    universe = universe.copy()

    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()

    for col in ["company", "exchange", "sector", "industry", "vol_bucket"]:
        if col not in universe.columns:
            universe[col] = "Unknown"

    keep_cols = ["ticker", "company", "exchange", "sector", "industry", "vol_bucket"]

    features = features.merge(universe[keep_cols], on="ticker", how="left")

    for col in ["company", "exchange", "sector", "industry", "vol_bucket"]:
        features[col] = features[col].fillna("Unknown").astype(str)

    return features


def add_group_relative_features(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()

    base_cols = [
        "ret_1m",
        "ret_3m",
        "ret_6m",
        "ret_12m",
        "vol_3m",
        "vol_6m",
        "vol_12m",
        "stock_drawdown",
    ]

    for group_col in ["vol_bucket", "exchange"]:
        if group_col not in features.columns:
            continue

        group = features.groupby(["date", group_col])

        for col in base_cols:
            if col not in features.columns:
                continue

            group_mean = group[col].transform("mean")
            group_std = group[col].transform("std")

            features[f"{col}_minus_{group_col}"] = features[col] - group_mean
            features[f"{col}_{group_col}_z"] = (features[col] - group_mean) / group_std.replace(0, np.nan)
            features[f"{col}_{group_col}_z"] = features[f"{col}_{group_col}_z"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return features


def add_targets(features: pd.DataFrame, prices: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    features = features.copy()
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)

    all_targets = []

    for horizon in horizons:
        future_prices = prices.shift(-horizon)
        future_returns = future_prices / prices - 1

        if "SPY" not in future_returns.columns:
            raise ValueError("SPY missing from prices.")

        spy_forward = future_returns["SPY"]

        rows = []

        for ticker in prices.columns:
            if ticker == "SPY":
                continue

            for date in prices.index:
                stock_ret = future_returns.loc[date, ticker]
                spy_ret = spy_forward.loc[date]

                if pd.isna(stock_ret) or pd.isna(spy_ret):
                    continue

                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        f"future_{horizon}m_return": stock_ret,
                        f"future_{horizon}m_spy_return": spy_ret,
                        f"future_{horizon}m_excess_return": stock_ret - spy_ret,
                        f"target_outperform_spy_{horizon}m": int(stock_ret > spy_ret),
                    }
                )

        target_h = pd.DataFrame(rows)

        if len(target_h) > 0:
            target_h[f"target_top_quintile_{horizon}m"] = 0

            for date, group in target_h.groupby("date"):
                cutoff = group[f"future_{horizon}m_return"].quantile(0.80)
                target_h.loc[group.index, f"target_top_quintile_{horizon}m"] = (
                    group[f"future_{horizon}m_return"] >= cutoff
                ).astype(int)

        all_targets.append(target_h)

    target = all_targets[0]

    for extra in all_targets[1:]:
        target = target.merge(extra, on=["date", "ticker"], how="outer")

    target["date"] = pd.to_datetime(target["date"])

    out = features.merge(target, on=["date", "ticker"], how="left")

    return out


def clean_modeling_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    categorical_cols = ["exchange", "sector", "industry", "vol_bucket"]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    df = pd.get_dummies(
        df,
        columns=[c for c in categorical_cols if c in df.columns],
        prefix=[c for c in categorical_cols if c in df.columns],
        drop_first=False,
    )

    non_numeric_keep = ["date", "ticker", "company"]

    for col in df.columns:
        if col not in non_numeric_keep:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    target_cols = [
        "future_1m_return",
        "future_1m_spy_return",
        "future_1m_excess_return",
        "target_outperform_spy_1m",
        "target_top_quintile_1m",
        "future_36m_return",
        "future_36m_spy_return",
        "future_36m_excess_return",
        "target_outperform_spy_36m",
        "target_top_quintile_36m",
    ]

    feature_cols = [
        c for c in df.columns
        if c not in non_numeric_keep and c not in target_cols
    ]

    missing_frac = df[feature_cols].isna().mean()
    keep_features = missing_frac[missing_frac < 0.40].index.tolist()
    feature_cols = keep_features

    for col in feature_cols:
        df[col] = df[col].fillna(df[col].median())

    final_cols = non_numeric_keep + target_cols + feature_cols
    final_cols = [c for c in final_cols if c in df.columns]

    df = df[final_cols]

    return df


def save_dataset(name: str, prices: pd.DataFrame, universe: pd.DataFrame):
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    print("")
    print("=" * 90)
    print(f"Building Week 16 dataset: {name}")

    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()
    tickers = universe["ticker"].dropna().unique().tolist()
    tickers = [t for t in tickers if t in prices.columns and t != "SPY"]

    print(f"Available tickers: {len(tickers)}")

    if len(tickers) == 0:
        raise ValueError(f"No tickers available for {name}")

    subset_prices = prices[tickers + ["SPY"]].copy()

    features = build_features_from_prices(subset_prices, tickers)
    features = add_metadata(features, universe)
    features = add_group_relative_features(features)
    features = add_targets(features, subset_prices, HORIZONS)

    clean = clean_modeling_dataset(features)

    output_path = f"data/processed/week16_{name}_modeling_dataset.parquet"
    report_path = f"outputs/reports/week16_{name}_modeling_dataset_summary.txt"

    clean.to_parquet(output_path)

    lines = []
    lines.append(f"Week 16 {name} Modeling Dataset Summary")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Universe name: {name}")
    lines.append(f"Ticker count: {clean['ticker'].nunique()}")
    lines.append(f"Shape: {clean.shape}")
    lines.append(f"Date range: {clean['date'].min()} to {clean['date'].max()}")
    lines.append(f"Missing total: {clean.isna().sum().sum()}")
    lines.append("")
    lines.append("Target balances:")
    for horizon in HORIZONS:
        col = f"target_outperform_spy_{horizon}m"
        if col in clean.columns:
            lines.append(f"{col}:")
            lines.append(str(clean[col].value_counts(normalize=True, dropna=True)))
            lines.append("")
    lines.append("Output file:")
    lines.append(output_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("Saved:", output_path)
    print("Saved:", report_path)


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    prices_path = "data/processed/week16_1000_monthly_prices.parquet"

    print("Loading Week 16 prices...")
    prices = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index)

    for name, path in UNIVERSE_FILES.items():
        print("")
        print("Loading universe:", name)
        universe = pd.read_csv(path)
        save_dataset(name, prices, universe)


if __name__ == "__main__":
    main()