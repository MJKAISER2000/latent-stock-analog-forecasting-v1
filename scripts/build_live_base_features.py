import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, ensure_output_dirs


CONFIG_PATH = "configs/final_model_config.yaml"

LIVE_PRICES_PATH = PROJECT_ROOT / "data" / "processed" / "live_500_monthly_prices.parquet"
LIVE_BASE_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "live_full500_modeling_dataset.parquet"

LIVE_FEATURE_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "live_base_feature_build_report.txt"
LIVE_FEATURE_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "live_base_feature_summary.csv"


RETURN_WINDOWS = [1, 3, 6, 12]
VOL_WINDOWS = [3, 6, 12]
MA_WINDOWS = [3, 6, 12]
DRAWDOWN_WINDOWS = [6, 12]


def load_live_prices() -> pd.DataFrame:
    if not LIVE_PRICES_PATH.exists():
        raise FileNotFoundError(
            f"Live monthly price file not found: {LIVE_PRICES_PATH}. "
            f"Run scripts/refresh_live_monthly_prices.py first."
        )

    prices = pd.read_parquet(LIVE_PRICES_PATH)
    prices.index = pd.to_datetime(prices.index).normalize()
    prices = prices.sort_index()
    prices.columns = [str(c).strip().upper() for c in prices.columns]

    return prices


def load_universe(config: dict) -> pd.DataFrame:
    universe_path = PROJECT_ROOT / config["paths"]["universe"]

    if not universe_path.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_path}")

    universe = pd.read_csv(universe_path)
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()

    if "sector" not in universe.columns:
        universe["sector"] = "Unknown"

    if "industry" not in universe.columns:
        universe["industry"] = "Unknown"

    if "company" not in universe.columns:
        universe["company"] = universe["ticker"]

    universe["sector"] = universe["sector"].fillna("Unknown").astype(str)
    universe["industry"] = universe["industry"].fillna("Unknown").astype(str)
    universe["company"] = universe["company"].fillna(universe["ticker"]).astype(str)

    universe = universe.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    return universe


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices / prices.shift(1) - 1.0
    return returns


def build_long_base(prices: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    tickers = [t for t in universe["ticker"].tolist() if t in prices.columns]

    if "SPY" not in prices.columns:
        raise ValueError("SPY must be present in live monthly prices.")

    rows = []

    for ticker in tickers:
        series = pd.to_numeric(prices[ticker], errors="coerce")

        temp = pd.DataFrame(
            {
                "date": series.index,
                "ticker": ticker,
                "price": series.values,
            }
        )

        rows.append(temp)

    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()

    out = out.merge(
        universe[["ticker", "company", "sector", "industry"]],
        on="ticker",
        how="left",
    )

    out["company"] = out["company"].fillna(out["ticker"])
    out["sector"] = out["sector"].fillna("Unknown")
    out["industry"] = out["industry"].fillna("Unknown")

    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    return out


def add_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    group = out.groupby("ticker", group_keys=False)

    out["ret_1m"] = group["price"].pct_change(1)

    for window in [3, 6, 12]:
        out[f"ret_{window}m"] = group["price"].pct_change(window)

    for window in MA_WINDOWS:
        out[f"price_ma_{window}m"] = group["price"].transform(
            lambda s: s.rolling(window).mean()
        )
        out[f"price_to_ma_{window}m"] = out["price"] / out[f"price_ma_{window}m"] - 1.0

    for window in VOL_WINDOWS:
        out[f"vol_{window}m"] = group["ret_1m"].transform(
            lambda s: s.rolling(window).std()
        )

    for window in DRAWDOWN_WINDOWS:
        rolling_max = group["price"].transform(
            lambda s: s.rolling(window).max()
        )
        out[f"stock_drawdown_{window}m"] = out["price"] / rolling_max - 1.0

    out["stock_drawdown"] = group["price"].transform(
        lambda s: s / s.cummax() - 1.0
    )

    out["future_1m_return"] = group["price"].pct_change(1).groupby(out["ticker"]).shift(-1)

    return out


def add_spy_features(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    spy = pd.to_numeric(prices["SPY"], errors="coerce").sort_index()
    spy_returns = spy.pct_change(1)

    spy_features = pd.DataFrame(index=spy.index)
    spy_features["date"] = spy_features.index
    spy_features["spy_price"] = spy
    spy_features["spy_ret_1m"] = spy_returns

    for window in [3, 6, 12]:
        spy_features[f"spy_ret_{window}m"] = spy.pct_change(window)
        spy_features[f"spy_vol_{window}m"] = spy_returns.rolling(window).std()

    spy_features["spy_drawdown"] = spy / spy.cummax() - 1.0

    spy_features = spy_features.reset_index(drop=True)
    spy_features["date"] = pd.to_datetime(spy_features["date"]).dt.normalize()

    out = out.merge(spy_features, on="date", how="left")

    out["future_1m_spy_return"] = out.groupby("ticker")["spy_ret_1m"].shift(-1)
    out["future_1m_excess_return"] = out["future_1m_return"] - out["future_1m_spy_return"]
    out["target_outperform_spy_1m"] = (
        out["future_1m_excess_return"] > 0
    ).astype(float)

    return out


def add_sector_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    sector_returns = (
        out.groupby(["date", "sector"])["ret_1m"]
        .mean()
        .reset_index()
        .rename(columns={"ret_1m": "sector_ret_1m"})
    )

    out = out.merge(sector_returns, on=["date", "sector"], how="left")

    for window in [3, 6, 12]:
        sector_window = (
            out.groupby(["date", "sector"])[f"ret_{window}m"]
            .mean()
            .reset_index()
            .rename(columns={f"ret_{window}m": f"sector_ret_{window}m"})
        )

        out = out.merge(sector_window, on=["date", "sector"], how="left")
        out[f"ret_{window}m_minus_sector"] = out[f"ret_{window}m"] - out[f"sector_ret_{window}m"]

    out["ret_1m_minus_sector"] = out["ret_1m"] - out["sector_ret_1m"]

    return out


def add_industry_and_sector_dummies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    sector_dummies = pd.get_dummies(out["sector"], prefix="sector", dtype=int)
    industry_dummies = pd.get_dummies(out["industry"], prefix="industry", dtype=int)

    out = pd.concat([out, sector_dummies, industry_dummies], axis=1)

    return out


def add_ranking_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ranking_label"] = np.nan

    for date, group in out.groupby("date"):
        valid = group.dropna(subset=["future_1m_return"]).copy()

        if len(valid) == 0:
            continue

        ranks = valid["future_1m_return"].rank(method="first", pct=True)

        labels = pd.cut(
            ranks,
            bins=[0.0, 0.20, 0.40, 0.60, 0.80, 1.0],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True,
        ).astype(float)

        out.loc[valid.index, "ranking_label"] = labels

    return out


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Keep rows with a real stock price.
    out = out.dropna(subset=["price"]).copy()

    # Do not include SPY itself as a stock candidate unless it is in universe.
    # If SPY is in the universe file, it will remain. Usually it should not.
    # The benchmark still exists through spy_* columns.
    if "SPY" in out["ticker"].unique():
        out = out[out["ticker"] != "SPY"].copy()

    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()

    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)

    return out


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    rows = []

    for col in numeric_cols:
        series = df[col]

        rows.append(
            {
                "column": col,
                "missing_count": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean()),
                "mean": float(series.mean(skipna=True)) if series.notna().any() else np.nan,
                "std": float(series.std(skipna=True)) if series.notna().any() else np.nan,
                "min": float(series.min(skipna=True)) if series.notna().any() else np.nan,
                "max": float(series.max(skipna=True)) if series.notna().any() else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False).reset_index(drop=True)


def write_report(df: pd.DataFrame, summary: pd.DataFrame, config: dict) -> None:
    lines = []
    lines.append("Latent Market Twin Live Base Feature Build Report")
    lines.append("================================================")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Output dataset: {LIVE_BASE_DATASET_PATH}")
    lines.append(f"Shape: {df.shape}")
    lines.append(f"Date range: {df['date'].min()} to {df['date'].max()}")
    lines.append(f"Ticker count: {df['ticker'].nunique()}")
    lines.append(f"Sector count: {df['sector'].nunique()}")
    lines.append(f"Industry count: {df['industry'].nunique()}")
    lines.append("")
    lines.append(f"Future 1m return missing rows: {int(df['future_1m_return'].isna().sum())}")
    lines.append(f"Ranking label missing rows: {int(df['ranking_label'].isna().sum())}")
    lines.append("")
    lines.append("Columns:")
    lines.append(", ".join(df.columns.tolist()))
    lines.append("")
    lines.append("Feature missingness summary head:")
    lines.append(summary.head(80).to_string(index=False))

    with open(LIVE_FEATURE_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    config = load_config(CONFIG_PATH)
    ensure_output_dirs(config)

    os.makedirs(PROJECT_ROOT / "data" / "processed", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "outputs" / "reports", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "outputs" / "tables", exist_ok=True)

    prices = load_live_prices()
    universe = load_universe(config)

    print("")
    print("=" * 100)
    print("BUILDING LIVE BASE FEATURES")
    print("=" * 100)
    print("Live prices shape:", prices.shape)
    print("Live price date range:", prices.index.min(), "to", prices.index.max())
    print("Universe shape:", universe.shape)

    df = build_long_base(prices, universe)
    df = add_stock_features(df)
    df = add_spy_features(df, prices)
    df = add_sector_features(df)
    df = add_industry_and_sector_dummies(df)
    df = add_ranking_labels(df)
    df = clean_dataset(df)

    summary = build_summary(df)

    df.to_parquet(LIVE_BASE_DATASET_PATH)
    summary.to_csv(LIVE_FEATURE_SUMMARY_PATH, index=False)

    write_report(df, summary, config)

    print("")
    print("=" * 100)
    print("LIVE BASE FEATURE BUILD COMPLETE")
    print("=" * 100)
    print("Dataset shape:", df.shape)
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Ticker count:", df["ticker"].nunique())
    print("Column count:", len(df.columns))
    print("Future 1m return missing:", int(df["future_1m_return"].isna().sum()))
    print("Ranking label missing:", int(df["ranking_label"].isna().sum()))
    print("")
    print("Saved dataset:", LIVE_BASE_DATASET_PATH)
    print("Saved summary:", LIVE_FEATURE_SUMMARY_PATH)
    print("Saved report:", LIVE_FEATURE_REPORT_PATH)
    print("")
    print("SAMPLE")
    print(df.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()