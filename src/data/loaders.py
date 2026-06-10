import os
from typing import Any

import pandas as pd


def require_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")


def load_base_dataset(config: dict[str, Any]) -> pd.DataFrame:
    path = config["paths"]["base_dataset"]
    require_file(path)

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    return df


def load_neighbor_dataset(config: dict[str, Any]) -> pd.DataFrame:
    path = config["paths"]["neighbor_dataset"]
    require_file(path)

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    return df


def load_monthly_prices(config: dict[str, Any]) -> pd.DataFrame:
    path = config["paths"]["monthly_prices"]
    require_file(path)

    prices = pd.read_parquet(path)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices.columns = [str(c).strip().upper() for c in prices.columns]

    return prices


def load_universe(config: dict[str, Any]) -> pd.DataFrame:
    path = config["paths"]["universe"]
    require_file(path)

    universe = pd.read_csv(path)
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()

    if "sector" in universe.columns:
        universe["sector"] = universe["sector"].fillna("Unknown").astype(str)

    if "industry" in universe.columns:
        universe["industry"] = universe["industry"].fillna("Unknown").astype(str)

    return universe


def load_monthly_returns(config: dict[str, Any]) -> pd.DataFrame:
    prices = load_monthly_prices(config)
    monthly_returns = prices / prices.shift(1) - 1

    rows = []

    for ticker in monthly_returns.columns:
        series = monthly_returns[ticker].dropna()

        for date, ret in series.items():
            rows.append(
                {
                    "date": pd.to_datetime(date),
                    "ticker": str(ticker).strip().upper(),
                    "monthly_return": float(ret),
                }
            )

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)

    return out


def get_latest_available_signal_date(df: pd.DataFrame) -> pd.Timestamp:
    if "date" not in df.columns:
        raise ValueError("Dataset must contain a date column.")

    dates = pd.to_datetime(df["date"]).dropna().sort_values()

    if len(dates) == 0:
        raise ValueError("No valid dates found.")

    return pd.Timestamp(dates.max())


def print_dataset_summary(name: str, df: pd.DataFrame) -> None:
    print("")
    print("=" * 80)
    print(name)
    print("=" * 80)
    print("Shape:", df.shape)

    if "date" in df.columns:
        print("Date range:", pd.to_datetime(df["date"]).min(), "to", pd.to_datetime(df["date"]).max())

    if "ticker" in df.columns:
        print("Ticker count:", df["ticker"].nunique())

    print("Columns:", len(df.columns))