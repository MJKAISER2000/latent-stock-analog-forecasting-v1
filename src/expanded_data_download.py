import os
import pandas as pd
import yfinance as yf


def load_ticker_universe(path: str) -> list[str]:
    df = pd.read_csv(path)
    tickers = df["ticker"].dropna().astype(str).str.strip().unique().tolist()
    return tickers


def download_expanded_prices():
    universe_path = "data/external/expanded_ticker_universe.csv"
    raw_dir = "data/raw"

    os.makedirs(raw_dir, exist_ok=True)

    tickers = load_ticker_universe(universe_path)

    if "SPY" not in tickers:
        tickers.append("SPY")

    print(f"Downloading expanded universe with {len(tickers)} tickers...")
    print(tickers)

    data = yf.download(
        tickers,
        start="2000-01-01",
        end="2026-01-01",
        auto_adjust=True,
        group_by="ticker",
        progress=True,
        threads=True,
    )

    output_path = os.path.join(raw_dir, "expanded_stock_prices_raw.parquet")
    data.to_parquet(output_path)

    print(f"Saved expanded raw price data to {output_path}")


if __name__ == "__main__":
    download_expanded_prices()