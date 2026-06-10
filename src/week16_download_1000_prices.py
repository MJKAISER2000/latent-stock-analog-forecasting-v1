import os
import time
import pandas as pd
import yfinance as yf


def load_tickers(path: str) -> list[str]:
    df = pd.read_csv(path)

    tickers = (
        df["ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .unique()
        .tolist()
    )

    return tickers


def download_prices(tickers: list[str], start: str = "2014-01-01") -> pd.DataFrame:
    all_prices = {}

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] Downloading {ticker}...")

        try:
            data = yf.download(
                ticker,
                start=start,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if data.empty:
                print(f"  WARNING: no data for {ticker}")
                continue

            if "Close" not in data.columns:
                print(f"  WARNING: no Close column for {ticker}")
                continue

            close = data["Close"].copy()

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            close.name = ticker

            # Require enough daily history for rolling features and 36m testing.
            if close.dropna().shape[0] < 750:
                print(f"  WARNING: skipping {ticker}, not enough history")
                continue

            all_prices[ticker] = close

        except Exception as e:
            print(f"  ERROR downloading {ticker}: {e}")

        time.sleep(0.05)

    if not all_prices:
        raise ValueError("No usable price data downloaded.")

    prices = pd.concat(all_prices.values(), axis=1)
    prices = prices.sort_index()

    return prices


def make_monthly_prices(daily_prices: pd.DataFrame) -> pd.DataFrame:
    monthly = daily_prices.resample("ME").last()
    monthly = monthly.dropna(axis=1, how="all")
    return monthly


def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    universe_path = "data/external/week16_1000_stock_universe.csv"

    raw_path = "data/raw/week16_1000_daily_prices.parquet"
    monthly_path = "data/processed/week16_1000_monthly_prices.parquet"
    report_path = "outputs/reports/week16_1000_price_download_summary.txt"

    tickers = load_tickers(universe_path)

    if "SPY" not in tickers:
        tickers.append("SPY")

    print(f"Total tickers including SPY: {len(tickers)}")

    daily_prices = download_prices(tickers, start="2014-01-01")
    monthly_prices = make_monthly_prices(daily_prices)

    daily_prices.to_parquet(raw_path)
    monthly_prices.to_parquet(monthly_path)

    usable_tickers = [c for c in monthly_prices.columns if c != "SPY"]

    lines = []
    lines.append("Week 16 1000+ Universe Price Download Summary")
    lines.append("=============================================")
    lines.append("")
    lines.append("Goal:")
    lines.append("Download daily and monthly prices for the 1000+ ticker broad universe.")
    lines.append("")
    lines.append(f"Input tickers excluding SPY: {len(tickers) - 1}")
    lines.append(f"Usable tickers excluding SPY: {len(usable_tickers)}")
    lines.append(f"Daily price shape: {daily_prices.shape}")
    lines.append(f"Monthly price shape: {monthly_prices.shape}")
    lines.append(f"Monthly date range: {monthly_prices.index.min()} to {monthly_prices.index.max()}")
    lines.append("")
    lines.append("Usable tickers first 100:")
    lines.append(", ".join(usable_tickers[:100]))
    lines.append("")
    lines.append("Missing monthly values by ticker, top 30:")
    lines.append(str(monthly_prices.isna().sum().sort_values(ascending=False).head(30)))
    lines.append("")
    lines.append("Output files:")
    lines.append(raw_path)
    lines.append(monthly_path)
    lines.append("")
    lines.append("Important note:")
    lines.append(
        "This is a broad current-listed universe, not survivorship-bias-free. "
        "It is being used for a large-universe latent-structure stress test."
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("")
    print("\n".join(lines))
    print("")
    print("Saved:", raw_path)
    print("Saved:", monthly_path)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()