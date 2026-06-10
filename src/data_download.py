import os
import yaml
import yfinance as yf
import pandas as pd


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download_stock_prices(tickers, start_date, end_date, raw_data_dir):
    os.makedirs(raw_data_dir, exist_ok=True)

    all_tickers = tickers.copy()
    if "SPY" not in all_tickers:
        all_tickers.append("SPY")

    print(f"Downloading price data for {len(all_tickers)} tickers...")

    data = yf.download(
        all_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        group_by="ticker",
        progress=True,
    )

    output_path = os.path.join(raw_data_dir, "stock_prices_raw.parquet")
    data.to_parquet(output_path)

    print(f"Saved stock price data to {output_path}")


def download_fred_series(series_code, name, start_date, end_date):
    """
    Downloads one FRED series directly as a CSV.
    This avoids pandas-datareader compatibility issues.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}"

    df = pd.read_csv(url)
    df.columns = ["date", name]
    df["date"] = pd.to_datetime(df["date"])

    df[name] = pd.to_numeric(df[name], errors="coerce")

    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    df = df.set_index("date")

    return df


def download_macro_data(start_date, end_date, raw_data_dir):
    os.makedirs(raw_data_dir, exist_ok=True)

    fred_series = {
        "fed_funds": "FEDFUNDS",
        "ten_year": "DGS10",
        "two_year": "DGS2",
        "cpi": "CPIAUCSL",
        "unemployment": "UNRATE",
        "recession": "USREC",
    }

    frames = []

    for name, code in fred_series.items():
        print(f"Downloading FRED series: {name} ({code})")
        series = download_fred_series(code, name, start_date, end_date)
        frames.append(series)

    macro = pd.concat(frames, axis=1)
    macro.index.name = "date"

    output_path = os.path.join(raw_data_dir, "macro_raw.parquet")
    macro.to_parquet(output_path)

    print(f"Saved macro data to {output_path}")


def main():
    config = load_config("configs/experiment_01.yaml")

    tickers = config["tickers"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    raw_data_dir = config["raw_data_dir"]

    download_stock_prices(tickers, start_date, end_date, raw_data_dir)
    download_macro_data(start_date, end_date, raw_data_dir)


if __name__ == "__main__":
    main()