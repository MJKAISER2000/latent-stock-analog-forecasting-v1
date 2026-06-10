import os
import yaml
import pandas as pd

from targets import build_targets
from regime_labels import add_market_regime_labels


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_adjusted_close(raw_prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    yfinance multi-ticker download gives a multi-index column structure.
    This extracts adjusted close / close prices into a clean table.
    """

    close_data = {}

    for ticker in tickers:
        try:
            if (ticker, "Close") in raw_prices.columns:
                close_data[ticker] = raw_prices[(ticker, "Close")]
            elif (ticker, "Adj Close") in raw_prices.columns:
                close_data[ticker] = raw_prices[(ticker, "Adj Close")]
            else:
                print(f"Could not find close column for {ticker}")
        except Exception as e:
            print(f"Error extracting {ticker}: {e}")

    close_df = pd.DataFrame(close_data)
    close_df.index = pd.to_datetime(close_df.index)
    close_df = close_df.sort_index()

    return close_df


def daily_to_monthly_prices(close_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts daily prices to month-end prices.
    """
    monthly = close_df.resample("ME").last()
    return monthly


def build_base_dataset(config: dict):
    raw_data_dir = config["raw_data_dir"]
    processed_data_dir = config["processed_data_dir"]
    tickers = config["tickers"]
    benchmark = config["benchmark"]
    horizon = config["prediction_horizon_months"]

    os.makedirs(processed_data_dir, exist_ok=True)

    all_tickers = tickers.copy()
    if benchmark not in all_tickers:
        all_tickers.append(benchmark)

    raw_prices_path = os.path.join(raw_data_dir, "stock_prices_raw.parquet")
    macro_path = os.path.join(raw_data_dir, "macro_raw.parquet")

    print("Loading raw price data...")
    raw_prices = pd.read_parquet(raw_prices_path)

    print("Extracting close prices...")
    close_df = extract_adjusted_close(raw_prices, all_tickers)

    print("Converting daily prices to monthly prices...")
    monthly_prices = daily_to_monthly_prices(close_df)

    monthly_prices_path = os.path.join(processed_data_dir, "monthly_prices.parquet")
    monthly_prices.to_parquet(monthly_prices_path)
    print(f"Saved monthly prices to {monthly_prices_path}")

    print("Building targets...")
    targets = build_targets(
        monthly_prices=monthly_prices,
        benchmark=benchmark,
        horizon_months=horizon,
    )

    targets_path = os.path.join(processed_data_dir, "targets.parquet")
    targets.to_parquet(targets_path)
    print(f"Saved targets to {targets_path}")

    print("Building regime labels...")
    regime = add_market_regime_labels(monthly_prices[benchmark])

    regime_path = os.path.join(processed_data_dir, "market_regimes.parquet")
    regime.to_parquet(regime_path)
    print(f"Saved regime labels to {regime_path}")

    print("Loading macro data...")
    macro = pd.read_parquet(macro_path)
    macro = macro.resample("ME").last()
    macro.index.name = "date"

    macro_path_out = os.path.join(processed_data_dir, "macro_monthly.parquet")
    macro.to_parquet(macro_path_out)
    print(f"Saved monthly macro data to {macro_path_out}")


def main():
    config = load_config("configs/experiment_01.yaml")
    build_base_dataset(config)


if __name__ == "__main__":
    main()