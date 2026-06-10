import os
import pandas as pd


def load_ticker_universe(path: str) -> list[str]:
    df = pd.read_csv(path)
    tickers = df["ticker"].dropna().astype(str).str.strip().unique().tolist()
    return tickers


def extract_close_prices(raw_prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    close_data = {}

    for ticker in tickers:
        try:
            if (ticker, "Close") in raw_prices.columns:
                close_data[ticker] = raw_prices[(ticker, "Close")]
            elif (ticker, "Adj Close") in raw_prices.columns:
                close_data[ticker] = raw_prices[(ticker, "Adj Close")]
            else:
                print(f"Missing close data for {ticker}")
        except Exception as e:
            print(f"Error extracting {ticker}: {e}")

    close_df = pd.DataFrame(close_data)
    close_df.index = pd.to_datetime(close_df.index)
    close_df = close_df.sort_index()

    return close_df


def compute_forward_returns(monthly_prices: pd.DataFrame, horizon_months: int = 12) -> pd.DataFrame:
    future_prices = monthly_prices.shift(-horizon_months)
    return future_prices / monthly_prices - 1


def build_targets(monthly_prices: pd.DataFrame, benchmark: str = "SPY", horizon_months: int = 12) -> pd.DataFrame:
    forward_returns = compute_forward_returns(monthly_prices, horizon_months)

    if benchmark not in forward_returns.columns:
        raise ValueError(f"{benchmark} not found in monthly price data.")

    spy_forward = forward_returns[benchmark]

    rows = []

    for ticker in forward_returns.columns:
        if ticker == benchmark:
            continue

        for date in forward_returns.index:
            stock_ret = forward_returns.loc[date, ticker]
            spy_ret = spy_forward.loc[date]

            if pd.isna(stock_ret) or pd.isna(spy_ret):
                continue

            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "future_12m_return": stock_ret,
                    "future_12m_spy_return": spy_ret,
                    "target_abs_direction": int(stock_ret > 0),
                    "target_outperform_spy": int(stock_ret > spy_ret),
                }
            )

    return pd.DataFrame(rows)


def add_market_regime_labels(spy_monthly: pd.Series) -> pd.DataFrame:
    df = spy_monthly.to_frame("spy_price").copy()
    df = df.sort_index()

    df["rolling_peak"] = df["spy_price"].cummax()
    df["drawdown"] = df["spy_price"] / df["rolling_peak"] - 1

    df["correction_regime"] = (df["drawdown"] <= -0.10).astype(int)
    df["bear_regime"] = (df["drawdown"] <= -0.20).astype(int)
    df["crash_regime"] = (df["drawdown"] <= -0.30).astype(int)

    return df


def main():
    raw_path = "data/raw/expanded_stock_prices_raw.parquet"
    universe_path = "data/external/expanded_ticker_universe.csv"
    processed_dir = "data/processed"

    os.makedirs(processed_dir, exist_ok=True)

    tickers = load_ticker_universe(universe_path)

    if "SPY" not in tickers:
        tickers.append("SPY")

    print("Loading expanded raw price data...")
    raw_prices = pd.read_parquet(raw_path)

    print("Extracting close prices...")
    close_df = extract_close_prices(raw_prices, tickers)

    print("Converting daily prices to monthly prices...")
    monthly_prices = close_df.resample("ME").last()

    print("Monthly prices shape:", monthly_prices.shape)

    monthly_prices_path = os.path.join(processed_dir, "expanded_monthly_prices.parquet")
    monthly_prices.to_parquet(monthly_prices_path)
    print("Saved:", monthly_prices_path)

    print("Building expanded targets...")
    targets = build_targets(monthly_prices, benchmark="SPY", horizon_months=12)

    targets_path = os.path.join(processed_dir, "expanded_targets.parquet")
    targets.to_parquet(targets_path)
    print("Saved:", targets_path)
    print("Targets shape:", targets.shape)

    print("Building expanded market regime labels...")
    regimes = add_market_regime_labels(monthly_prices["SPY"])

    regimes_path = os.path.join(processed_dir, "expanded_market_regimes.parquet")
    regimes.to_parquet(regimes_path)
    print("Saved:", regimes_path)

    print("")
    print("Target balance:")
    print(targets["target_outperform_spy"].value_counts(normalize=True))

    print("")
    print("Ticker count in targets:", targets["ticker"].nunique())
    print("Date range:", targets["date"].min(), "to", targets["date"].max())


if __name__ == "__main__":
    main()