import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:
    import yfinance as yf
except ImportError as exc:
    raise ImportError(
        "yfinance is required. Install it with: pip install yfinance"
    ) from exc


from src.utils.config import load_config, ensure_output_dirs


CONFIG_PATH = "configs/final_model_config.yaml"

INCLUDE_INCOMPLETE_CURRENT_MONTH = False

LIVE_DAILY_PRICES_PATH = PROJECT_ROOT / "data" / "processed" / "live_500_daily_prices.parquet"
LIVE_PRICES_PATH = PROJECT_ROOT / "data" / "processed" / "live_500_monthly_prices.parquet"
LIVE_RETURNS_PATH = PROJECT_ROOT / "data" / "processed" / "live_500_monthly_returns.parquet"

LIVE_PRICE_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "live_monthly_price_refresh_report.txt"
LIVE_PRICE_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "live_monthly_price_refresh_summary.csv"


def load_universe_tickers(config: dict) -> list[str]:
    universe_path = PROJECT_ROOT / config["paths"]["universe"]

    if not universe_path.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_path}")

    universe = pd.read_csv(universe_path)
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()

    tickers = sorted(universe["ticker"].dropna().unique().tolist())

    if "SPY" not in tickers:
        tickers.append("SPY")

    return tickers


def normalize_yfinance_columns(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if data.empty:
        raise ValueError("yfinance returned empty data.")

    if isinstance(data.columns, pd.MultiIndex):
        field_level_values = list(data.columns.get_level_values(0).unique())

        if "Close" not in field_level_values:
            raise ValueError(f"No Close field found. Fields: {field_level_values}")

        prices = data["Close"].copy()

    else:
        if "Close" not in data.columns:
            raise ValueError(f"No Close column found. Columns: {list(data.columns)}")

        prices = data[["Close"]].copy()
        prices.columns = tickers[:1]

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices.columns = [str(c).strip().upper() for c in prices.columns]

    for ticker in tickers:
        if ticker not in prices.columns:
            prices[ticker] = pd.NA

    prices = prices[tickers].copy()

    return prices


def download_daily_adjusted_prices(
    tickers: list[str],
    start: str = "2013-12-01",
) -> pd.DataFrame:
    print("")
    print("=" * 100)
    print("DOWNLOADING LIVE DAILY ADJUSTED PRICES")
    print("=" * 100)
    print("Ticker count:", len(tickers))
    print("Start:", start)

    data = yf.download(
        tickers=tickers,
        start=start,
        interval="1d",
        auto_adjust=True,
        group_by="column",
        progress=True,
        threads=True,
    )

    daily_prices = normalize_yfinance_columns(data, tickers)
    daily_prices = daily_prices.dropna(how="all")

    return daily_prices


def get_latest_completed_month_end(today: pd.Timestamp | None = None) -> pd.Timestamp:
    """
    Return the latest completed calendar month-end.

    Example:
    If today is 2026-06-03, latest completed month-end is 2026-05-31.
    If today is 2026-07-01, latest completed month-end is 2026-06-30.
    """

    if today is None:
        today = pd.Timestamp.today().normalize()
    else:
        today = pd.Timestamp(today).normalize()

    current_month_start = today.to_period("M").to_timestamp()
    latest_completed = current_month_start - pd.offsets.MonthEnd(1)

    return pd.Timestamp(latest_completed).normalize()


def resample_daily_to_month_end(
    daily_prices: pd.DataFrame,
    include_incomplete_current_month: bool = False,
) -> pd.DataFrame:
    """
    Convert daily adjusted closes to month-end adjusted closes.

    If include_incomplete_current_month=False, remove the current incomplete month.
    This prevents a partial month from being labeled as if it were the completed
    month-end price.
    """

    monthly = daily_prices.resample("ME").last()
    monthly = monthly.dropna(how="all")
    monthly.index = pd.to_datetime(monthly.index).normalize()

    if not include_incomplete_current_month:
        latest_completed_month_end = get_latest_completed_month_end()
        monthly = monthly[monthly.index <= latest_completed_month_end].copy()

    return monthly


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices / prices.shift(1) - 1
    return returns


def build_refresh_summary(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    rows = []

    latest_month = prices.index.max() if len(prices) > 0 else None

    for ticker in tickers:
        if ticker not in prices.columns:
            rows.append(
                {
                    "ticker": ticker,
                    "present": False,
                    "missing_count": None,
                    "non_missing_count": 0,
                    "first_date": None,
                    "last_date": None,
                    "latest_price": None,
                    "has_latest_month": False,
                }
            )
            continue

        series = pd.to_numeric(prices[ticker], errors="coerce")
        non_missing = series.dropna()

        if len(non_missing) == 0:
            first_date = None
            last_date = None
            latest_price = None
            has_latest_month = False
        else:
            first_date = non_missing.index.min()
            last_date = non_missing.index.max()
            latest_price = float(non_missing.iloc[-1])
            has_latest_month = (
                latest_month is not None
                and pd.Timestamp(last_date).normalize() == pd.Timestamp(latest_month).normalize()
            )

        rows.append(
            {
                "ticker": ticker,
                "present": True,
                "missing_count": int(series.isna().sum()),
                "non_missing_count": int(series.notna().sum()),
                "first_date": first_date,
                "last_date": last_date,
                "latest_price": latest_price,
                "has_latest_month": has_latest_month,
            }
        )

    summary = pd.DataFrame(rows)
    return summary


def write_report(
    daily_prices: pd.DataFrame,
    monthly_prices: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    summary: pd.DataFrame,
    paths: dict,
) -> None:
    missing_latest = summary[summary["has_latest_month"] == False].copy()

    latest_completed_month_end = get_latest_completed_month_end()

    lines = []
    lines.append("Latent Market Twin Live Monthly Price Refresh Report")
    lines.append("===================================================")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Method:")
    lines.append("Downloaded daily adjusted closes with yfinance auto_adjust=True.")
    lines.append("Resampled to month-end using the last available trading day in each month.")
    lines.append("")
    lines.append(f"Include incomplete current month: {INCLUDE_INCOMPLETE_CURRENT_MONTH}")
    lines.append(f"Latest completed month-end allowed: {latest_completed_month_end}")
    lines.append("")
    lines.append(f"Daily price shape: {daily_prices.shape}")
    lines.append(f"Daily date range: {daily_prices.index.min()} to {daily_prices.index.max()}")
    lines.append("")
    lines.append(f"Monthly price shape: {monthly_prices.shape}")
    lines.append(f"Monthly return shape: {monthly_returns.shape}")
    lines.append(f"Monthly date range: {monthly_prices.index.min()} to {monthly_prices.index.max()}")
    lines.append(f"Ticker count: {len(monthly_prices.columns)}")
    lines.append("")
    lines.append(f"Saved daily prices: {paths['daily_prices']}")
    lines.append(f"Saved monthly prices: {paths['prices']}")
    lines.append(f"Saved monthly returns: {paths['returns']}")
    lines.append(f"Saved summary: {paths['summary']}")
    lines.append("")
    lines.append("Missing latest-month tickers:")
    if len(missing_latest) == 0:
        lines.append("None")
    else:
        lines.append(missing_latest.to_string(index=False))
    lines.append("")
    lines.append("Summary head:")
    lines.append(summary.head(50).to_string(index=False))

    with open(paths["report"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    config = load_config(CONFIG_PATH)
    ensure_output_dirs(config)

    os.makedirs(PROJECT_ROOT / "data" / "processed", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "outputs" / "reports", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "outputs" / "tables", exist_ok=True)

    tickers = load_universe_tickers(config)

    daily_prices = download_daily_adjusted_prices(
        tickers=tickers,
        start="2013-12-01",
    )

    monthly_prices = resample_daily_to_month_end(
        daily_prices=daily_prices,
        include_incomplete_current_month=INCLUDE_INCOMPLETE_CURRENT_MONTH,
    )

    monthly_returns = compute_monthly_returns(monthly_prices)

    summary = build_refresh_summary(monthly_prices, tickers)

    daily_prices.to_parquet(LIVE_DAILY_PRICES_PATH)
    monthly_prices.to_parquet(LIVE_PRICES_PATH)
    monthly_returns.to_parquet(LIVE_RETURNS_PATH)
    summary.to_csv(LIVE_PRICE_SUMMARY_PATH, index=False)

    paths = {
        "daily_prices": LIVE_DAILY_PRICES_PATH,
        "prices": LIVE_PRICES_PATH,
        "returns": LIVE_RETURNS_PATH,
        "summary": LIVE_PRICE_SUMMARY_PATH,
        "report": LIVE_PRICE_REPORT_PATH,
    }

    write_report(
        daily_prices=daily_prices,
        monthly_prices=monthly_prices,
        monthly_returns=monthly_returns,
        summary=summary,
        paths=paths,
    )

    missing_latest = summary[summary["has_latest_month"] == False].copy()

    print("")
    print("=" * 100)
    print("LIVE MONTHLY PRICE REFRESH COMPLETE")
    print("=" * 100)
    print("Include incomplete current month:", INCLUDE_INCOMPLETE_CURRENT_MONTH)
    print("Latest completed month-end allowed:", get_latest_completed_month_end())
    print("")
    print("Daily price shape:", daily_prices.shape)
    print("Daily date range:", daily_prices.index.min(), "to", daily_prices.index.max())
    print("")
    print("Monthly price shape:", monthly_prices.shape)
    print("Monthly return shape:", monthly_returns.shape)
    print("Monthly date range:", monthly_prices.index.min(), "to", monthly_prices.index.max())
    print("Ticker count:", len(monthly_prices.columns))
    print("Missing latest-month tickers:", len(missing_latest))
    print("")
    print("Saved daily prices:", LIVE_DAILY_PRICES_PATH)
    print("Saved monthly prices:", LIVE_PRICES_PATH)
    print("Saved monthly returns:", LIVE_RETURNS_PATH)
    print("Saved summary:", LIVE_PRICE_SUMMARY_PATH)
    print("Saved report:", LIVE_PRICE_REPORT_PATH)

    if len(missing_latest) > 0:
        print("")
        print("MISSING LATEST-MONTH TICKERS")
        print(missing_latest.head(50).to_string(index=False))


if __name__ == "__main__":
    main()