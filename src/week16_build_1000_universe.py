import os
from io import StringIO

import pandas as pd
import requests


MAX_TICKERS = 1500


def read_nasdaq_trader_file(url: str) -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    text = response.text

    # Nasdaq Trader symbol files have a footer line like:
    # File Creation Time:...
    lines = [
        line for line in text.splitlines()
        if line.strip() and not line.startswith("File Creation Time")
    ]

    clean_text = "\n".join(lines)
    return pd.read_csv(StringIO(clean_text), sep="|")


def clean_ticker(ticker: str) -> str:
    ticker = str(ticker).strip().upper()

    # yfinance uses '-' for class shares instead of '.'
    ticker = ticker.replace(".", "-")

    return ticker


def is_common_stock_like(row: pd.Series) -> bool:
    ticker = str(row.get("ticker", "")).upper()
    name = str(row.get("company", "")).upper()

    if ticker == "":
        return False

    # Remove weird ticker formats that usually cause bad yfinance downloads.
    bad_symbols = ["$", "^", "/", "="]
    if any(x in ticker for x in bad_symbols):
        return False

    # Remove warrants, units, rights, preferreds, notes, ETFs, funds, etc.
    bad_name_terms = [
        "ETF",
        "ETN",
        "FUND",
        "TRUST",
        "WARRANT",
        "RIGHT",
        "UNIT",
        "PREFERRED",
        "PREF",
        "NOTE",
        "DEBENTURE",
        "BOND",
        "ACQUISITION CORP",
        "SPAC",
        "REIT PREFERRED",
        "DEPOSITARY",
        "ADR",
    ]

    if any(term in name for term in bad_name_terms):
        return False

    return True


def main():
    os.makedirs("data/external", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    output_path = "data/external/week16_1000_stock_universe.csv"
    report_path = "outputs/reports/week16_1000_universe_summary.txt"

    nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    other_url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

    print("Downloading Nasdaq-listed symbols...")
    nasdaq = read_nasdaq_trader_file(nasdaq_url)

    print("Downloading other exchange-listed symbols...")
    other = read_nasdaq_trader_file(other_url)

    nasdaq = nasdaq.rename(
        columns={
            "Symbol": "ticker",
            "Security Name": "company",
        }
    )

    other = other.rename(
        columns={
            "ACT Symbol": "ticker",
            "Security Name": "company",
            "Exchange": "exchange",
        }
    )

    nasdaq["exchange"] = "NASDAQ"

    keep_cols = ["ticker", "company", "exchange"]

    nasdaq = nasdaq[[c for c in keep_cols if c in nasdaq.columns]].copy()
    other = other[[c for c in keep_cols if c in other.columns]].copy()

    universe = pd.concat([nasdaq, other], ignore_index=True)

    universe["ticker"] = universe["ticker"].apply(clean_ticker)
    universe["company"] = universe["company"].astype(str).str.strip()
    universe["exchange"] = universe["exchange"].astype(str).str.strip()

    universe = universe.dropna(subset=["ticker"])
    universe = universe.drop_duplicates(subset=["ticker"])

    universe = universe[universe.apply(is_common_stock_like, axis=1)].copy()

    # Add placeholder metadata so the Week 15/16 feature scripts still work.
    universe["sector"] = "Unknown"
    universe["industry"] = "Unknown"

    # Keep a manageable broad universe first.
    # Later price filtering will remove names without enough history.
    universe = universe.sort_values(["exchange", "ticker"]).reset_index(drop=True)
    universe = universe.head(MAX_TICKERS).copy()

    universe.to_csv(output_path, index=False)

    lines = []
    lines.append("Week 16 1000+ Stock Universe Summary")
    lines.append("====================================")
    lines.append("")
    lines.append("Goal:")
    lines.append("Build a broad 1000+ ticker U.S. stock universe before constructing balanced volatility buckets.")
    lines.append("")
    lines.append(f"Max tickers requested: {MAX_TICKERS}")
    lines.append(f"Universe shape: {universe.shape}")
    lines.append(f"Ticker count: {universe['ticker'].nunique()}")
    lines.append("")
    lines.append("Exchange counts:")
    lines.append(str(universe["exchange"].value_counts()))
    lines.append("")
    lines.append("First 50 tickers:")
    lines.append(", ".join(universe["ticker"].head(50).tolist()))
    lines.append("")
    lines.append("Output file:")
    lines.append(output_path)
    lines.append("")
    lines.append("Important note:")
    lines.append(
        "This universe is not survivorship-bias-free and does not yet include true sector metadata. "
        "It is intended as a larger broad-market stress-test universe. "
        "The next step is to download prices and filter names by data availability."
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("")
    print("Saved:", output_path)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()