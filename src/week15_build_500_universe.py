import os
from io import StringIO

import pandas as pd
import requests


def read_sp500_table(url: str) -> pd.DataFrame:
    """
    More reliable than pd.read_html(url) because it sends browser-like headers.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    print("Requesting S&P 500 page...")
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    print("Parsing HTML tables...")
    tables = pd.read_html(StringIO(response.text))

    if len(tables) == 0:
        raise ValueError("No tables found on the S&P 500 page.")

    return tables[0].copy()


def main():
    os.makedirs("data/external", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    output_path = "data/external/week15_500_stock_universe.csv"
    report_path = "outputs/reports/week15_500_universe_summary.txt"

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    sp500 = read_sp500_table(url)

    sp500.columns = [str(c).strip() for c in sp500.columns]

    print("Columns found:")
    print(sp500.columns.tolist())

    sp500 = sp500.rename(
        columns={
            "Symbol": "ticker",
            "Security": "company",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "industry",
        }
    )

    keep_cols = ["ticker", "company", "sector", "industry"]

    for col in keep_cols:
        if col not in sp500.columns:
            sp500[col] = ""

    universe = sp500[keep_cols].copy()

    # yfinance ticker formatting.
    # Example: BRK.B on Wikipedia becomes BRK-B for yfinance.
    universe["ticker"] = (
        universe["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
    )

    universe["company"] = universe["company"].astype(str).str.strip()
    universe["sector"] = universe["sector"].astype(str).str.strip()
    universe["industry"] = universe["industry"].astype(str).str.strip()

    universe = universe.dropna(subset=["ticker"])
    universe = universe[universe["ticker"] != ""]
    universe = universe.drop_duplicates(subset=["ticker"])
    universe = universe.sort_values("ticker").reset_index(drop=True)

    universe.to_csv(output_path, index=False)

    lines = []
    lines.append("Week 15 500-Stock Universe Summary")
    lines.append("==================================")
    lines.append("")
    lines.append("Goal:")
    lines.append("Build a broad 500-stock starting universe before creating a high-volatility subset.")
    lines.append("")
    lines.append("Source:")
    lines.append("Wikipedia S&P 500 constituents table")
    lines.append("")
    lines.append(f"Universe shape: {universe.shape}")
    lines.append(f"Ticker count: {universe['ticker'].nunique()}")
    lines.append("")
    lines.append("Sector counts:")
    lines.append(str(universe["sector"].value_counts()))
    lines.append("")
    lines.append("First 20 tickers:")
    lines.append(", ".join(universe["ticker"].head(20).tolist()))
    lines.append("")
    lines.append("Output file:")
    lines.append(output_path)
    lines.append("")
    lines.append("Important note:")
    lines.append(
        "This is not survivorship-bias-free because it uses the current S&P 500 list. "
        "It is useful as a broader 500-stock test universe, but later research should use historical index membership."
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("")
    print("\n".join(lines))
    print("")
    print("Saved:", output_path)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()