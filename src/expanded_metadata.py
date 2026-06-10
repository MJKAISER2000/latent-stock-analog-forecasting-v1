import os
import time
import pandas as pd
import yfinance as yf


FALLBACK_SECTOR_MAP = {
    # Technology / semiconductors / software
    "AAPL": ("Technology", "Consumer Electronics"),
    "MSFT": ("Technology", "Software"),
    "NVDA": ("Technology", "Semiconductors"),
    "AVGO": ("Technology", "Semiconductors"),
    "AMD": ("Technology", "Semiconductors"),
    "ADBE": ("Technology", "Software"),
    "CSCO": ("Technology", "Communication Equipment"),
    "INTC": ("Technology", "Semiconductors"),
    "QCOM": ("Technology", "Semiconductors"),
    "TXN": ("Technology", "Semiconductors"),
    "AMAT": ("Technology", "Semiconductor Equipment"),
    "INTU": ("Technology", "Software"),
    "ADI": ("Technology", "Semiconductors"),
    "LRCX": ("Technology", "Semiconductor Equipment"),
    "PANW": ("Technology", "Cybersecurity"),
    "KLAC": ("Technology", "Semiconductor Equipment"),
    "MU": ("Technology", "Semiconductors"),
    "SNPS": ("Technology", "Software"),
    "CDNS": ("Technology", "Software"),
    "NXPI": ("Technology", "Semiconductors"),
    "FTNT": ("Technology", "Cybersecurity"),
    "MRVL": ("Technology", "Semiconductors"),
    "DDOG": ("Technology", "Software"),
    "TEAM": ("Technology", "Software"),
    "ZS": ("Technology", "Cybersecurity"),
    "CRWD": ("Technology", "Cybersecurity"),
    "MCHP": ("Technology", "Semiconductors"),
    "ON": ("Technology", "Semiconductors"),

    # Communication
    "GOOGL": ("Communication Services", "Internet Content"),
    "GOOG": ("Communication Services", "Internet Content"),
    "META": ("Communication Services", "Internet Content"),
    "NFLX": ("Communication Services", "Entertainment"),
    "TMUS": ("Communication Services", "Telecom"),
    "CHTR": ("Communication Services", "Telecom"),
    "EA": ("Communication Services", "Gaming"),
    "WBD": ("Communication Services", "Entertainment"),
    "TTD": ("Communication Services", "Advertising Technology"),
    "SIRI": ("Communication Services", "Audio Entertainment"),

    # Consumer discretionary
    "AMZN": ("Consumer Discretionary", "Internet Retail"),
    "TSLA": ("Consumer Discretionary", "Automobiles"),
    "BKNG": ("Consumer Discretionary", "Travel"),
    "SBUX": ("Consumer Discretionary", "Restaurants"),
    "MAR": ("Consumer Discretionary", "Hotels"),
    "MELI": ("Consumer Discretionary", "Internet Retail"),
    "ORLY": ("Consumer Discretionary", "Auto Parts Retail"),
    "ABNB": ("Consumer Discretionary", "Travel"),
    "ROST": ("Consumer Discretionary", "Apparel Retail"),
    "LULU": ("Consumer Discretionary", "Apparel"),
    "DLTR": ("Consumer Staples", "Discount Retail"),

    # Consumer staples
    "COST": ("Consumer Staples", "Warehouse Retail"),
    "PEP": ("Consumer Staples", "Beverages"),
    "MDLZ": ("Consumer Staples", "Packaged Foods"),
    "MNST": ("Consumer Staples", "Beverages"),
    "KDP": ("Consumer Staples", "Beverages"),
    "KHC": ("Consumer Staples", "Packaged Foods"),
    "WBA": ("Consumer Staples", "Pharmacy Retail"),

    # Healthcare
    "AMGN": ("Healthcare", "Biotechnology"),
    "GILD": ("Healthcare", "Biotechnology"),
    "VRTX": ("Healthcare", "Biotechnology"),
    "REGN": ("Healthcare", "Biotechnology"),
    "ISRG": ("Healthcare", "Medical Devices"),
    "IDXX": ("Healthcare", "Diagnostics"),
    "BIIB": ("Healthcare", "Biotechnology"),
    "AZN": ("Healthcare", "Pharmaceuticals"),
    "DXCM": ("Healthcare", "Medical Devices"),
    "GEHC": ("Healthcare", "Medical Technology"),
    "ILMN": ("Healthcare", "Life Sciences"),

    # Industrials
    "HON": ("Industrials", "Industrial Conglomerates"),
    "ADP": ("Industrials", "Human Capital Management"),
    "CSX": ("Industrials", "Railroads"),
    "ROP": ("Industrials", "Industrial Technology"),
    "PCAR": ("Industrials", "Trucks"),
    "PAYX": ("Industrials", "Human Capital Management"),
    "FAST": ("Industrials", "Industrial Distribution"),
    "ODFL": ("Industrials", "Trucking"),
    "CTAS": ("Industrials", "Business Services"),

    # Utilities / energy
    "AEP": ("Utilities", "Electric Utilities"),
    "EXC": ("Utilities", "Electric Utilities"),
    "XEL": ("Utilities", "Electric Utilities"),
    "BKR": ("Energy", "Oilfield Services"),

    # Financials
    "PYPL": ("Financials", "Payments"),
}


def load_ticker_universe(path: str) -> list[str]:
    df = pd.read_csv(path)
    return df["ticker"].dropna().astype(str).str.strip().unique().tolist()


def get_yfinance_metadata(ticker: str) -> dict:
    """
    Attempts to pull sector and industry from yfinance.
    Uses fallback values if yfinance metadata is missing or fails.
    """

    fallback_sector, fallback_industry = FALLBACK_SECTOR_MAP.get(
        ticker,
        ("Unknown", "Unknown"),
    )

    result = {
        "ticker": ticker,
        "sector": fallback_sector,
        "industry": fallback_industry,
        "long_name": "",
        "market_cap": None,
    }

    try:
        info = yf.Ticker(ticker).info

        result["sector"] = info.get("sector") or fallback_sector
        result["industry"] = info.get("industry") or fallback_industry
        result["long_name"] = info.get("longName") or info.get("shortName") or ""
        result["market_cap"] = info.get("marketCap")

    except Exception as e:
        print(f"Metadata failed for {ticker}: {e}")

    return result


def main():
    universe_path = "data/external/expanded_ticker_universe.csv"
    output_path = "data/external/expanded_ticker_metadata.csv"

    tickers = load_ticker_universe(universe_path)

    rows = []

    print(f"Fetching metadata for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] {ticker}")
        rows.append(get_yfinance_metadata(ticker))

        # small pause so yfinance does not get hammered
        time.sleep(0.25)

    metadata = pd.DataFrame(rows)

    metadata["sector"] = metadata["sector"].fillna("Unknown")
    metadata["industry"] = metadata["industry"].fillna("Unknown")

    metadata.to_csv(output_path, index=False)

    print("")
    print("Saved metadata to:", output_path)
    print("")
    print("Sector counts:")
    print(metadata["sector"].value_counts())
    print("")
    print("Industry counts, top 20:")
    print(metadata["industry"].value_counts().head(20))


if __name__ == "__main__":
    main()