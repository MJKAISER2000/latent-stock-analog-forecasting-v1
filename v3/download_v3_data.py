"""Download the expanded v3 data from yfinance.

Two downloads:
1. Daily OHLCV (auto-adjusted) for the full ~503-ticker universe + SPY.
   v1 only kept adjusted closes; high/low/volume unlock range-based
   volatility, liquidity, and lottery-demand features.
2. Daily closes for macro / hedge tickers: VIX, treasury yield indices,
   bond/gold/credit ETFs, style indices.

Outputs (data/processed/):
    v3_daily_open.parquet, v3_daily_high.parquet, v3_daily_low.parquet,
    v3_daily_close.parquet, v3_daily_volume.parquet   (wide: date x ticker)
    v3_macro_daily.parquet                            (wide: date x ticker)
"""

import os
import sys
import time

import pandas as pd
import yfinance as yf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UNIVERSE_PATH = os.path.join(PROJECT_ROOT, "data", "external", "week15_500_stock_universe.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

START = "2013-01-01"
BATCH_SIZE = 100
FIELDS = ["Open", "High", "Low", "Close", "Volume"]

MACRO_TICKERS = [
    "^VIX",    # implied vol
    "^TNX",    # 10y yield
    "^IRX",    # 13w yield
    "^FVX",    # 5y yield
    "TLT",     # long treasuries
    "IEF",     # 7-10y treasuries
    "GLD",     # gold
    "HYG",     # high yield credit
    "LQD",     # investment grade credit
    "QQQ",     # growth/tech
    "IWM",     # small caps
    "SPY",
]


def download_batched(tickers: list[str], start: str) -> dict[str, pd.DataFrame]:
    """Download OHLCV in batches with one retry; returns {field: wide panel}."""
    panels: dict[str, list[pd.DataFrame]] = {f: [] for f in FIELDS}

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        for attempt in (1, 2):
            try:
                raw = yf.download(
                    batch,
                    start=start,
                    auto_adjust=True,
                    progress=False,
                    group_by="column",
                    threads=True,
                )
                break
            except Exception as exc:
                print(f"batch {i // BATCH_SIZE + 1} attempt {attempt} failed: {exc}", flush=True)
                if attempt == 2:
                    raise
                time.sleep(15)

        for field in FIELDS:
            block = raw[field] if isinstance(raw.columns, pd.MultiIndex) else raw[[field]]
            if isinstance(block, pd.Series):
                block = block.to_frame(batch[0])
            panels[field].append(block)

        print(f"batch {i // BATCH_SIZE + 1}/{(len(tickers) - 1) // BATCH_SIZE + 1} done "
              f"({len(batch)} tickers)", flush=True)
        time.sleep(2)

    return {f: pd.concat(parts, axis=1).sort_index() for f, parts in panels.items()}


def main() -> None:
    universe = pd.read_csv(UNIVERSE_PATH)
    tickers = sorted(set(
        universe["ticker"].astype(str).str.strip().str.upper().tolist() + ["SPY"]
    ))
    print(f"Universe: {len(tickers)} tickers from {START}", flush=True)

    panels = download_batched(tickers, START)

    for field, panel in panels.items():
        panel.index = pd.to_datetime(panel.index)
        panel.columns = [str(c).strip().upper() for c in panel.columns]
        panel = panel.loc[:, ~panel.columns.duplicated()]
        path = os.path.join(OUT_DIR, f"v3_daily_{field.lower()}.parquet")
        panel.to_parquet(path)
        good = panel.notna().any().sum()
        print(f"saved {path}: {panel.shape}, tickers with data: {good}", flush=True)

    print("Downloading macro tickers...", flush=True)
    macro = yf.download(
        MACRO_TICKERS, start=START, auto_adjust=True, progress=False, group_by="column",
    )["Close"]
    macro.index = pd.to_datetime(macro.index)
    macro_path = os.path.join(OUT_DIR, "v3_macro_daily.parquet")
    macro.to_parquet(macro_path)
    print(f"saved {macro_path}: {macro.shape}", flush=True)
    print("DOWNLOAD COMPLETE", flush=True)


if __name__ == "__main__":
    main()
