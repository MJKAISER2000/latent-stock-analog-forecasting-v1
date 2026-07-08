"""Build the v3 expanded modeling dataset.

Starts from the v1 live dataset (live_full500_modeling_dataset.parquet — all
v1 features + targets) and adds:

Per-stock daily-data features (prefix v3_), all trailing windows so every
value is knowable at its month-end date:
  liquidity : log dollar volume, volume trend, Amihud illiquidity
  range vol : Parkinson volatility (1m/3m), intramonth range, downside vol
  shape     : daily-return skew / kurtosis (3m), MAX / MIN daily move (1m)
  position  : distance from 52w high/low, price vs 50d / 200d MA, MA cross
  market    : rolling 12m beta to SPY, 6m correlation, 6m idiosyncratic vol

Macro context features (prefix v3m_, one value per date shared by all
stocks — LightGBM uses them as split context / interactions):
  VIX level, change, 12m z-score; 10y & 13w yields, term spread + change;
  TLT / GLD / HYG / LQD returns; credit risk appetite (HYG-LQD); HYG
  drawdown; QQQ-IWM growth/small spread; SPY vs 10-month MA.

Output: data/processed/v3_modeling_dataset.parquet
"""

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(PROJECT_ROOT, "data", "processed")

BASE_PATH = os.path.join(DATA, "live_full500_modeling_dataset.parquet")
OUT_PATH = os.path.join(DATA, "v3_modeling_dataset.parquet")

D1M, D3M, D6M, D12M = 21, 63, 126, 252


def load_panel(name: str) -> pd.DataFrame:
    panel = pd.read_parquet(os.path.join(DATA, f"v3_daily_{name}.parquet"))
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()


def build_stock_daily_features() -> dict[str, pd.DataFrame]:
    close = load_panel("close")
    high = load_panel("high")
    low = load_panel("low")
    volume = load_panel("volume")

    dret = close.pct_change()
    spy = dret["SPY"]
    dollar_vol = close * volume

    feats: dict[str, pd.DataFrame] = {}

    # --- liquidity ---
    feats["v3_log_dollar_vol_1m"] = np.log(dollar_vol.rolling(D1M).mean().clip(lower=1.0))
    feats["v3_volume_ratio_3m_12m"] = (
        volume.rolling(D3M).mean() / volume.rolling(D12M).mean()
    )
    feats["v3_amihud_3m"] = (
        (dret.abs() / dollar_vol.replace(0, np.nan)).rolling(D3M, min_periods=40).mean() * 1e9
    )

    # --- range-based volatility ---
    log_hl_sq = np.log((high / low).clip(lower=1.0)) ** 2
    parkinson_factor = 1.0 / (4.0 * np.log(2.0))
    feats["v3_parkinson_vol_1m"] = np.sqrt(
        log_hl_sq.rolling(D1M).mean() * parkinson_factor
    ) * np.sqrt(252)
    feats["v3_parkinson_vol_3m"] = np.sqrt(
        log_hl_sq.rolling(D3M).mean() * parkinson_factor
    ) * np.sqrt(252)
    feats["v3_intramonth_range"] = (
        high.rolling(D1M).max() - low.rolling(D1M).min()
    ) / close
    feats["v3_downside_vol_3m"] = (
        dret.where(dret < 0).rolling(D3M, min_periods=20).std() * np.sqrt(252)
    )

    # --- return shape / lottery demand ---
    feats["v3_ret_skew_3m"] = dret.rolling(D3M, min_periods=40).skew()
    feats["v3_ret_kurt_3m"] = dret.rolling(D3M, min_periods=40).kurt()
    feats["v3_max_daily_ret_1m"] = dret.rolling(D1M).max()
    feats["v3_min_daily_ret_1m"] = dret.rolling(D1M).min()

    # --- price position ---
    feats["v3_pct_from_52w_high"] = close / close.rolling(D12M).max() - 1.0
    feats["v3_pct_from_52w_low"] = close / close.rolling(D12M).min() - 1.0
    feats["v3_price_to_ma50"] = close / close.rolling(50).mean()
    feats["v3_price_to_ma200"] = close / close.rolling(200).mean()
    feats["v3_ma50_over_ma200"] = close.rolling(50).mean() / close.rolling(200).mean()

    # --- market relationship ---
    spy_var = spy.rolling(D12M).var()
    beta = dret.rolling(D12M).cov(spy).div(spy_var, axis=0)
    feats["v3_beta_spy_12m"] = beta
    feats["v3_corr_spy_6m"] = dret.rolling(D6M).corr(spy)
    resid = dret.sub(beta.mul(spy, axis=0))
    feats["v3_idio_vol_6m"] = resid.rolling(D6M, min_periods=60).std() * np.sqrt(252)

    return feats


def snapshot_at_month_ends(
    feats: dict[str, pd.DataFrame], month_ends: pd.DatetimeIndex
) -> pd.DataFrame:
    """Take the last daily value at or before each month-end, long format."""
    out = None
    for name, panel in feats.items():
        # ffill so a holiday month-end still picks up the last trading day.
        snapped = panel.ffill(limit=7).reindex(month_ends, method="ffill")
        long = snapped.stack()
        long.name = name
        long.index.names = ["date", "ticker"]
        frame = long.reset_index()
        out = frame if out is None else out.merge(frame, on=["date", "ticker"], how="outer")
    return out


def build_macro_monthly(month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    macro = pd.read_parquet(os.path.join(DATA, "v3_macro_daily.parquet"))
    macro.index = pd.to_datetime(macro.index)
    macro = macro.sort_index().ffill(limit=7)
    m = macro.reindex(month_ends, method="ffill")

    def ret(col: str, months: int) -> pd.Series:
        return m[col] / m[col].shift(months) - 1.0

    out = pd.DataFrame(index=month_ends)
    out["v3m_vix"] = m["^VIX"]
    out["v3m_vix_chg_1m"] = m["^VIX"].diff()
    out["v3m_vix_z_12m"] = (
        (m["^VIX"] - m["^VIX"].rolling(12).mean()) / m["^VIX"].rolling(12).std()
    )
    out["v3m_yield_10y"] = m["^TNX"]
    out["v3m_yield_13w"] = m["^IRX"]
    out["v3m_term_spread"] = m["^TNX"] - m["^IRX"]
    out["v3m_term_spread_chg_3m"] = out["v3m_term_spread"].diff(3)
    out["v3m_tlt_ret_3m"] = ret("TLT", 3)
    out["v3m_gld_ret_3m"] = ret("GLD", 3)
    out["v3m_hyg_ret_3m"] = ret("HYG", 3)
    out["v3m_credit_appetite_3m"] = ret("HYG", 3) - ret("LQD", 3)
    out["v3m_hyg_drawdown"] = m["HYG"] / m["HYG"].cummax() - 1.0
    out["v3m_qqq_minus_iwm_3m"] = ret("QQQ", 3) - ret("IWM", 3)
    out["v3m_spy_to_ma10m"] = m["SPY"] / m["SPY"].rolling(10).mean()

    out.index.name = "date"
    return out.reset_index()


def main() -> None:
    base = pd.read_parquet(BASE_PATH)
    base["date"] = pd.to_datetime(base["date"])
    base["ticker"] = base["ticker"].astype(str).str.strip().str.upper()
    month_ends = pd.DatetimeIndex(sorted(base["date"].unique()))
    print(f"Base: {base.shape}, {month_ends.min().date()} to {month_ends.max().date()}", flush=True)

    print("Computing daily per-stock features...", flush=True)
    daily_feats = build_stock_daily_features()

    print("Snapshotting at month-ends...", flush=True)
    stock_feats = snapshot_at_month_ends(daily_feats, month_ends)
    stock_feats["ticker"] = stock_feats["ticker"].astype(str).str.strip().str.upper()
    print(f"Stock features: {stock_feats.shape}", flush=True)

    macro_feats = build_macro_monthly(month_ends)
    print(f"Macro features: {macro_feats.shape}", flush=True)

    merged = base.merge(stock_feats, on=["date", "ticker"], how="left")
    merged = merged.merge(macro_feats, on="date", how="left")

    new_cols = [c for c in merged.columns if c.startswith(("v3_", "v3m_"))]
    coverage = merged[new_cols].notna().mean().sort_values()
    print("\nNew feature coverage (lowest 8):")
    print(coverage.head(8).to_string())
    print(f"\nTotal columns: {len(merged.columns)} "
          f"({len(new_cols)} new v3 features)", flush=True)

    merged.to_parquet(OUT_PATH)
    print(f"Saved {OUT_PATH}: {merged.shape}", flush=True)


if __name__ == "__main__":
    main()
