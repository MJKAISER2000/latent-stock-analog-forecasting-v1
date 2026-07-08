"""Portfolio overlays for LTSAF v2.

Every overlay is a pure function on one month's target weights, using only
information available at the signal date, so the walk-forward guarantees of
the v1 backtest carry through unchanged.

Weights convention: a pandas Series indexed by ticker containing only stock
positions (no CASH row). Whatever the overlays don't allocate is cash; the
engine computes cash as 1 - sum(stock weights).
"""

import numpy as np
import pandas as pd


def dip_tilt(
    weights: pd.Series,
    price_to_ma: pd.Series,
    strength: float = 1.0,
    min_mult: float = 0.5,
    max_mult: float = 1.5,
) -> pd.Series:
    """Buy dips / sell high inside the sleeve, automatically.

    Each holding's weight is multiplied by ``1 - strength * (price_to_ma - 1)``
    (clipped to [min_mult, max_mult]) and the sleeve is renormalized to its
    original total. A stock 10% below its own moving average gets a ~1.10x
    tilt (buying the dip); a stock 10% extended above it gets ~0.90x (selling
    high). Total equity exposure is unchanged — this only redistributes it.
    """
    if len(weights) == 0:
        return weights

    ratio = price_to_ma.reindex(weights.index)
    ratio = pd.to_numeric(ratio, errors="coerce").fillna(1.0)

    mult = (1.0 - strength * (ratio - 1.0)).clip(min_mult, max_mult)
    tilted = weights * mult

    total_before = weights.sum()
    total_after = tilted.sum()
    if total_after <= 0:
        return weights

    return tilted * (total_before / total_after)


def spy_trend_is_on(
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    ma_months: int = 10,
) -> bool:
    """True when SPY's month-end close is at or above its trailing MA.

    Uses only prices up to the signal date. If there isn't enough history to
    compute the MA yet, defaults to risk-on.
    """
    spy = prices.loc[prices.index <= signal_date, "SPY"].dropna()
    if len(spy) < ma_months:
        return True
    return bool(spy.iloc[-1] >= spy.iloc[-ma_months:].mean())


def trend_hedge(
    weights: pd.Series,
    trend_on: bool,
    exposure_when_off: float = 0.4,
) -> pd.Series:
    """Scale the whole stock sleeve down when the market trend is off."""
    if trend_on:
        return weights
    return weights * exposure_when_off


def vol_target_exposure(
    past_returns: list[float],
    target_annual_vol: float = 0.15,
    window_months: int = 6,
    max_exposure: float = 1.0,
) -> float:
    """Exposure multiplier from the variant's own trailing realized vol.

    ``past_returns`` must contain only months strictly before the current
    signal date. Until a full window exists, exposure is 1.0.
    """
    if len(past_returns) < window_months:
        return 1.0

    recent = np.asarray(past_returns[-window_months:], dtype=float)
    realized = float(np.std(recent, ddof=1) * np.sqrt(12))
    if realized <= 0 or np.isnan(realized):
        return 1.0

    return float(np.clip(target_annual_vol / realized, 0.0, max_exposure))


def apply_no_trade_band(
    target: pd.Series,
    drifted: pd.Series,
    band: float = 0.005,
) -> pd.Series:
    """Skip trades smaller than the band; keep the drifted weight instead.

    Both inputs are stock-only weight Series. For each ticker in the union,
    trade to target only if |target - drifted| > band, otherwise hold the
    drifted position. New positions and full exits below the band are also
    skipped, which is what saves the recurring churn at the bottom of the
    book.
    """
    tickers = target.index.union(drifted.index)
    t = target.reindex(tickers, fill_value=0.0)
    d = drifted.reindex(tickers, fill_value=0.0)

    trade = (t - d).abs() > band
    actual = d.copy()
    actual[trade] = t[trade]

    return actual[actual != 0.0]
