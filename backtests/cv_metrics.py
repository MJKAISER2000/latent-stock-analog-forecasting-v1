"""Performance metrics for the walk-forward cross-validation backtest.

All functions operate on a pandas Series of simple (non-log) monthly returns
indexed by month-end date. Risk-free rate is assumed 0, so Sharpe here is
mean/std of raw monthly returns annualized by sqrt(12).
"""

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


def annualized_return(monthly_returns: pd.Series) -> float:
    """Geometric annualized return from monthly simple returns."""
    r = monthly_returns.dropna()
    if len(r) == 0:
        return np.nan
    total_growth = (1.0 + r).prod()
    if total_growth <= 0:
        return -1.0
    return float(total_growth ** (MONTHS_PER_YEAR / len(r)) - 1.0)


def annualized_volatility(monthly_returns: pd.Series) -> float:
    r = monthly_returns.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def sharpe_ratio(monthly_returns: pd.Series) -> float:
    """Annualized Sharpe with rf = 0."""
    r = monthly_returns.dropna()
    if len(r) < 2:
        return np.nan
    std = r.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(r.mean() / std * np.sqrt(MONTHS_PER_YEAR))


def sortino_ratio(monthly_returns: pd.Series) -> float:
    """Annualized Sortino with rf = 0 (downside deviation of negative months)."""
    r = monthly_returns.dropna()
    if len(r) < 2:
        return np.nan
    downside = r[r < 0]
    if len(downside) == 0:
        return np.inf
    downside_dev = np.sqrt((downside ** 2).mean())
    if downside_dev == 0:
        return np.nan
    return float(r.mean() / downside_dev * np.sqrt(MONTHS_PER_YEAR))


def max_drawdown(monthly_returns: pd.Series) -> float:
    """Max peak-to-trough drawdown of the compounded equity curve (negative number)."""
    r = monthly_returns.dropna()
    if len(r) == 0:
        return np.nan
    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def information_ratio(monthly_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Annualized IR of strategy minus benchmark on the overlapping months."""
    joined = pd.concat([monthly_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(joined) < 2:
        return np.nan
    active = joined.iloc[:, 0] - joined.iloc[:, 1]
    te = active.std(ddof=1)
    if te == 0 or np.isnan(te):
        return np.nan
    return float(active.mean() / te * np.sqrt(MONTHS_PER_YEAR))


def hit_rate_vs_benchmark(monthly_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Share of months the strategy beat the benchmark."""
    joined = pd.concat([monthly_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(joined) == 0:
        return np.nan
    return float((joined.iloc[:, 0] > joined.iloc[:, 1]).mean())


def summarize_returns(
    monthly_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    label: str = "",
) -> dict:
    """One row of summary stats for a return series."""
    r = monthly_returns.dropna()
    row = {
        "label": label,
        "months": int(len(r)),
        "start": r.index.min() if len(r) else pd.NaT,
        "end": r.index.max() if len(r) else pd.NaT,
        "total_return": float((1.0 + r).prod() - 1.0) if len(r) else np.nan,
        "annualized_return": annualized_return(r),
        "annualized_volatility": annualized_volatility(r),
        "sharpe_ratio": sharpe_ratio(r),
        "sortino_ratio": sortino_ratio(r),
        "max_drawdown": max_drawdown(r),
        "best_month": float(r.max()) if len(r) else np.nan,
        "worst_month": float(r.min()) if len(r) else np.nan,
        "pct_positive_months": float((r > 0).mean()) if len(r) else np.nan,
    }

    if benchmark_returns is not None:
        row["information_ratio_vs_benchmark"] = information_ratio(r, benchmark_returns)
        row["hit_rate_vs_benchmark"] = hit_rate_vs_benchmark(r, benchmark_returns)

    return row


def contiguous_folds(dates: list, n_folds: int) -> list[tuple[int, list]]:
    """Split an ordered list of dates into n contiguous, near-equal folds.

    Returns a list of (fold_number, dates_in_fold), fold numbers starting at 1.
    """
    n = len(dates)
    if n == 0 or n_folds <= 0:
        return []
    n_folds = min(n_folds, n)
    fold_sizes = [n // n_folds] * n_folds
    for i in range(n % n_folds):
        fold_sizes[i] += 1

    folds = []
    start = 0
    for fold_number, size in enumerate(fold_sizes, start=1):
        folds.append((fold_number, dates[start:start + size]))
        start += size

    return folds
