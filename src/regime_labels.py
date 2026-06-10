import pandas as pd


def add_market_regime_labels(spy_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Creates simple market regime labels using SPY drawdowns.

    correction_regime = SPY is down 10% or more from previous high
    bear_regime = SPY is down 20% or more from previous high
    crash_regime = SPY is down 30% or more from previous high
    """

    if isinstance(spy_monthly, pd.Series):
        df = spy_monthly.to_frame("spy_price")
    else:
        df = spy_monthly.copy()
        if df.shape[1] == 1:
            df.columns = ["spy_price"]

    df = df.sort_index()

    df["rolling_peak"] = df["spy_price"].cummax()
    df["drawdown"] = df["spy_price"] / df["rolling_peak"] - 1

    df["correction_regime"] = (df["drawdown"] <= -0.10).astype(int)
    df["bear_regime"] = (df["drawdown"] <= -0.20).astype(int)
    df["crash_regime"] = (df["drawdown"] <= -0.30).astype(int)

    return df[
        [
            "spy_price",
            "drawdown",
            "correction_regime",
            "bear_regime",
            "crash_regime",
        ]
    ]