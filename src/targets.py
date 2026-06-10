import pandas as pd


def compute_forward_returns(monthly_prices: pd.DataFrame, horizon_months: int = 12) -> pd.DataFrame:
    """
    monthly_prices:
        rows = dates
        columns = tickers
        values = adjusted close prices

    returns:
        dataframe of forward returns over horizon_months
    """
    future_prices = monthly_prices.shift(-horizon_months)
    forward_returns = future_prices / monthly_prices - 1
    return forward_returns


def build_targets(monthly_prices: pd.DataFrame, benchmark: str = "SPY", horizon_months: int = 12) -> pd.DataFrame:
    """
    Builds absolute direction and benchmark outperformance targets.
    """

    forward_returns = compute_forward_returns(monthly_prices, horizon_months)

    if benchmark not in forward_returns.columns:
        raise ValueError(f"Benchmark {benchmark} not found in price data.")

    benchmark_forward_return = forward_returns[benchmark]

    rows = []

    for ticker in forward_returns.columns:
        if ticker == benchmark:
            continue

        for date in forward_returns.index:
            stock_ret = forward_returns.loc[date, ticker]
            bench_ret = benchmark_forward_return.loc[date]

            if pd.isna(stock_ret) or pd.isna(bench_ret):
                continue

            rows.append({
                "date": date,
                "ticker": ticker,
                "future_12m_return": stock_ret,
                "future_12m_spy_return": bench_ret,
                "target_abs_direction": int(stock_ret > 0),
                "target_outperform_spy": int(stock_ret > bench_ret),
            })

    targets = pd.DataFrame(rows)
    return targets