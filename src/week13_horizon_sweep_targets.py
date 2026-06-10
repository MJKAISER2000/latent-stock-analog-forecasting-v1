import os
import pandas as pd
import numpy as np


HORIZONS = [1, 3, 6, 12, 24, 36]


def compute_forward_returns(monthly_prices: pd.DataFrame, horizon_months: int) -> pd.DataFrame:
    future_prices = monthly_prices.shift(-horizon_months)
    return future_prices / monthly_prices - 1


def build_targets_for_horizon(
    monthly_prices: pd.DataFrame,
    horizon: int,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    forward_returns = compute_forward_returns(monthly_prices, horizon)

    if benchmark not in forward_returns.columns:
        raise ValueError(f"Benchmark {benchmark} not found in prices.")

    spy_forward = forward_returns[benchmark]

    rows = []

    for ticker in forward_returns.columns:
        if ticker == benchmark:
            continue

        for date in forward_returns.index:
            stock_ret = forward_returns.loc[date, ticker]
            spy_ret = spy_forward.loc[date]

            if pd.isna(stock_ret) or pd.isna(spy_ret):
                continue

            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "horizon_months": horizon,
                    "future_h_return": stock_ret,
                    "future_h_spy_return": spy_ret,
                    "future_h_excess_return": stock_ret - spy_ret,
                    "target_outperform_spy_h": int(stock_ret > spy_ret),
                }
            )

    df = pd.DataFrame(rows)

    # Cross-sectional top-quintile target by date.
    df["target_top_quintile_h"] = 0

    for date, group in df.groupby("date"):
        cutoff = group["future_h_return"].quantile(0.80)
        df.loc[group.index, "target_top_quintile_h"] = (
            group["future_h_return"] >= cutoff
        ).astype(int)

    return df


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    prices_path = "data/processed/expanded_monthly_prices.parquet"
    output_path = "data/processed/week13_horizon_sweep_targets.parquet"
    report_path = "outputs/reports/week13_horizon_sweep_targets_summary.txt"

    print("Loading expanded monthly prices...")
    prices = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    all_targets = []

    print("Building horizon sweep targets...")

    for horizon in HORIZONS:
        print(f"Building targets for horizon={horizon} months...")
        targets_h = build_targets_for_horizon(
            monthly_prices=prices,
            horizon=horizon,
            benchmark="SPY",
        )
        all_targets.append(targets_h)

    targets = pd.concat(all_targets, ignore_index=True)
    targets = targets.sort_values(["horizon_months", "date", "ticker"]).reset_index(drop=True)

    targets.to_parquet(output_path)

    lines = []
    lines.append("Week 13 Horizon Sweep Targets Summary")
    lines.append("====================================")
    lines.append("")
    lines.append("Goal:")
    lines.append(
        "Create prediction targets for multiple horizons so we can test which horizon has the strongest tradable signal."
    )
    lines.append("")
    lines.append(f"Horizons tested: {HORIZONS}")
    lines.append("")
    lines.append(f"Output shape: {targets.shape}")
    lines.append(f"Ticker count: {targets['ticker'].nunique()}")
    lines.append(f"Date range: {targets['date'].min()} to {targets['date'].max()}")
    lines.append("")
    lines.append("Rows by horizon:")
    lines.append(str(targets["horizon_months"].value_counts().sort_index()))
    lines.append("")
    lines.append("SPY outperformance balance by horizon:")
    lines.append(
        str(
            targets.groupby("horizon_months")["target_outperform_spy_h"]
            .mean()
            .sort_index()
        )
    )
    lines.append("")
    lines.append("Top-quintile balance by horizon:")
    lines.append(
        str(
            targets.groupby("horizon_months")["target_top_quintile_h"]
            .mean()
            .sort_index()
        )
    )
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "These targets allow us to train horizon-specific models for 1, 3, 6, 12, 24, and 36 month predictions."
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("")
    print("Saved:", output_path)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()