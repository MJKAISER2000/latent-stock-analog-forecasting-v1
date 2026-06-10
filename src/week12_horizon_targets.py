import os
import pandas as pd
import numpy as np


def compute_forward_returns(monthly_prices: pd.DataFrame, horizon_months: int) -> pd.DataFrame:
    future_prices = monthly_prices.shift(-horizon_months)
    return future_prices / monthly_prices - 1


def build_horizon_targets(monthly_prices: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    rows = []

    horizons = {
        "1m": 1,
        "12m": 12,
    }

    forward_returns_by_horizon = {
        name: compute_forward_returns(monthly_prices, months)
        for name, months in horizons.items()
    }

    for horizon_name, forward_returns in forward_returns_by_horizon.items():
        spy_forward = forward_returns[benchmark]

        temp_rows = []

        for ticker in forward_returns.columns:
            if ticker == benchmark:
                continue

            for date in forward_returns.index:
                stock_ret = forward_returns.loc[date, ticker]
                spy_ret = spy_forward.loc[date]

                if pd.isna(stock_ret) or pd.isna(spy_ret):
                    continue

                temp_rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        f"future_{horizon_name}_return": stock_ret,
                        f"future_{horizon_name}_spy_return": spy_ret,
                        f"future_{horizon_name}_excess_return": stock_ret - spy_ret,
                        f"target_outperform_spy_{horizon_name}": int(stock_ret > spy_ret),
                    }
                )

        temp = pd.DataFrame(temp_rows)

        # Cross-sectional top-quintile label by date.
        # 1 means the stock was in the top 20% of returns among available stocks that month.
        quintile_col = f"target_top_quintile_{horizon_name}"
        return_col = f"future_{horizon_name}_return"

        temp[quintile_col] = 0

        for date, group in temp.groupby("date"):
            cutoff = group[return_col].quantile(0.80)
            temp.loc[group.index, quintile_col] = (group[return_col] >= cutoff).astype(int)

        if horizon_name == "1m":
            final = temp
        else:
            final = final.merge(temp, on=["date", "ticker"], how="outer")

    final = final.sort_values(["date", "ticker"]).reset_index(drop=True)
    return final


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    prices_path = "data/processed/expanded_monthly_prices.parquet"
    output_path = "data/processed/week12_horizon_targets.parquet"
    report_path = "outputs/reports/week12_horizon_targets_summary.txt"

    print("Loading expanded monthly prices...")
    prices = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    print("Building 1-month and 12-month horizon targets...")
    targets = build_horizon_targets(prices, benchmark="SPY")

    targets.to_parquet(output_path)

    lines = []
    lines.append("Week 12 Horizon Targets Summary")
    lines.append("===============================")
    lines.append("")
    lines.append("Goal:")
    lines.append(
        "Create aligned prediction targets for both 1-month and 12-month horizons."
    )
    lines.append("")
    lines.append("Targets created:")
    lines.append("- future_1m_return")
    lines.append("- future_1m_spy_return")
    lines.append("- future_1m_excess_return")
    lines.append("- target_outperform_spy_1m")
    lines.append("- target_top_quintile_1m")
    lines.append("- future_12m_return")
    lines.append("- future_12m_spy_return")
    lines.append("- future_12m_excess_return")
    lines.append("- target_outperform_spy_12m")
    lines.append("- target_top_quintile_12m")
    lines.append("")
    lines.append(f"Output shape: {targets.shape}")
    lines.append(f"Ticker count: {targets['ticker'].nunique()}")
    lines.append(f"Date range: {targets['date'].min()} to {targets['date'].max()}")
    lines.append("")
    lines.append("1-month SPY outperformance target balance:")
    lines.append(str(targets["target_outperform_spy_1m"].value_counts(normalize=True)))
    lines.append("")
    lines.append("1-month top-quintile target balance:")
    lines.append(str(targets["target_top_quintile_1m"].value_counts(normalize=True)))
    lines.append("")
    lines.append("12-month SPY outperformance target balance:")
    lines.append(str(targets["target_outperform_spy_12m"].value_counts(normalize=True)))
    lines.append("")
    lines.append("12-month top-quintile target balance:")
    lines.append(str(targets["target_top_quintile_12m"].value_counts(normalize=True)))
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "This file lets us compare models trained for monthly prediction against models trained for longer-term prediction."
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("")
    print("Saved:", output_path)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()