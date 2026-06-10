import os
import pandas as pd


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    prices_path = "data/processed/expanded_monthly_prices.parquet"
    output_path = "data/processed/expanded_realized_monthly_returns.parquet"
    report_path = "outputs/reports/week11_realized_monthly_returns_summary.txt"

    print("Loading expanded monthly prices...")
    prices = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    print("Computing realized next-month returns...")

    # This is the actual return from month t to month t+1.
    next_month_returns = prices.shift(-1) / prices - 1

    rows = []

    for ticker in next_month_returns.columns:
        for date in next_month_returns.index:
            ret = next_month_returns.loc[date, ticker]

            if pd.isna(ret):
                continue

            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "realized_next_1m_return": ret,
                }
            )

    realized = pd.DataFrame(rows)
    realized["date"] = pd.to_datetime(realized["date"])
    realized = realized.sort_values(["date", "ticker"]).reset_index(drop=True)

    realized.to_parquet(output_path)

    lines = []
    lines.append("Week 11 Realized Monthly Returns Summary")
    lines.append("=======================================")
    lines.append("")
    lines.append("Goal:")
    lines.append(
        "Create true next-month realized returns for realistic monthly portfolio backtesting."
    )
    lines.append("")
    lines.append("Input:")
    lines.append(prices_path)
    lines.append("")
    lines.append("Output:")
    lines.append(output_path)
    lines.append("")
    lines.append(f"Realized returns shape: {realized.shape}")
    lines.append(f"Ticker count: {realized['ticker'].nunique()}")
    lines.append(f"Date range: {realized['date'].min()} to {realized['date'].max()}")
    lines.append("")
    lines.append("Return summary:")
    lines.append(str(realized["realized_next_1m_return"].describe()))
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "This file allows the backtester to rank stocks at month t, hold them for one month, "
        "and measure the actual realized return from t to t+1."
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("")
    print("Saved:", output_path)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()