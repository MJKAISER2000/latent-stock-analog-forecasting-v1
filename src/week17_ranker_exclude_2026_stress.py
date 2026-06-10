import os
import pandas as pd
import numpy as np


RETURN_COLS = {
    "top5": "top5_return",
    "top10": "top10_return",
    "top15": "top15_return",
    "top20": "top20_return",
    "top25": "top25_return",
    "top50": "top50_return",
}


def performance_stats(monthly_returns: pd.Series) -> dict:
    monthly_returns = monthly_returns.dropna()

    if len(monthly_returns) == 0:
        return {
            "months": 0,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_no_risk_free": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
            "return_over_abs_drawdown": np.nan,
        }

    total_return = (1 + monthly_returns).prod() - 1
    annualized_return = (1 + total_return) ** (12 / len(monthly_returns)) - 1
    annualized_volatility = monthly_returns.std() * np.sqrt(12)

    sharpe = np.nan
    if annualized_volatility != 0:
        sharpe = annualized_return / annualized_volatility

    cumulative = (1 + monthly_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = drawdown.min()

    win_rate = (monthly_returns > 0).mean()

    return_over_abs_drawdown = np.nan
    if max_drawdown != 0 and not pd.isna(max_drawdown):
        return_over_abs_drawdown = annualized_return / abs(max_drawdown)

    return {
        "months": len(monthly_returns),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_no_risk_free": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "return_over_abs_drawdown": return_over_abs_drawdown,
    }


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    curves_path = "outputs/tables/week17_ranker_topn_stress_curves.csv"

    curves = pd.read_csv(curves_path)
    curves["date"] = pd.to_datetime(curves["date"])

    periods = {
        "full_period": curves,
        "exclude_2026": curves[curves["date"] < "2026-01-01"],
        "only_2021_2024": curves[curves["date"] < "2025-01-01"],
        "only_2025_2026": curves[curves["date"] >= "2025-01-01"],
    }

    rows = []

    for period_name, period_df in periods.items():
        for label, return_col in RETURN_COLS.items():
            if return_col not in period_df.columns:
                continue

            stats = performance_stats(period_df[return_col])

            rows.append(
                {
                    "period": period_name,
                    "portfolio": label,
                    **stats,
                }
            )

    stats_df = pd.DataFrame(rows)

    output_path = "outputs/tables/week17_ranker_exclude_2026_stress_stats.csv"
    report_path = "outputs/reports/week17_ranker_exclude_2026_stress_summary.txt"

    stats_df.to_csv(output_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 17 Ranker Exclude-2026 Stress Test\n")
        f.write("======================================\n\n")
        f.write("Goal:\n")
        f.write("Test whether the LightGBM ranker remains strong before the 2026 explosion.\n\n")
        f.write(stats_df.to_string(index=False))
        f.write("\n")

    print("")
    print("Saved:", output_path)
    print("Saved:", report_path)
    print("")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()