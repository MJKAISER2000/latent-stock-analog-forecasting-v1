import os
import pandas as pd
import numpy as np


CURVES_PATH = "outputs/tables/week20_neighbor_ensemble_curves.csv"
STATS_PATH = "outputs/tables/week20_neighbor_ensemble_stats.csv"

OUTPUT_YEARLY_PATH = "outputs/tables/week20_ensemble_diagnostics_yearly_stats.csv"
OUTPUT_MONTHLY_PATH = "outputs/tables/week20_ensemble_diagnostics_monthly_returns.csv"
OUTPUT_FULL_PATH = "outputs/tables/week20_ensemble_diagnostics_full_stats.csv"
REPORT_PATH = "outputs/reports/week20_ensemble_diagnostics_summary.txt"


SELECTED_MODELS = {
    "ensemble_70_30_drawdown_100cash": (
        "ensemble_70_original20_30_neighbor10_tech_drawdown_20_100cash_return"
    ),
    "ensemble_50_50_drawdown_100cash": (
        "ensemble_50_original20_50_neighbor10_tech_drawdown_20_100cash_return"
    ),
    "ensemble_30_70_drawdown_100cash": (
        "ensemble_30_original20_70_neighbor10_tech_drawdown_20_100cash_return"
    ),
    "ensemble_70_30_base": (
        "ensemble_70_original20_30_neighbor10_base_no_filter_return"
    ),
    "ensemble_50_50_base": (
        "ensemble_50_original20_50_neighbor10_base_no_filter_return"
    ),
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

    curves = pd.read_csv(CURVES_PATH)
    curves["date"] = pd.to_datetime(curves["date"])
    curves = curves.sort_values("date").reset_index(drop=True)
    curves["year"] = curves["date"].dt.year

    available_return_cols = [c for c in curves.columns if c.endswith("_return")]

    print("Available return columns:")
    for col in available_return_cols:
        print(col)

    selected = {}

    for model_name, col in SELECTED_MODELS.items():
        if col not in curves.columns:
            print(f"WARNING: missing selected model column: {model_name} -> {col}")
            continue

        selected[model_name] = col

    if len(selected) == 0:
        raise ValueError("No selected model columns found. Check column names in curves file.")

    full_rows = []
    yearly_rows = []
    monthly_rows = []

    for model_name, col in selected.items():
        full_stats = performance_stats(curves[col])

        full_rows.append(
            {
                "model": model_name,
                "return_col": col,
                **full_stats,
            }
        )

        for year, group in curves.groupby("year"):
            stats = performance_stats(group[col])

            yearly_rows.append(
                {
                    "model": model_name,
                    "return_col": col,
                    "year": year,
                    **stats,
                }
            )

        temp = curves[["date", "year", col]].copy()
        temp = temp.rename(columns={col: "monthly_return"})
        temp["model"] = model_name
        temp["return_col"] = col
        monthly_rows.append(temp)

    full_stats = pd.DataFrame(full_rows).sort_values(
        "return_over_abs_drawdown",
        ascending=False,
    )

    yearly_stats = pd.DataFrame(yearly_rows).sort_values(["model", "year"])
    monthly_returns = pd.concat(monthly_rows, ignore_index=True)

    best_months = monthly_returns.sort_values("monthly_return", ascending=False).head(30)
    worst_months = monthly_returns.sort_values("monthly_return", ascending=True).head(30)

    full_stats.to_csv(OUTPUT_FULL_PATH, index=False)
    yearly_stats.to_csv(OUTPUT_YEARLY_PATH, index=False)
    monthly_returns.to_csv(OUTPUT_MONTHLY_PATH, index=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Week 20 Ensemble Diagnostics Summary\n")
        f.write("===================================\n\n")

        f.write("Goal:\n")
        f.write(
            "Diagnose whether portfolio-level blending of original ranker and latent-neighbor ranker improves stability year by year.\n\n"
        )

        f.write("Full-period stats:\n")
        f.write(full_stats.to_string(index=False))
        f.write("\n\n")

        f.write("Yearly stats:\n")
        f.write(yearly_stats.to_string(index=False))
        f.write("\n\n")

        f.write("Worst months:\n")
        f.write(worst_months.to_string(index=False))
        f.write("\n\n")

        f.write("Best months:\n")
        f.write(best_months.to_string(index=False))
        f.write("\n")

    print("")
    print("Saved:", OUTPUT_FULL_PATH)
    print("Saved:", OUTPUT_YEARLY_PATH)
    print("Saved:", OUTPUT_MONTHLY_PATH)
    print("Saved:", REPORT_PATH)

    print("")
    print("FULL-PERIOD ENSEMBLE DIAGNOSTICS")
    print(full_stats.to_string(index=False))

    print("")
    print("YEARLY ENSEMBLE DIAGNOSTICS")
    print(yearly_stats.to_string(index=False))

    print("")
    print("WORST MONTHS")
    print(worst_months.to_string(index=False))

    print("")
    print("BEST MONTHS")
    print(best_months.to_string(index=False))


if __name__ == "__main__":
    main()