import os
import pandas as pd
import numpy as np


BASE_STRATEGY_RETURN_COL = "top5_return"

REGIME_RULES = [
    {
        "name": "base_no_filter",
        "cash_weight_when_off": 0.0,
        "rule": "always_on",
    },
    {
        "name": "tech_6m_under_spy_50cash",
        "cash_weight_when_off": 0.50,
        "rule": "tech_6m_under_spy",
    },
    {
        "name": "tech_6m_negative_50cash",
        "cash_weight_when_off": 0.50,
        "rule": "tech_6m_negative",
    },
    {
        "name": "tech_12m_under_spy_50cash",
        "cash_weight_when_off": 0.50,
        "rule": "tech_12m_under_spy",
    },
    {
        "name": "tech_12m_negative_50cash",
        "cash_weight_when_off": 0.50,
        "rule": "tech_12m_negative",
    },
    {
        "name": "tech_drawdown_20_50cash",
        "cash_weight_when_off": 0.50,
        "rule": "tech_drawdown_20",
    },
    {
        "name": "tech_6m_under_spy_100cash",
        "cash_weight_when_off": 1.00,
        "rule": "tech_6m_under_spy",
    },
]


def performance_stats(monthly_returns: pd.Series) -> dict:
    monthly_returns = monthly_returns.dropna()

    if len(monthly_returns) == 0:
        return {
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
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_no_risk_free": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "return_over_abs_drawdown": return_over_abs_drawdown,
    }


def load_base_strategy_returns() -> pd.DataFrame:
    path = "outputs/tables/week17_ranker_topn_stress_curves.csv"

    curves = pd.read_csv(path)
    curves["date"] = pd.to_datetime(curves["date"])

    out = curves[["date", BASE_STRATEGY_RETURN_COL]].copy()
    out = out.rename(columns={BASE_STRATEGY_RETURN_COL: "base_ranker_return"})

    return out


def load_prices() -> pd.DataFrame:
    path = "data/processed/week15_500_monthly_prices.parquet"

    prices = pd.read_parquet(path)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    return prices


def load_universe() -> pd.DataFrame:
    path = "data/external/week15_500_stock_universe.csv"

    universe = pd.read_csv(path)
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()
    universe["sector"] = universe["sector"].fillna("Unknown").astype(str)

    return universe


def build_tech_regime_features(prices: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    tech_tickers = universe.loc[
        universe["sector"] == "Information Technology", "ticker"
    ].tolist()

    tech_tickers = [t for t in tech_tickers if t in prices.columns]

    if "SPY" not in prices.columns:
        raise ValueError("SPY missing from prices.")

    monthly_returns = prices / prices.shift(1) - 1

    tech_monthly_return = monthly_returns[tech_tickers].mean(axis=1)
    spy_monthly_return = monthly_returns["SPY"]

    regime = pd.DataFrame(index=prices.index)
    regime["date"] = regime.index
    regime["tech_monthly_return"] = tech_monthly_return
    regime["spy_monthly_return"] = spy_monthly_return

    regime["tech_ret_6m"] = (1 + tech_monthly_return).rolling(6).apply(np.prod, raw=True) - 1
    regime["spy_ret_6m"] = prices["SPY"] / prices["SPY"].shift(6) - 1

    regime["tech_ret_12m"] = (1 + tech_monthly_return).rolling(12).apply(np.prod, raw=True) - 1
    regime["spy_ret_12m"] = prices["SPY"] / prices["SPY"].shift(12) - 1

    tech_index = (1 + tech_monthly_return.fillna(0)).cumprod()
    regime["tech_index"] = tech_index
    regime["tech_drawdown"] = tech_index / tech_index.cummax() - 1

    regime = regime.reset_index(drop=True)
    regime["date"] = pd.to_datetime(regime["date"])

    return regime


def apply_regime_rule(df: pd.DataFrame, rule: str) -> pd.Series:
    if rule == "always_on":
        return pd.Series(True, index=df.index)

    if rule == "tech_6m_under_spy":
        return df["tech_ret_6m"] >= df["spy_ret_6m"]

    if rule == "tech_6m_negative":
        return df["tech_ret_6m"] >= 0

    if rule == "tech_12m_under_spy":
        return df["tech_ret_12m"] >= df["spy_ret_12m"]

    if rule == "tech_12m_negative":
        return df["tech_ret_12m"] >= 0

    if rule == "tech_drawdown_20":
        return df["tech_drawdown"] > -0.20

    raise ValueError(f"Unknown rule: {rule}")


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    base = load_base_strategy_returns()
    prices = load_prices()
    universe = load_universe()

    regime = build_tech_regime_features(prices, universe)

    merged = base.merge(regime, on="date", how="left")

    stats_rows = []
    curves = merged[["date"]].copy()

    for rule in REGIME_RULES:
        temp = merged.copy()

        is_on = apply_regime_rule(temp, rule["rule"]).fillna(True)
        cash_weight = rule["cash_weight_when_off"]

        temp["risk_on"] = is_on.astype(int)

        # Cash assumed to return 0 for now.
        temp["filtered_return"] = np.where(
            temp["risk_on"] == 1,
            temp["base_ranker_return"],
            temp["base_ranker_return"] * (1 - cash_weight),
        )

        stats = {
            "rule_name": rule["name"],
            "rule": rule["rule"],
            "cash_weight_when_off": cash_weight,
            "risk_on_rate": temp["risk_on"].mean(),
            **performance_stats(temp["filtered_return"]),
        }

        stats_rows.append(stats)

        curves[f"{rule['name']}_return"] = temp["filtered_return"]
        curves[f"{rule['name']}_cumulative"] = (1 + temp["filtered_return"]).cumprod()

    stats_df = pd.DataFrame(stats_rows)

    stats_path = "outputs/tables/week18_tech_regime_filter_stats.csv"
    curves_path = "outputs/tables/week18_tech_regime_filter_curves.csv"
    regime_path = "outputs/tables/week18_tech_regime_features.csv"
    report_path = "outputs/reports/week18_tech_regime_filter_summary.txt"

    stats_df.to_csv(stats_path, index=False)
    curves.to_csv(curves_path, index=False)
    regime.to_csv(regime_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 18 Tech Regime Filter Summary\n")
        f.write("=================================\n\n")
        f.write("Goal:\n")
        f.write("Test whether reducing ranker exposure when technology is weak improves robustness.\n\n")
        f.write(stats_df.sort_values("return_over_abs_drawdown", ascending=False).to_string(index=False))
        f.write("\n")

    print("")
    print("Saved:", stats_path)
    print("Saved:", curves_path)
    print("Saved:", regime_path)
    print("Saved:", report_path)

    print("")
    print("RESULTS BY RETURN / DRAWDOWN")
    print(stats_df.sort_values("return_over_abs_drawdown", ascending=False).to_string(index=False))

    print("")
    print("RESULTS BY ANNUALIZED RETURN")
    print(stats_df.sort_values("annualized_return", ascending=False).to_string(index=False))

    print("")
    print("RESULTS BY SHARPE")
    print(stats_df.sort_values("sharpe_no_risk_free", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()