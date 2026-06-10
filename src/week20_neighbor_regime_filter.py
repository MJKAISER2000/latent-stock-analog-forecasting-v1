import os
import pandas as pd
import numpy as np


CURVES_PATH = "outputs/tables/week20_neighbor_feature_ablation_curves.csv"
PRICE_PATH = "data/processed/week15_500_monthly_prices.parquet"
UNIVERSE_PATH = "data/external/week15_500_stock_universe.csv"

OUTPUT_STATS_PATH = "outputs/tables/week20_neighbor_regime_filter_stats.csv"
OUTPUT_CURVES_PATH = "outputs/tables/week20_neighbor_regime_filter_curves.csv"
OUTPUT_REGIME_PATH = "outputs/tables/week20_neighbor_regime_features.csv"
REPORT_PATH = "outputs/reports/week20_neighbor_regime_filter_summary.txt"


BASE_RETURN_COLS = {
    "neighbor_only_top5": "neighbor_only_top5_return",
    "neighbor_only_top10": "neighbor_only_top10_return",
    "neighbor_only_top20": "neighbor_only_top20_return",
    "original_only_top20": "original_only_top20_return",
}


REGIME_RULES = [
    {
        "name": "base_no_filter",
        "rule": "always_on",
        "cash_weight_when_off": 0.0,
    },
    {
        "name": "tech_6m_under_spy_50cash",
        "rule": "tech_6m_under_spy",
        "cash_weight_when_off": 0.50,
    },
    {
        "name": "tech_6m_under_spy_100cash",
        "rule": "tech_6m_under_spy",
        "cash_weight_when_off": 1.00,
    },
    {
        "name": "tech_12m_under_spy_50cash",
        "rule": "tech_12m_under_spy",
        "cash_weight_when_off": 0.50,
    },
    {
        "name": "tech_12m_under_spy_100cash",
        "rule": "tech_12m_under_spy",
        "cash_weight_when_off": 1.00,
    },
    {
        "name": "tech_6m_negative_50cash",
        "rule": "tech_6m_negative",
        "cash_weight_when_off": 0.50,
    },
    {
        "name": "tech_6m_negative_100cash",
        "rule": "tech_6m_negative",
        "cash_weight_when_off": 1.00,
    },
    {
        "name": "tech_drawdown_20_50cash",
        "rule": "tech_drawdown_20",
        "cash_weight_when_off": 0.50,
    },
    {
        "name": "tech_drawdown_20_100cash",
        "rule": "tech_drawdown_20",
        "cash_weight_when_off": 1.00,
    },
]


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


def load_curves() -> pd.DataFrame:
    curves = pd.read_csv(CURVES_PATH)
    curves["date"] = pd.to_datetime(curves["date"])
    curves = curves.sort_values("date").reset_index(drop=True)
    return curves


def load_prices() -> pd.DataFrame:
    prices = pd.read_parquet(PRICE_PATH)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices.columns = [str(c).strip().upper() for c in prices.columns]
    return prices


def load_universe() -> pd.DataFrame:
    universe = pd.read_csv(UNIVERSE_PATH)
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()
    universe["sector"] = universe["sector"].fillna("Unknown").astype(str)
    return universe


def build_tech_regime_features(prices: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    tech_tickers = universe.loc[
        universe["sector"] == "Information Technology",
        "ticker",
    ].tolist()

    tech_tickers = [t for t in tech_tickers if t in prices.columns]

    if len(tech_tickers) == 0:
        raise ValueError("No Information Technology tickers found in price data.")

    if "SPY" not in prices.columns:
        raise ValueError("SPY missing from prices.")

    monthly_returns = prices / prices.shift(1) - 1

    tech_monthly_return = monthly_returns[tech_tickers].mean(axis=1)
    spy_monthly_return = monthly_returns["SPY"]

    regime = pd.DataFrame(index=prices.index)
    regime["date"] = regime.index
    regime["tech_monthly_return"] = tech_monthly_return
    regime["spy_monthly_return"] = spy_monthly_return

    regime["tech_ret_6m"] = (
        (1 + tech_monthly_return).rolling(6).apply(np.prod, raw=True) - 1
    )
    regime["spy_ret_6m"] = prices["SPY"] / prices["SPY"].shift(6) - 1

    regime["tech_ret_12m"] = (
        (1 + tech_monthly_return).rolling(12).apply(np.prod, raw=True) - 1
    )
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

    if rule == "tech_12m_under_spy":
        return df["tech_ret_12m"] >= df["spy_ret_12m"]

    if rule == "tech_6m_negative":
        return df["tech_ret_6m"] >= 0

    if rule == "tech_drawdown_20":
        return df["tech_drawdown"] > -0.20

    raise ValueError(f"Unknown rule: {rule}")


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    curves = load_curves()
    prices = load_prices()
    universe = load_universe()

    regime = build_tech_regime_features(prices, universe)
    merged = curves.merge(regime, on="date", how="left")

    print("Loaded curves:", curves.shape)
    print("Available curve columns:")
    print([c for c in curves.columns if c.endswith("_return")])

    stats_rows = []
    filtered_curves = merged[["date"]].copy()

    for portfolio_name, return_col in BASE_RETURN_COLS.items():
        if return_col not in merged.columns:
            print(f"Skipping missing return column: {return_col}")
            continue

        for rule in REGIME_RULES:
            temp = merged.copy()

            is_on = apply_regime_rule(temp, rule["rule"]).fillna(True)
            cash_weight = rule["cash_weight_when_off"]

            temp["risk_on"] = is_on.astype(int)

            temp["filtered_return"] = np.where(
                temp["risk_on"] == 1,
                temp[return_col],
                temp[return_col] * (1 - cash_weight),
            )

            stats = {
                "portfolio": portfolio_name,
                "base_return_col": return_col,
                "rule_name": rule["name"],
                "rule": rule["rule"],
                "cash_weight_when_off": cash_weight,
                "risk_on_rate": temp["risk_on"].mean(),
                **performance_stats(temp["filtered_return"]),
            }

            stats_rows.append(stats)

            curve_name = f"{portfolio_name}_{rule['name']}"
            filtered_curves[f"{curve_name}_return"] = temp["filtered_return"]
            filtered_curves[f"{curve_name}_cumulative"] = (
                1 + temp["filtered_return"]
            ).cumprod()

    stats_df = pd.DataFrame(stats_rows)

    stats_df.to_csv(OUTPUT_STATS_PATH, index=False)
    filtered_curves.to_csv(OUTPUT_CURVES_PATH, index=False)
    regime.to_csv(OUTPUT_REGIME_PATH, index=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Week 20 Latent Neighbor Regime Filter Summary\n")
        f.write("============================================\n\n")
        f.write("Goal:\n")
        f.write(
            "Apply tech/growth regime filters to the stock latent-neighbor ranker outputs.\n\n"
        )

        f.write("Top by return/drawdown:\n")
        f.write(
            stats_df.sort_values("return_over_abs_drawdown", ascending=False)
            .head(30)
            .to_string(index=False)
        )
        f.write("\n\nTop by annualized return:\n")
        f.write(
            stats_df.sort_values("annualized_return", ascending=False)
            .head(30)
            .to_string(index=False)
        )
        f.write("\n\nTop by Sharpe:\n")
        f.write(
            stats_df.sort_values("sharpe_no_risk_free", ascending=False)
            .head(30)
            .to_string(index=False)
        )
        f.write("\n")

    print("")
    print("Saved:", OUTPUT_STATS_PATH)
    print("Saved:", OUTPUT_CURVES_PATH)
    print("Saved:", OUTPUT_REGIME_PATH)
    print("Saved:", REPORT_PATH)

    print("")
    print("LATENT NEIGHBOR + REGIME FILTER — TOP BY RETURN / DRAWDOWN")
    print(
        stats_df.sort_values("return_over_abs_drawdown", ascending=False)
        .head(30)
        .to_string(index=False)
    )

    print("")
    print("LATENT NEIGHBOR + REGIME FILTER — TOP BY ANNUALIZED RETURN")
    print(
        stats_df.sort_values("annualized_return", ascending=False)
        .head(30)
        .to_string(index=False)
    )

    print("")
    print("LATENT NEIGHBOR + REGIME FILTER — TOP BY SHARPE")
    print(
        stats_df.sort_values("sharpe_no_risk_free", ascending=False)
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()