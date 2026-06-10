import os
import pandas as pd
import numpy as np


TARGET_DATASET = "week15_full500"
TARGET_STRATEGY_NAME = "h1_top5_inverse_vol"
HORIZON_MONTHS = 1
TOP_N = 5

TRANSACTION_COSTS = [0.001, 0.0025, 0.005, 0.01, 0.02]


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


def load_monthly_returns() -> pd.DataFrame:
    prices_path = "data/processed/week15_500_monthly_prices.parquet"

    prices = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    monthly_returns = prices / prices.shift(1) - 1

    rows = []

    for ticker in monthly_returns.columns:
        for date in monthly_returns.index:
            ret = monthly_returns.loc[date, ticker]

            if pd.isna(ret):
                continue

            rows.append(
                {
                    "date": date,
                    "ticker": str(ticker).strip().upper(),
                    "monthly_return": ret,
                }
            )

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])

    return out


def load_features() -> pd.DataFrame:
    path = "data/processed/week15_full500_modeling_dataset.parquet"

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    wanted = ["date", "ticker", "vol_12m"]
    existing = [c for c in wanted if c in df.columns]

    out = df[existing].copy()
    out["vol_12m"] = pd.to_numeric(out["vol_12m"], errors="coerce")

    return out


def load_predictions() -> pd.DataFrame:
    path = "outputs/tables/week17_lgbm_ranker_predictions_week15_full500_1m.csv"

    pred = pd.read_csv(path)
    pred["date"] = pd.to_datetime(pred["date"])
    pred["ticker"] = pred["ticker"].astype(str).str.strip().str.upper()

    return pred


def build_baskets(pred: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows = []

    for date, group in pred.groupby("date"):
        selected = group.sort_values("ranker_score", ascending=False).head(top_n)

        rows.append(
            {
                "signal_date": date,
                "tickers": selected["ticker"].tolist(),
                "selected_tickers": ", ".join(selected["ticker"].tolist()),
            }
        )

    return pd.DataFrame(rows)


def basket_return_inverse_vol_weighted(month_rets: pd.DataFrame, signal_features: pd.DataFrame) -> float:
    merged = month_rets.merge(
        signal_features[["ticker", "vol_12m"]],
        on="ticker",
        how="left",
    )

    merged["vol_12m"] = pd.to_numeric(merged["vol_12m"], errors="coerce")
    merged["vol_12m"] = merged["vol_12m"].replace(0, np.nan)

    if merged["vol_12m"].isna().all():
        return float(merged["monthly_return"].mean())

    merged["inv_vol"] = 1 / merged["vol_12m"]
    merged["inv_vol"] = merged["inv_vol"].replace([np.inf, -np.inf], np.nan)
    merged["inv_vol"] = merged["inv_vol"].fillna(merged["inv_vol"].median())

    if merged["inv_vol"].sum() == 0 or pd.isna(merged["inv_vol"].sum()):
        return float(merged["monthly_return"].mean())

    merged["weight"] = merged["inv_vol"] / merged["inv_vol"].sum()

    return float((merged["monthly_return"] * merged["weight"]).sum())


def build_returns_for_cost(
    baskets: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    features: pd.DataFrame,
    transaction_cost: float,
) -> pd.DataFrame:
    all_dates = sorted(pd.to_datetime(monthly_returns["date"].unique()))

    rows = []

    for _, row in baskets.iterrows():
        signal_date = pd.to_datetime(row["signal_date"])
        tickers = row["tickers"]

        future_dates = [d for d in all_dates if d > signal_date]
        holding_dates = future_dates[:HORIZON_MONTHS]

        signal_features = features[
            (features["date"] == signal_date)
            & (features["ticker"].isin(tickers))
        ].copy()

        for holding_idx, hold_date in enumerate(holding_dates, start=1):
            month_rets = monthly_returns[
                (monthly_returns["date"] == hold_date)
                & (monthly_returns["ticker"].isin(tickers))
            ]

            if len(month_rets) == 0:
                continue

            basket_return = basket_return_inverse_vol_weighted(month_rets, signal_features)

            if holding_idx == 1:
                basket_return = basket_return - transaction_cost

            rows.append(
                {
                    "date": hold_date,
                    "signal_date": signal_date,
                    "basket_return": basket_return,
                }
            )

    basket_returns = pd.DataFrame(rows)

    strategy = (
        basket_returns.groupby("date")
        .agg(
            strategy_monthly_return=("basket_return", "mean"),
            active_basket_count=("basket_return", "count"),
        )
        .reset_index()
    )

    strategy["date"] = pd.to_datetime(strategy["date"])
    strategy = strategy.sort_values("date").reset_index(drop=True)

    return strategy


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    print("Loading predictions, returns, and features...")
    pred = load_predictions()
    monthly_returns = load_monthly_returns()
    features = load_features()

    baskets = build_baskets(pred, TOP_N)

    stats_rows = []
    curves = None

    for cost in TRANSACTION_COSTS:
        print(f"Testing transaction cost: {cost}")

        strategy = build_returns_for_cost(
            baskets=baskets,
            monthly_returns=monthly_returns,
            features=features,
            transaction_cost=cost,
        )

        label = f"tcost_{cost:.4f}".replace(".", "_")

        stats = {
            "transaction_cost": cost,
            "strategy": f"{TARGET_DATASET}_lgbm_ranker_{TARGET_STRATEGY_NAME}_{label}",
            **performance_stats(strategy["strategy_monthly_return"]),
            "avg_active_basket_count": strategy["active_basket_count"].mean(),
        }

        stats_rows.append(stats)

        temp = strategy[["date", "strategy_monthly_return"]].rename(
            columns={"strategy_monthly_return": f"{label}_return"}
        )
        temp[f"{label}_cumulative"] = (1 + temp[f"{label}_return"]).cumprod()

        if curves is None:
            curves = temp
        else:
            curves = curves.merge(temp, on="date", how="outer")

    stats_df = pd.DataFrame(stats_rows)

    stats_path = "outputs/tables/week17_ranker_transaction_cost_stress_stats.csv"
    curves_path = "outputs/tables/week17_ranker_transaction_cost_stress_curves.csv"
    report_path = "outputs/reports/week17_ranker_transaction_cost_stress_summary.txt"

    stats_df.to_csv(stats_path, index=False)
    curves.to_csv(curves_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 17 Ranker Transaction Cost Stress Test\n")
        f.write("==========================================\n\n")
        f.write(f"Target dataset: {TARGET_DATASET}\n")
        f.write(f"Target strategy: {TARGET_STRATEGY_NAME}\n")
        f.write(f"Transaction costs tested: {TRANSACTION_COSTS}\n\n")
        f.write(stats_df.to_string(index=False))
        f.write("\n")

    print("")
    print("Saved:", stats_path)
    print("Saved:", curves_path)
    print("Saved:", report_path)
    print("")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()