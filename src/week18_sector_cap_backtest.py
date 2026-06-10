import os
import pandas as pd
import numpy as np


CONFIGS = [
    {"name": "top5_uncapped", "top_n": 5, "max_per_sector": None},
    {"name": "top5_max2_sector", "top_n": 5, "max_per_sector": 2},
    {"name": "top10_uncapped", "top_n": 10, "max_per_sector": None},
    {"name": "top10_max3_sector", "top_n": 10, "max_per_sector": 3},
    {"name": "top10_max2_sector", "top_n": 10, "max_per_sector": 2},
]

TRANSACTION_COST = 0.001


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


def load_predictions() -> pd.DataFrame:
    path = "outputs/tables/week17_lgbm_ranker_predictions_week15_full500_1m.csv"

    pred = pd.read_csv(path)
    pred["date"] = pd.to_datetime(pred["date"])
    pred["ticker"] = pred["ticker"].astype(str).str.strip().str.upper()

    return pred


def load_universe_metadata() -> pd.DataFrame:
    path = "data/external/week15_500_stock_universe.csv"

    universe = pd.read_csv(path)
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()
    universe["sector"] = universe["sector"].fillna("Unknown").astype(str)
    universe["industry"] = universe["industry"].fillna("Unknown").astype(str)

    return universe[["ticker", "company", "sector", "industry"]]


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

    out = df[["date", "ticker", "vol_12m"]].copy()
    out["vol_12m"] = pd.to_numeric(out["vol_12m"], errors="coerce")

    return out


def select_with_sector_cap(group: pd.DataFrame, top_n: int, max_per_sector):
    ranked = group.sort_values("ranker_score", ascending=False)

    if max_per_sector is None:
        return ranked.head(top_n).copy()

    selected_rows = []
    sector_counts = {}

    for _, row in ranked.iterrows():
        sector = row["sector"]

        current_count = sector_counts.get(sector, 0)

        if current_count >= max_per_sector:
            continue

        selected_rows.append(row)
        sector_counts[sector] = current_count + 1

        if len(selected_rows) >= top_n:
            break

    selected = pd.DataFrame(selected_rows)

    return selected


def build_baskets(pred: pd.DataFrame, universe: pd.DataFrame, config: dict) -> pd.DataFrame:
    pred_meta = pred.merge(universe, on="ticker", how="left")
    pred_meta["sector"] = pred_meta["sector"].fillna("Unknown")

    rows = []

    for date, group in pred_meta.groupby("date"):
        selected = select_with_sector_cap(
            group=group,
            top_n=config["top_n"],
            max_per_sector=config["max_per_sector"],
        )

        rows.append(
            {
                "signal_date": date,
                "config": config["name"],
                "top_n": config["top_n"],
                "max_per_sector": config["max_per_sector"],
                "tickers": selected["ticker"].tolist(),
                "selected_tickers": ", ".join(selected["ticker"].tolist()),
                "selected_sectors": ", ".join(selected["sector"].tolist()),
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


def build_returns(
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
        holding_dates = future_dates[:1]

        signal_features = features[
            (features["date"] == signal_date)
            & (features["ticker"].isin(tickers))
        ].copy()

        for hold_date in holding_dates:
            month_rets = monthly_returns[
                (monthly_returns["date"] == hold_date)
                & (monthly_returns["ticker"].isin(tickers))
            ]

            if len(month_rets) == 0:
                continue

            basket_return = basket_return_inverse_vol_weighted(month_rets, signal_features)
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


def summarize_sector_exposure(baskets: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in baskets.iterrows():
        sectors = [s.strip() for s in str(row["selected_sectors"]).split(",") if s.strip()]

        for sector in sectors:
            rows.append(
                {
                    "config": row["config"],
                    "signal_date": row["signal_date"],
                    "sector": sector,
                }
            )

    exploded = pd.DataFrame(rows)

    if exploded.empty:
        return pd.DataFrame()

    summary = (
        exploded.groupby(["config", "sector"])
        .agg(selection_count=("sector", "count"))
        .reset_index()
    )

    summary["selection_share"] = summary.groupby("config")["selection_count"].transform(
        lambda x: x / x.sum()
    )

    summary = summary.sort_values(["config", "selection_count"], ascending=[True, False])

    return summary


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    pred = load_predictions()
    universe = load_universe_metadata()
    monthly_returns = load_monthly_returns()
    features = load_features()

    stats_rows = []
    curves = None
    holdings_rows = []
    sector_summaries = []

    for config in CONFIGS:
        print("")
        print("=" * 80)
        print("Testing config:", config)

        baskets = build_baskets(pred, universe, config)

        strategy = build_returns(
            baskets=baskets,
            monthly_returns=monthly_returns,
            features=features,
            transaction_cost=TRANSACTION_COST,
        )

        stats = {
            "config": config["name"],
            "top_n": config["top_n"],
            "max_per_sector": config["max_per_sector"],
            "transaction_cost": TRANSACTION_COST,
            "strategy": f"week18_sectorcap_{config['name']}",
            **performance_stats(strategy["strategy_monthly_return"]),
            "avg_active_basket_count": strategy["active_basket_count"].mean(),
        }

        stats_rows.append(stats)

        temp = strategy[["date", "strategy_monthly_return"]].rename(
            columns={"strategy_monthly_return": f"{config['name']}_return"}
        )
        temp[f"{config['name']}_cumulative"] = (1 + temp[f"{config['name']}_return"]).cumprod()

        if curves is None:
            curves = temp
        else:
            curves = curves.merge(temp, on="date", how="outer")

        baskets["strategy"] = f"week18_sectorcap_{config['name']}"
        holdings_rows.append(baskets)

        sector_summary = summarize_sector_exposure(baskets)
        if not sector_summary.empty:
            sector_summaries.append(sector_summary)

    stats_df = pd.DataFrame(stats_rows)
    holdings_df = pd.concat(holdings_rows, ignore_index=True)
    sector_summary_df = pd.concat(sector_summaries, ignore_index=True)

    stats_path = "outputs/tables/week18_sector_cap_backtest_stats.csv"
    curves_path = "outputs/tables/week18_sector_cap_backtest_curves.csv"
    holdings_path = "outputs/tables/week18_sector_cap_backtest_holdings.csv"
    sector_path = "outputs/tables/week18_sector_cap_exposure_summary.csv"
    report_path = "outputs/reports/week18_sector_cap_backtest_summary.txt"

    stats_df.to_csv(stats_path, index=False)
    curves.to_csv(curves_path, index=False)
    holdings_df.to_csv(holdings_path, index=False)
    sector_summary_df.to_csv(sector_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 18 Sector-Cap Ranker Backtest\n")
        f.write("==================================\n\n")
        f.write("Goal:\n")
        f.write(
            "Test whether sector constraints reduce Information Technology concentration while preserving ranker performance.\n\n"
        )
        f.write("Stats:\n")
        f.write(stats_df.to_string(index=False))
        f.write("\n\nSector exposure summary:\n")
        f.write(sector_summary_df.to_string(index=False))
        f.write("\n")

    print("")
    print("Saved:", stats_path)
    print("Saved:", curves_path)
    print("Saved:", holdings_path)
    print("Saved:", sector_path)
    print("Saved:", report_path)

    print("")
    print("PERFORMANCE")
    print(stats_df.to_string(index=False))

    print("")
    print("SECTOR EXPOSURE")
    print(sector_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()