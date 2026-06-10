import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


CONFIGS = [
    {
        "name": "leadership_h1_top5_inverse_vol",
        "horizon": 1,
        "top_n": 5,
        "inverse_vol": True,
        "vol_filter": None,
    },
    {
        "name": "leadership_h1_top10_inverse_vol",
        "horizon": 1,
        "top_n": 10,
        "inverse_vol": True,
        "vol_filter": None,
    },
    {
        "name": "leadership_h36_top5_base",
        "horizon": 36,
        "top_n": 5,
        "inverse_vol": False,
        "vol_filter": None,
    },
    {
        "name": "leadership_h36_top10_base",
        "horizon": 36,
        "top_n": 10,
        "inverse_vol": False,
        "vol_filter": None,
    },
    {
        "name": "leadership_h36_top10_vol20",
        "horizon": 36,
        "top_n": 10,
        "inverse_vol": False,
        "vol_filter": 0.20,
    },
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
    path = "data/processed/week18_full500_leadership_modeling_dataset.parquet"

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    wanted = ["date", "ticker", "vol_12m"]
    out = df[wanted].copy()
    out["vol_12m"] = pd.to_numeric(out["vol_12m"], errors="coerce")

    return out


def load_predictions(horizon: int) -> pd.DataFrame:
    path = f"outputs/tables/week18_leadership_ranker_predictions_full500_{horizon}m.csv"

    pred = pd.read_csv(path)
    pred["date"] = pd.to_datetime(pred["date"])
    pred["ticker"] = pred["ticker"].astype(str).str.strip().str.upper()

    return pred


def build_baskets(pred: pd.DataFrame, features: pd.DataFrame, config: dict) -> pd.DataFrame:
    merged = pred.merge(features, on=["date", "ticker"], how="left")

    rows = []

    for date, group in merged.groupby("date"):
        g = group.copy()

        if config["vol_filter"] is not None:
            filtered = g[g["vol_12m"] <= config["vol_filter"]].copy()

            if len(filtered) >= config["top_n"]:
                g = filtered

        selected = g.sort_values("ranker_score", ascending=False).head(config["top_n"])

        rows.append(
            {
                "signal_date": date,
                "config": config["name"],
                "horizon": config["horizon"],
                "top_n": config["top_n"],
                "tickers": selected["ticker"].tolist(),
                "selected_tickers": ", ".join(selected["ticker"].tolist()),
            }
        )

    return pd.DataFrame(rows)


def basket_return_equal_weight(month_rets: pd.DataFrame) -> float:
    return float(month_rets["monthly_return"].mean())


def basket_return_inverse_vol_weighted(month_rets: pd.DataFrame, signal_features: pd.DataFrame) -> float:
    merged = month_rets.merge(
        signal_features[["ticker", "vol_12m"]],
        on="ticker",
        how="left",
    )

    merged["vol_12m"] = pd.to_numeric(merged["vol_12m"], errors="coerce")
    merged["vol_12m"] = merged["vol_12m"].replace(0, np.nan)

    if merged["vol_12m"].isna().all():
        return basket_return_equal_weight(merged)

    merged["inv_vol"] = 1 / merged["vol_12m"]
    merged["inv_vol"] = merged["inv_vol"].replace([np.inf, -np.inf], np.nan)
    merged["inv_vol"] = merged["inv_vol"].fillna(merged["inv_vol"].median())

    if merged["inv_vol"].sum() == 0 or pd.isna(merged["inv_vol"].sum()):
        return basket_return_equal_weight(merged)

    merged["weight"] = merged["inv_vol"] / merged["inv_vol"].sum()

    return float((merged["monthly_return"] * merged["weight"]).sum())


def build_overlapping_returns(
    baskets: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    features: pd.DataFrame,
    holding_horizon_months: int,
    inverse_vol: bool,
    transaction_cost: float,
) -> pd.DataFrame:
    all_dates = sorted(pd.to_datetime(monthly_returns["date"].unique()))

    rows = []

    for _, row in baskets.iterrows():
        signal_date = pd.to_datetime(row["signal_date"])
        tickers = row["tickers"]

        future_dates = [d for d in all_dates if d > signal_date]
        holding_dates = future_dates[:holding_horizon_months]

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

            if inverse_vol:
                basket_return = basket_return_inverse_vol_weighted(month_rets, signal_features)
            else:
                basket_return = basket_return_equal_weight(month_rets)

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

    if basket_returns.empty:
        return pd.DataFrame(columns=["date", "strategy_monthly_return", "active_basket_count"])

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


def build_benchmark(monthly_returns: pd.DataFrame, ticker: str, dates: pd.Series) -> pd.DataFrame:
    out = monthly_returns[
        (monthly_returns["ticker"] == ticker)
        & (monthly_returns["date"].isin(pd.to_datetime(dates)))
    ].copy()

    return out[["date", "monthly_return"]].rename(columns={"monthly_return": f"{ticker.lower()}_return"})


def plot_curves(curves: pd.DataFrame, output_path: str):
    plt.figure(figsize=(12, 7))

    cumulative_cols = [c for c in curves.columns if c.endswith("_cumulative")]

    for col in cumulative_cols:
        label = col.replace("_cumulative", "")
        plt.plot(curves["date"], curves[col], label=label)

    plt.title("Week 18 Leadership Ranker Backtest")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print("Saved plot:", output_path)


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    monthly_returns = load_monthly_returns()
    features = load_features()

    stats_rows = []
    curves = None
    holdings_rows = []

    for config in CONFIGS:
        print("")
        print("=" * 80)
        print("Backtesting:", config)

        pred = load_predictions(config["horizon"])

        baskets = build_baskets(pred, features, config)

        strategy = build_overlapping_returns(
            baskets=baskets,
            monthly_returns=monthly_returns,
            features=features,
            holding_horizon_months=config["horizon"],
            inverse_vol=config["inverse_vol"],
            transaction_cost=TRANSACTION_COST,
        )

        stats = {
            "strategy": f"week18_{config['name']}",
            "config": config["name"],
            "horizon": config["horizon"],
            "top_n": config["top_n"],
            "inverse_vol": config["inverse_vol"],
            "vol_filter": config["vol_filter"],
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

        baskets["strategy"] = f"week18_{config['name']}"
        holdings_rows.append(baskets)

    spy = build_benchmark(monthly_returns, "SPY", curves["date"])
    curves = curves.merge(spy, on="date", how="left")
    curves["spy_cumulative"] = (1 + curves["spy_return"]).cumprod()

    stats_rows.append(
        {
            "strategy": "spy",
            "config": "benchmark",
            "horizon": "benchmark",
            "top_n": "spy",
            "inverse_vol": False,
            "vol_filter": None,
            **performance_stats(curves["spy_return"]),
            "avg_active_basket_count": np.nan,
        }
    )

    stats_df = pd.DataFrame(stats_rows)
    holdings_df = pd.concat(holdings_rows, ignore_index=True)

    stats_path = "outputs/tables/week18_leadership_ranker_backtest_stats.csv"
    curves_path = "outputs/tables/week18_leadership_ranker_backtest_curves.csv"
    holdings_path = "outputs/tables/week18_leadership_ranker_backtest_holdings.csv"
    figure_path = "outputs/figures/week18_leadership_ranker_backtest_equity_curves.png"
    report_path = "outputs/reports/week18_leadership_ranker_backtest_summary.txt"

    stats_df.to_csv(stats_path, index=False)
    curves.to_csv(curves_path, index=False)
    holdings_df.to_csv(holdings_path, index=False)

    plot_curves(curves, figure_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 18 Leadership Ranker Backtest Summary\n")
        f.write("=========================================\n\n")
        f.write("Goal:\n")
        f.write("Backtest the ranker trained with sector and industry leadership features.\n\n")
        f.write(stats_df.sort_values("annualized_return", ascending=False).to_string(index=False))
        f.write("\n")

    print("")
    print("Saved:", stats_path)
    print("Saved:", curves_path)
    print("Saved:", holdings_path)
    print("Saved:", figure_path)
    print("Saved:", report_path)

    print("")
    print("RESULTS BY ANNUALIZED RETURN")
    print(stats_df.sort_values("annualized_return", ascending=False).to_string(index=False))

    print("")
    print("RESULTS BY SHARPE")
    print(stats_df.sort_values("sharpe_no_risk_free", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()