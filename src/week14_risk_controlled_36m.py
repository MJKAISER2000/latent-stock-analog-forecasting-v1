import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


RISK_CONFIGS = [
    {
        "name": "base_pure_36m_top10",
        "top_n": 10,
        "max_stock_vol_12m": None,
        "min_stock_drawdown": None,
        "max_position_vol_weight": False,
        "market_bear_cash_filter": False,
    },
    {
        "name": "vol_filter_12m_lt_0_15",
        "top_n": 10,
        "max_stock_vol_12m": 0.15,
        "min_stock_drawdown": None,
        "max_position_vol_weight": False,
        "market_bear_cash_filter": False,
    },
    {
        "name": "vol_filter_12m_lt_0_20",
        "top_n": 10,
        "max_stock_vol_12m": 0.20,
        "min_stock_drawdown": None,
        "max_position_vol_weight": False,
        "market_bear_cash_filter": False,
    },
    {
        "name": "drawdown_gt_minus_30",
        "top_n": 10,
        "max_stock_vol_12m": None,
        "min_stock_drawdown": -0.30,
        "max_position_vol_weight": False,
        "market_bear_cash_filter": False,
    },
    {
        "name": "drawdown_gt_minus_40",
        "top_n": 10,
        "max_stock_vol_12m": None,
        "min_stock_drawdown": -0.40,
        "max_position_vol_weight": False,
        "market_bear_cash_filter": False,
    },
    {
        "name": "inverse_vol_weighted",
        "top_n": 10,
        "max_stock_vol_12m": None,
        "min_stock_drawdown": None,
        "max_position_vol_weight": True,
        "market_bear_cash_filter": False,
    },
    {
        "name": "bear_market_cash_filter",
        "top_n": 10,
        "max_stock_vol_12m": None,
        "min_stock_drawdown": None,
        "max_position_vol_weight": False,
        "market_bear_cash_filter": True,
    },
    {
        "name": "combined_vol20_drawdown40",
        "top_n": 10,
        "max_stock_vol_12m": 0.20,
        "min_stock_drawdown": -0.40,
        "max_position_vol_weight": False,
        "market_bear_cash_filter": False,
    },
    {
        "name": "combined_vol_weight_bear_cash",
        "top_n": 10,
        "max_stock_vol_12m": None,
        "min_stock_drawdown": None,
        "max_position_vol_weight": True,
        "market_bear_cash_filter": True,
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

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_no_risk_free": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }


def load_monthly_returns() -> pd.DataFrame:
    prices_path = "data/processed/expanded_monthly_prices.parquet"

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
                    "ticker": ticker,
                    "monthly_return": ret,
                }
            )

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])

    return out


def load_features() -> pd.DataFrame:
    path = "data/processed/week12_aligned_modeling_dataset.parquet"

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip()

    needed = [
        "date",
        "ticker",
        "vol_12m",
        "stock_drawdown",
        "bear_regime",
        "correction_regime",
        "crash_regime",
    ]

    existing = [c for c in needed if c in df.columns]

    return df[existing].copy()


def load_h36_predictions() -> pd.DataFrame:
    path = "outputs/tables/week13_predictions_horizon_36m.csv"

    pred = pd.read_csv(path)
    pred.columns = pred.columns.str.strip()
    pred["date"] = pd.to_datetime(pred["date"])
    pred["ticker"] = pred["ticker"].astype(str).str.strip()

    pred = pred.rename(
        columns={"predicted_prob_outperform_h": "score_36m"}
    )

    return pred


def build_filtered_baskets(
    pred: pd.DataFrame,
    features: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    merged = pred.merge(features, on=["date", "ticker"], how="left")

    rows = []

    for date, group in merged.groupby("date"):
        g = group.copy()

        if config["max_stock_vol_12m"] is not None and "vol_12m" in g.columns:
            g = g[g["vol_12m"] <= config["max_stock_vol_12m"]]

        if config["min_stock_drawdown"] is not None and "stock_drawdown" in g.columns:
            g = g[g["stock_drawdown"] >= config["min_stock_drawdown"]]

        if len(g) < config["top_n"]:
            g = group.copy()

        selected = g.sort_values("score_36m", ascending=False).head(config["top_n"])

        if "bear_regime" in selected.columns:
            bear_regime = int(selected["bear_regime"].max())
        else:
            bear_regime = 0

        rows.append(
            {
                "signal_date": date,
                "strategy": config["name"],
                "tickers": selected["ticker"].tolist(),
                "selected_tickers": ", ".join(selected["ticker"].tolist()),
                "bear_regime": bear_regime,
            }
        )

    return pd.DataFrame(rows)


def basket_return_equal_weight(
    month_rets: pd.DataFrame,
) -> float:
    return month_rets["monthly_return"].mean()


def basket_return_inverse_vol_weighted(
    month_rets: pd.DataFrame,
    feature_slice: pd.DataFrame,
) -> float:
    merged = month_rets.merge(
        feature_slice[["ticker", "vol_12m"]],
        on="ticker",
        how="left",
    )

    merged["vol_12m"] = pd.to_numeric(merged["vol_12m"], errors="coerce")
    merged["vol_12m"] = merged["vol_12m"].replace(0, np.nan)

    if merged["vol_12m"].isna().all():
        return merged["monthly_return"].mean()

    merged["inv_vol"] = 1 / merged["vol_12m"]
    merged["inv_vol"] = merged["inv_vol"].replace([np.inf, -np.inf], np.nan)
    merged["inv_vol"] = merged["inv_vol"].fillna(merged["inv_vol"].median())

    if merged["inv_vol"].sum() == 0 or pd.isna(merged["inv_vol"].sum()):
        return merged["monthly_return"].mean()

    merged["weight"] = merged["inv_vol"] / merged["inv_vol"].sum()

    return float((merged["monthly_return"] * merged["weight"]).sum())


def build_overlapping_returns(
    baskets: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    features: pd.DataFrame,
    config: dict,
    horizon_months: int = 36,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    all_dates = sorted(pd.to_datetime(monthly_returns["date"].unique()))

    basket_rows = []

    for _, row in baskets.iterrows():
        signal_date = pd.to_datetime(row["signal_date"])
        tickers = row["tickers"]

        future_dates = [d for d in all_dates if d > signal_date]
        holding_dates = future_dates[:horizon_months]

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

            if config["max_position_vol_weight"]:
                basket_return = basket_return_inverse_vol_weighted(
                    month_rets=month_rets,
                    feature_slice=signal_features,
                )
            else:
                basket_return = basket_return_equal_weight(month_rets)

            if holding_idx == 1:
                basket_return = basket_return - transaction_cost

            # Simple defensive overlay:
            # if basket was opened in a bear regime, hold 50% cash / 50% strategy.
            # Cash return assumed 0 for now.
            if config["market_bear_cash_filter"] and row["bear_regime"] == 1:
                basket_return = 0.5 * basket_return

            basket_rows.append(
                {
                    "date": hold_date,
                    "signal_date": signal_date,
                    "basket_return": basket_return,
                    "active_tickers": ", ".join(tickers),
                }
            )

    basket_returns = pd.DataFrame(basket_rows)

    if basket_returns.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "strategy_monthly_return",
                "active_basket_count",
            ]
        )

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


def build_equal_weight_benchmark(monthly_returns: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(dates)

    return (
        monthly_returns[monthly_returns["date"].isin(dates)]
        .groupby("date")["monthly_return"]
        .mean()
        .reset_index()
        .rename(columns={"monthly_return": "equal_weight_monthly_return"})
    )


def build_spy_benchmark(monthly_returns: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(dates)

    spy = monthly_returns[
        (monthly_returns["ticker"] == "SPY")
        & (monthly_returns["date"].isin(dates))
    ].copy()

    return spy[["date", "monthly_return"]].rename(
        columns={"monthly_return": "spy_monthly_return"}
    )


def add_cumulative(df: pd.DataFrame, return_col: str, cumulative_col: str) -> pd.DataFrame:
    df = df.copy()
    df[cumulative_col] = (1 + df[return_col]).cumprod()
    return df


def plot_curves(curves: pd.DataFrame, output_path: str):
    plt.figure(figsize=(11, 7))

    for col in curves.columns:
        if col == "date":
            continue

        if col.endswith("_cumulative"):
            plt.plot(
                curves["date"],
                curves[col],
                label=col.replace("_cumulative", ""),
            )

    plt.title("Week 14 Risk-Controlled 36M Model")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print("Saved plot:", output_path)


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    print("Loading 36-month predictions...")
    pred = load_h36_predictions()

    print("Loading risk features...")
    features = load_features()

    print("Loading monthly returns...")
    monthly_returns = load_monthly_returns()

    all_stats = []
    all_curves = None
    all_holdings = []

    for config in RISK_CONFIGS:
        print("")
        print("=" * 80)
        print("Backtesting:", config["name"])

        baskets = build_filtered_baskets(
            pred=pred,
            features=features,
            config=config,
        )

        strategy = build_overlapping_returns(
            baskets=baskets,
            monthly_returns=monthly_returns,
            features=features,
            config=config,
            horizon_months=36,
            transaction_cost=0.001,
        )

        if strategy.empty:
            continue

        stats = {
            "strategy": config["name"],
            **performance_stats(strategy["strategy_monthly_return"]),
            "avg_active_basket_count": strategy["active_basket_count"].mean(),
        }

        all_stats.append(stats)

        curve = strategy[["date", "strategy_monthly_return"]].rename(
            columns={
                "strategy_monthly_return": f"{config['name']}_return",
            }
        )

        curve = add_cumulative(
            curve,
            f"{config['name']}_return",
            f"{config['name']}_cumulative",
        )

        if all_curves is None:
            all_curves = curve
        else:
            all_curves = all_curves.merge(curve, on="date", how="outer")

        temp_holdings = baskets.copy()
        temp_holdings["config_name"] = config["name"]
        all_holdings.append(temp_holdings)

    if all_curves is None:
        raise ValueError("No curves were created.")

    all_curves = all_curves.sort_values("date").reset_index(drop=True)

    eq = build_equal_weight_benchmark(monthly_returns, all_curves["date"])
    spy = build_spy_benchmark(monthly_returns, all_curves["date"])

    all_curves = all_curves.merge(eq, on="date", how="left")
    all_curves = all_curves.merge(spy, on="date", how="left")

    all_curves = add_cumulative(
        all_curves,
        "equal_weight_monthly_return",
        "equal_weight_cumulative",
    )

    all_curves = add_cumulative(
        all_curves,
        "spy_monthly_return",
        "spy_cumulative",
    )

    benchmark_rows = [
        {
            "strategy": "equal_weight",
            **performance_stats(all_curves["equal_weight_monthly_return"]),
            "avg_active_basket_count": np.nan,
        },
        {
            "strategy": "spy",
            **performance_stats(all_curves["spy_monthly_return"]),
            "avg_active_basket_count": np.nan,
        },
    ]

    stats_df = pd.DataFrame(all_stats + benchmark_rows)
    holdings_df = pd.concat(all_holdings, ignore_index=True)

    stats_path = "outputs/tables/week14_risk_controlled_36m_stats.csv"
    curves_path = "outputs/tables/week14_risk_controlled_36m_curves.csv"
    holdings_path = "outputs/tables/week14_risk_controlled_36m_holdings.csv"
    figure_path = "outputs/figures/week14_risk_controlled_36m_equity_curves.png"
    report_path = "outputs/reports/week14_risk_controlled_36m_summary.txt"

    stats_df.to_csv(stats_path, index=False)
    all_curves.to_csv(curves_path, index=False)
    holdings_df.to_csv(holdings_path, index=False)

    plot_cols = ["date"]

    preferred = [
        "base_pure_36m_top10_cumulative",
        "inverse_vol_weighted_cumulative",
        "bear_market_cash_filter_cumulative",
        "combined_vol_weight_bear_cash_cumulative",
        "drawdown_gt_minus_40_cumulative",
        "vol_filter_12m_lt_0_20_cumulative",
        "equal_weight_cumulative",
        "spy_cumulative",
    ]

    for col in preferred:
        if col in all_curves.columns:
            plot_cols.append(col)

    plot_curves(all_curves[plot_cols], figure_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 14 Risk-Controlled 36M Model Summary\n")
        f.write("========================================\n\n")
        f.write("Goal:\n")
        f.write(
            "Test whether risk controls can reduce drawdown while preserving most of the return "
            "from the pure 36-month top-10 structural model.\n\n"
        )
        f.write("Risk controls tested:\n")
        f.write("- volatility filters\n")
        f.write("- drawdown filters\n")
        f.write("- inverse-volatility weighting\n")
        f.write("- bear-market cash overlay\n")
        f.write("- combined risk filters\n\n")
        f.write("Performance stats:\n")
        f.write(
            stats_df.sort_values("annualized_return", ascending=False)
            .to_string(index=False)
        )
        f.write("\n")

    print("")
    print("Saved:", stats_path)
    print("Saved:", curves_path)
    print("Saved:", holdings_path)
    print("Saved:", figure_path)
    print("Saved:", report_path)
    print("")
    print("Top strategies:")
    print(
        stats_df.sort_values("annualized_return", ascending=False)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()