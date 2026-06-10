import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


MODEL_CONFIGS = [
    {"name": "h1_top5", "horizon": 1, "top_n": 5},
    {"name": "h12_top5", "horizon": 12, "top_n": 5},
    {"name": "h24_top5", "horizon": 24, "top_n": 5},
    {"name": "h36_top5", "horizon": 36, "top_n": 5},
    {"name": "h36_top10", "horizon": 36, "top_n": 10},
]

RISK_VARIANTS = [
    {"name": "base", "vol_filter": None, "inverse_vol": False, "bear_cash": False},
    {"name": "vol20", "vol_filter": 0.20, "inverse_vol": False, "bear_cash": False},
    {"name": "inverse_vol", "vol_filter": None, "inverse_vol": True, "bear_cash": False},
    {"name": "bear_cash", "vol_filter": None, "inverse_vol": False, "bear_cash": True},
    {"name": "vol20_inverse_vol", "vol_filter": 0.20, "inverse_vol": True, "bear_cash": False},
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
    out["ticker"] = out["ticker"].astype(str).str.strip()

    return out


def load_features() -> pd.DataFrame:
    path = "data/processed/week12_aligned_modeling_dataset.parquet"

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip()

    wanted = [
        "date",
        "ticker",
        "vol_12m",
        "stock_drawdown",
        "bear_regime",
        "correction_regime",
        "crash_regime",
    ]

    existing = [c for c in wanted if c in df.columns]

    out = df[existing].copy()

    if "vol_12m" in out.columns:
        out["vol_12m"] = pd.to_numeric(out["vol_12m"], errors="coerce")

    if "stock_drawdown" in out.columns:
        out["stock_drawdown"] = pd.to_numeric(out["stock_drawdown"], errors="coerce")

    return out


def load_predictions(horizon: int) -> pd.DataFrame:
    path = f"outputs/tables/week13_predictions_horizon_{horizon}m.csv"

    pred = pd.read_csv(path)
    pred.columns = pred.columns.str.strip()
    pred["date"] = pd.to_datetime(pred["date"])
    pred["ticker"] = pred["ticker"].astype(str).str.strip()

    pred = pred.rename(columns={"predicted_prob_outperform_h": "score"})

    return pred


def build_baskets(
    pred: pd.DataFrame,
    features: pd.DataFrame,
    top_n: int,
    risk_variant: dict,
    model_name: str,
) -> pd.DataFrame:
    merged = pred.merge(features, on=["date", "ticker"], how="left")

    rows = []

    for date, group in merged.groupby("date"):
        g = group.copy()

        if risk_variant["vol_filter"] is not None and "vol_12m" in g.columns:
            filtered = g[g["vol_12m"] <= risk_variant["vol_filter"]].copy()

            # If filter is too strict, fall back to original group.
            if len(filtered) >= top_n:
                g = filtered

        selected = g.sort_values("score", ascending=False).head(top_n)

        bear_regime = 0
        if "bear_regime" in selected.columns:
            bear_regime = int(selected["bear_regime"].fillna(0).max())

        rows.append(
            {
                "signal_date": date,
                "model_name": model_name,
                "risk_variant": risk_variant["name"],
                "top_n": top_n,
                "tickers": selected["ticker"].tolist(),
                "selected_tickers": ", ".join(selected["ticker"].tolist()),
                "bear_regime": bear_regime,
            }
        )

    return pd.DataFrame(rows)


def basket_return_equal_weight(month_rets: pd.DataFrame) -> float:
    return float(month_rets["monthly_return"].mean())


def basket_return_inverse_vol_weighted(
    month_rets: pd.DataFrame,
    signal_features: pd.DataFrame,
) -> float:
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
    risk_variant: dict,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    all_dates = sorted(pd.to_datetime(monthly_returns["date"].unique()))

    basket_rows = []

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

            if risk_variant["inverse_vol"]:
                basket_return = basket_return_inverse_vol_weighted(
                    month_rets=month_rets,
                    signal_features=signal_features,
                )
            else:
                basket_return = basket_return_equal_weight(month_rets)

            if holding_idx == 1:
                basket_return = basket_return - transaction_cost

            if risk_variant["bear_cash"] and row["bear_regime"] == 1:
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


def plot_top_curves(curves: pd.DataFrame, stats: pd.DataFrame, output_path: str):
    plt.figure(figsize=(11, 7))

    top_strategy_names = (
        stats[~stats["strategy"].isin(["spy", "equal_weight"])]
        .sort_values("sharpe_no_risk_free", ascending=False)
        .head(6)["strategy"]
        .tolist()
    )

    for strategy in top_strategy_names:
        col = f"{strategy}_cumulative"
        if col in curves.columns:
            plt.plot(curves["date"], curves[col], label=strategy)

    if "spy_cumulative" in curves.columns:
        plt.plot(curves["date"], curves["spy_cumulative"], label="SPY")

    if "equal_weight_cumulative" in curves.columns:
        plt.plot(curves["date"], curves["equal_weight_cumulative"], label="Equal Weight")

    plt.title("Week 14 Cross-Horizon Risk-Control Comparison")
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

    print("Loading monthly returns...")
    monthly_returns = load_monthly_returns()

    print("Loading risk features...")
    features = load_features()

    all_stats = []
    all_curves = None
    all_holdings = []

    for model_config in MODEL_CONFIGS:
        horizon = model_config["horizon"]
        top_n = model_config["top_n"]
        model_name = model_config["name"]

        print("")
        print("=" * 80)
        print(f"Loading predictions for {model_name}")

        pred = load_predictions(horizon)

        for risk_variant in RISK_VARIANTS:
            strategy_name = f"{model_name}_{risk_variant['name']}"

            print("Backtesting:", strategy_name)

            baskets = build_baskets(
                pred=pred,
                features=features,
                top_n=top_n,
                risk_variant=risk_variant,
                model_name=model_name,
            )

            strategy = build_overlapping_returns(
                baskets=baskets,
                monthly_returns=monthly_returns,
                features=features,
                holding_horizon_months=horizon,
                risk_variant=risk_variant,
                transaction_cost=0.001,
            )

            if strategy.empty:
                continue

            stats = {
                "strategy": strategy_name,
                "model": model_name,
                "horizon_months": horizon,
                "top_n": top_n,
                "risk_variant": risk_variant["name"],
                **performance_stats(strategy["strategy_monthly_return"]),
                "avg_active_basket_count": strategy["active_basket_count"].mean(),
            }

            all_stats.append(stats)

            curve = strategy[["date", "strategy_monthly_return"]].rename(
                columns={
                    "strategy_monthly_return": f"{strategy_name}_return",
                }
            )

            curve = add_cumulative(
                curve,
                f"{strategy_name}_return",
                f"{strategy_name}_cumulative",
            )

            if all_curves is None:
                all_curves = curve
            else:
                all_curves = all_curves.merge(curve, on="date", how="outer")

            temp_holdings = baskets.copy()
            temp_holdings["strategy"] = strategy_name
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
            "model": "benchmark",
            "horizon_months": "benchmark",
            "top_n": "all",
            "risk_variant": "benchmark",
            **performance_stats(all_curves["equal_weight_monthly_return"]),
            "avg_active_basket_count": np.nan,
        },
        {
            "strategy": "spy",
            "model": "benchmark",
            "horizon_months": "benchmark",
            "top_n": "spy",
            "risk_variant": "benchmark",
            **performance_stats(all_curves["spy_monthly_return"]),
            "avg_active_basket_count": np.nan,
        },
    ]

    stats_df = pd.DataFrame(all_stats + benchmark_rows)
    holdings_df = pd.concat(all_holdings, ignore_index=True)

    stats_path = "outputs/tables/week14_cross_horizon_risk_stats.csv"
    curves_path = "outputs/tables/week14_cross_horizon_risk_curves.csv"
    holdings_path = "outputs/tables/week14_cross_horizon_risk_holdings.csv"
    figure_path = "outputs/figures/week14_cross_horizon_risk_equity_curves.png"
    report_path = "outputs/reports/week14_cross_horizon_risk_summary.txt"

    stats_df.to_csv(stats_path, index=False)
    all_curves.to_csv(curves_path, index=False)
    holdings_df.to_csv(holdings_path, index=False)

    plot_top_curves(all_curves, stats_df, figure_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 14 Cross-Horizon Risk-Control Summary\n")
        f.write("=========================================\n\n")
        f.write("Goal:\n")
        f.write(
            "Apply simple risk-control layers across the best horizon models to see whether "
            "risk management changes which model is best.\n\n"
        )
        f.write("Models tested:\n")
        f.write(str(MODEL_CONFIGS))
        f.write("\n\n")
        f.write("Risk variants tested:\n")
        f.write(str(RISK_VARIANTS))
        f.write("\n\n")
        f.write("Top strategies by annualized return:\n")
        f.write(
            stats_df.sort_values("annualized_return", ascending=False)
            .head(20)
            .to_string(index=False)
        )
        f.write("\n\nTop strategies by Sharpe:\n")
        f.write(
            stats_df.sort_values("sharpe_no_risk_free", ascending=False)
            .head(20)
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
    print("Top strategies by annualized return:")
    print(
        stats_df.sort_values("annualized_return", ascending=False)
        .head(15)
        .to_string(index=False)
    )
    print("")
    print("Top strategies by Sharpe:")
    print(
        stats_df.sort_values("sharpe_no_risk_free", ascending=False)
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()