import os
import pandas as pd
import numpy as np

import lightgbm as lgb


DATASET_PATH = "data/processed/week19_full500_with_market_pca_latents.parquet"

PREDICTION_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]

HORIZON = 1
TOP_NS = [5, 10, 20]
TRANSACTION_COST = 0.001


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    non_feature_cols = [
        "date",
        "ticker",
        "company",

        "future_1m_return",
        "future_1m_spy_return",
        "future_1m_excess_return",
        "target_outperform_spy_1m",
        "target_top_quintile_1m",

        "future_36m_return",
        "future_36m_spy_return",
        "future_36m_excess_return",
        "target_outperform_spy_36m",
        "target_top_quintile_36m",
    ]

    return [c for c in df.columns if c not in non_feature_cols]


def make_ranking_label(df: pd.DataFrame, return_col: str) -> pd.Series:
    labels = pd.Series(index=df.index, dtype=float)

    for date, group in df.groupby("date"):
        ranks = group[return_col].rank(method="first", pct=True)

        relevance = pd.cut(
            ranks,
            bins=[0.0, 0.20, 0.40, 0.60, 0.80, 1.0],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True,
        ).astype(int)

        labels.loc[group.index] = relevance

    return labels.astype(int)


def group_sizes_by_date(df: pd.DataFrame) -> list[int]:
    return df.groupby("date").size().tolist()


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
    prices.columns = [str(c).strip().upper() for c in prices.columns]

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


def load_features_for_weights() -> pd.DataFrame:
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    if "vol_12m" not in df.columns:
        raise ValueError("vol_12m is missing from dataset.")

    out = df[["date", "ticker", "vol_12m"]].copy()
    out["vol_12m"] = pd.to_numeric(out["vol_12m"], errors="coerce")

    return out


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
        return float(merged["monthly_return"].mean())

    merged["inv_vol"] = 1 / merged["vol_12m"]
    merged["inv_vol"] = merged["inv_vol"].replace([np.inf, -np.inf], np.nan)
    merged["inv_vol"] = merged["inv_vol"].fillna(merged["inv_vol"].median())

    if merged["inv_vol"].sum() == 0 or pd.isna(merged["inv_vol"].sum()):
        return float(merged["monthly_return"].mean())

    merged["weight"] = merged["inv_vol"] / merged["inv_vol"].sum()

    return float((merged["monthly_return"] * merged["weight"]).sum())


def train_predict_for_year(
    df: pd.DataFrame,
    feature_cols: list[str],
    prediction_year: int,
) -> pd.DataFrame:
    print("")
    print("=" * 100)
    print(f"PCA latent walk-forward fold: train before {prediction_year}, predict {prediction_year}")

    train = df[df["date"].dt.year < prediction_year].copy()
    predict = df[df["date"].dt.year == prediction_year].copy()

    if len(train) == 0 or len(predict) == 0:
        print("Skipping year due to empty train/predict set.")
        return pd.DataFrame()

    train_dates = sorted(train["date"].unique())
    val_start_idx = int(len(train_dates) * 0.80)
    val_dates = set(train_dates[val_start_idx:])

    train_fit = train[~train["date"].isin(val_dates)].copy()
    val = train[train["date"].isin(val_dates)].copy()

    print("Train fit:", train_fit.shape)
    print("Validation:", val.shape)
    print("Predict:", predict.shape)

    safe_feature_names = [f"feature_{i}" for i in range(len(feature_cols))]

    X_train = train_fit[feature_cols].copy()
    X_val = val[feature_cols].copy()
    X_predict = predict[feature_cols].copy()

    X_train.columns = safe_feature_names
    X_val.columns = safe_feature_names
    X_predict.columns = safe_feature_names

    y_train = train_fit["ranking_label"].astype(int)
    y_val = val["ranking_label"].astype(int)

    train_group = group_sizes_by_date(train_fit)
    val_group = group_sizes_by_date(val)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    ranker.fit(
        X_train,
        y_train,
        group=train_group,
        eval_set=[(X_val, y_val)],
        eval_group=[val_group],
        eval_at=[5, 10, 25],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0),
        ],
    )

    scores = ranker.predict(X_predict)

    out = predict[
        [
            "date",
            "ticker",
            "future_1m_return",
            "future_1m_spy_return",
            "future_1m_excess_return",
            "target_outperform_spy_1m",
            "ranking_label",
        ]
    ].copy()

    out["prediction_year"] = prediction_year
    out["model"] = "walk_forward_lgbm_ranker_with_market_pca_latents"
    out["best_iteration"] = ranker.best_iteration_
    out["ranker_score"] = scores
    out["rank_by_date"] = out.groupby("date")["ranker_score"].rank(
        ascending=False,
        method="first",
    )

    return out


def build_baskets(predictions: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows = []

    for date, group in predictions.groupby("date"):
        selected = group.sort_values("ranker_score", ascending=False).head(top_n)

        rows.append(
            {
                "signal_date": date,
                "top_n": top_n,
                "tickers": selected["ticker"].tolist(),
                "selected_tickers": ", ".join(selected["ticker"].tolist()),
            }
        )

    return pd.DataFrame(rows)


def build_strategy_returns(
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

            basket_return = basket_return_inverse_vol_weighted(
                month_rets,
                signal_features,
            )
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


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    print("Loading PCA latent stock dataset...")
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    df = df.dropna(
        subset=[
            "future_1m_return",
            "future_1m_spy_return",
            "future_1m_excess_return",
            "target_outperform_spy_1m",
        ]
    ).copy()

    feature_cols = get_feature_columns(df)

    latent_cols = [c for c in feature_cols if c.startswith("market_pca_")]

    print("Total feature count:", len(feature_cols))
    print("Market PCA latent feature count:", len(latent_cols))
    print("Market PCA latent features:", latent_cols)

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    df[feature_cols] = df[feature_cols].fillna(0.0)

    df["ranking_label"] = make_ranking_label(df, "future_1m_return")

    all_predictions = []

    for year in PREDICTION_YEARS:
        pred_year = train_predict_for_year(df, feature_cols, year)

        if not pred_year.empty:
            all_predictions.append(pred_year)

    predictions = pd.concat(all_predictions, ignore_index=True)

    pred_path = "outputs/tables/week19_walk_forward_pca_latent_ranker_predictions.csv"
    predictions.to_csv(pred_path, index=False)

    monthly_returns = load_monthly_returns()
    features = load_features_for_weights()

    stats_rows = []
    curves = None
    holdings_rows = []

    for top_n in TOP_NS:
        baskets = build_baskets(predictions, top_n)

        strategy = build_strategy_returns(
            baskets=baskets,
            monthly_returns=monthly_returns,
            features=features,
            transaction_cost=TRANSACTION_COST,
        )

        stats = {
            "strategy": f"week19_walk_forward_pca_latent_top{top_n}_inverse_vol",
            "top_n": top_n,
            "transaction_cost": TRANSACTION_COST,
            **performance_stats(strategy["strategy_monthly_return"]),
            "avg_active_basket_count": strategy["active_basket_count"].mean(),
        }

        stats_rows.append(stats)

        temp = strategy[["date", "strategy_monthly_return"]].rename(
            columns={"strategy_monthly_return": f"top{top_n}_return"}
        )
        temp[f"top{top_n}_cumulative"] = (1 + temp[f"top{top_n}_return"]).cumprod()

        if curves is None:
            curves = temp
        else:
            curves = curves.merge(temp, on="date", how="outer")

        baskets["strategy"] = f"walk_forward_pca_latent_top{top_n}"
        holdings_rows.append(baskets)

    stats_df = pd.DataFrame(stats_rows)
    holdings_df = pd.concat(holdings_rows, ignore_index=True)

    stats_path = "outputs/tables/week19_walk_forward_pca_latent_ranker_stats.csv"
    curves_path = "outputs/tables/week19_walk_forward_pca_latent_ranker_curves.csv"
    holdings_path = "outputs/tables/week19_walk_forward_pca_latent_ranker_holdings.csv"
    report_path = "outputs/reports/week19_walk_forward_pca_latent_ranker_summary.txt"

    stats_df.to_csv(stats_path, index=False)
    curves.to_csv(curves_path, index=False)
    holdings_df.to_csv(holdings_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 19 Walk-Forward PCA Latent Market Ranker Summary\n")
        f.write("====================================================\n\n")
        f.write("Goal:\n")
        f.write(
            "Test whether adding latent market-state PCA features improves walk-forward ranker performance.\n\n"
        )
        f.write(f"Dataset: {DATASET_PATH}\n")
        f.write(f"Total feature count: {len(feature_cols)}\n")
        f.write(f"Market PCA latent feature count: {len(latent_cols)}\n")
        f.write(f"Market PCA latent features: {latent_cols}\n\n")
        f.write("Stats:\n")
        f.write(
            stats_df.sort_values("annualized_return", ascending=False).to_string(
                index=False
            )
        )
        f.write("\n")

    print("")
    print("Saved:", pred_path)
    print("Saved:", stats_path)
    print("Saved:", curves_path)
    print("Saved:", holdings_path)
    print("Saved:", report_path)

    print("")
    print("PCA LATENT WALK-FORWARD RESULTS BY ANNUALIZED RETURN")
    print(stats_df.sort_values("annualized_return", ascending=False).to_string(index=False))

    print("")
    print("PCA LATENT WALK-FORWARD RESULTS BY SHARPE")
    print(stats_df.sort_values("sharpe_no_risk_free", ascending=False).to_string(index=False))

    print("")
    print("PCA LATENT WALK-FORWARD RESULTS BY RETURN / DRAWDOWN")
    print(
        stats_df.sort_values("return_over_abs_drawdown", ascending=False).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()