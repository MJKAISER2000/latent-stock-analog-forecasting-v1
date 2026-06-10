import os
import pandas as pd
import numpy as np

import lightgbm as lgb


DATASET_PATH = "data/processed/week19_full500_with_market_pca_latents.parquet"

PREDICTION_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
TOP_NS = [5, 10, 20]
TRANSACTION_COST = 0.001

FEATURE_SETS = [
    "original_only",
    "pca_only",
    "original_plus_pca",
]


def get_base_non_feature_cols() -> list[str]:
    return [
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

        # CRITICAL: do not let the model train on the ranking label.
        "ranking_label",
    ]


def get_feature_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    non_feature_cols = get_base_non_feature_cols()

    all_features = [c for c in df.columns if c not in non_feature_cols]

    pca_cols = [c for c in all_features if c.startswith("market_pca_")]
    original_cols = [c for c in all_features if not c.startswith("market_pca_")]

    if feature_set == "original_only":
        return original_cols

    if feature_set == "pca_only":
        return pca_cols

    if feature_set == "original_plus_pca":
        return original_cols + pca_cols

    raise ValueError(f"Unknown feature_set: {feature_set}")


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
    feature_set: str,
) -> pd.DataFrame:
    print("")
    print("=" * 100)
    print(
        f"Ablation fold | feature_set={feature_set} | "
        f"train before {prediction_year}, predict {prediction_year}"
    )

    train = df[df["date"].dt.year < prediction_year].copy()
    predict = df[df["date"].dt.year == prediction_year].copy()

    if len(train) == 0 or len(predict) == 0:
        return pd.DataFrame()

    train_dates = sorted(train["date"].unique())
    val_start_idx = int(len(train_dates) * 0.80)
    val_dates = set(train_dates[val_start_idx:])

    train_fit = train[~train["date"].isin(val_dates)].copy()
    val = train[train["date"].isin(val_dates)].copy()

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

        # Controlled ablation: remove sampling randomness.
        subsample=1.0,
        colsample_bytree=1.0,

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
    out["feature_set"] = feature_set
    out["model"] = "walk_forward_lgbm_ranker_pca_ablation"
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

    print("Loading merged PCA latent dataset...")
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

    df["ranking_label"] = make_ranking_label(df, "future_1m_return")

    monthly_returns = load_monthly_returns()
    features_for_weights = load_features_for_weights()

    all_stats = []
    all_predictions = []
    all_curves = None
    all_holdings = []

    for feature_set in FEATURE_SETS:
        feature_cols = get_feature_columns(df, feature_set)

        leaked_cols = [
            c for c in feature_cols
            if "future" in c.lower()
            or "target" in c.lower()
            or "ranking_label" in c.lower()
        ]

        if leaked_cols:
            raise ValueError(f"Leakage columns found in feature set {feature_set}: {leaked_cols}")

        print("")
        print("#" * 100)
        print(f"FEATURE SET: {feature_set}")
        print("Feature count:", len(feature_cols))
        print("PCA cols:", [c for c in feature_cols if c.startswith("market_pca_")])

        working = df.copy()

        for col in feature_cols:
            working[col] = pd.to_numeric(working[col], errors="coerce")

        working[feature_cols] = working[feature_cols].replace([np.inf, -np.inf], np.nan)
        working[feature_cols] = working[feature_cols].fillna(working[feature_cols].median())
        working[feature_cols] = working[feature_cols].fillna(0.0)

        predictions_for_set = []

        for year in PREDICTION_YEARS:
            pred_year = train_predict_for_year(
                working,
                feature_cols,
                year,
                feature_set,
            )

            if not pred_year.empty:
                predictions_for_set.append(pred_year)

        predictions = pd.concat(predictions_for_set, ignore_index=True)
        all_predictions.append(predictions)

        for top_n in TOP_NS:
            baskets = build_baskets(predictions, top_n)

            strategy = build_strategy_returns(
                baskets=baskets,
                monthly_returns=monthly_returns,
                features=features_for_weights,
                transaction_cost=TRANSACTION_COST,
            )

            stats = {
                "feature_set": feature_set,
                "strategy": f"week19_ablation_{feature_set}_top{top_n}_inverse_vol",
                "top_n": top_n,
                "transaction_cost": TRANSACTION_COST,
                **performance_stats(strategy["strategy_monthly_return"]),
                "avg_active_basket_count": strategy["active_basket_count"].mean(),
            }

            all_stats.append(stats)

            curve_col = f"{feature_set}_top{top_n}_return"
            temp = strategy[["date", "strategy_monthly_return"]].rename(
                columns={"strategy_monthly_return": curve_col}
            )
            temp[f"{feature_set}_top{top_n}_cumulative"] = (
                1 + temp[curve_col]
            ).cumprod()

            if all_curves is None:
                all_curves = temp
            else:
                all_curves = all_curves.merge(temp, on="date", how="outer")

            baskets["feature_set"] = feature_set
            baskets["strategy"] = f"ablation_{feature_set}_top{top_n}"
            all_holdings.append(baskets)

    stats_df = pd.DataFrame(all_stats)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    holdings_df = pd.concat(all_holdings, ignore_index=True)

    stats_path = "outputs/tables/week19_pca_latent_ablation_stats.csv"
    predictions_path = "outputs/tables/week19_pca_latent_ablation_predictions.csv"
    curves_path = "outputs/tables/week19_pca_latent_ablation_curves.csv"
    holdings_path = "outputs/tables/week19_pca_latent_ablation_holdings.csv"
    report_path = "outputs/reports/week19_pca_latent_ablation_summary.txt"

    stats_df.to_csv(stats_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    all_curves.to_csv(curves_path, index=False)
    holdings_df.to_csv(holdings_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 19 PCA Latent Ablation Summary\n")
        f.write("==================================\n\n")
        f.write("Goal:\n")
        f.write(
            "Controlled test of whether PCA latent market-state features improve walk-forward ranker performance.\n\n"
        )
        f.write("Feature sets:\n")
        f.write(str(FEATURE_SETS))
        f.write("\n\n")
        f.write("Results sorted by annualized return:\n")
        f.write(stats_df.sort_values("annualized_return", ascending=False).to_string(index=False))
        f.write("\n\n")
        f.write("Results sorted by Sharpe:\n")
        f.write(stats_df.sort_values("sharpe_no_risk_free", ascending=False).to_string(index=False))
        f.write("\n")

    print("")
    print("Saved:", stats_path)
    print("Saved:", predictions_path)
    print("Saved:", curves_path)
    print("Saved:", holdings_path)
    print("Saved:", report_path)

    print("")
    print("ABLATION RESULTS BY ANNUALIZED RETURN")
    print(stats_df.sort_values("annualized_return", ascending=False).to_string(index=False))

    print("")
    print("ABLATION RESULTS BY SHARPE")
    print(stats_df.sort_values("sharpe_no_risk_free", ascending=False).to_string(index=False))

    print("")
    print("ABLATION RESULTS BY RETURN / DRAWDOWN")
    print(stats_df.sort_values("return_over_abs_drawdown", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()