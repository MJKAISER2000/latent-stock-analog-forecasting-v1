import os
import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn.metrics import roc_auc_score


DATASET_NAME = "week18_full500_leadership"
DATASET_PATH = "data/processed/week18_full500_leadership_modeling_dataset.parquet"
HORIZONS = [1, 36]


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


def time_based_split(df: pd.DataFrame):
    train = df[df["date"] < "2018-01-01"].copy()
    val = df[(df["date"] >= "2018-01-01") & (df["date"] < "2021-01-01")].copy()
    test = df[df["date"] >= "2021-01-01"].copy()
    return train, val, test


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


def top_ranked_signal(test_df: pd.DataFrame, scores: np.ndarray, horizon: int, top_n: int) -> float:
    temp = test_df.copy()
    temp["ranker_score"] = scores

    return_col = f"future_{horizon}m_return"

    returns = []

    for date, group in temp.groupby("date"):
        top = group.sort_values("ranker_score", ascending=False).head(top_n)
        returns.append(top[return_col].mean())

    return float(np.mean(returns))


def top_ranked_excess_signal(test_df: pd.DataFrame, scores: np.ndarray, horizon: int, top_n: int) -> float:
    temp = test_df.copy()
    temp["ranker_score"] = scores

    return_col = f"future_{horizon}m_excess_return"

    returns = []

    for date, group in temp.groupby("date"):
        top = group.sort_values("ranker_score", ascending=False).head(top_n)
        returns.append(top[return_col].mean())

    return float(np.mean(returns))


def classification_auc_from_scores(test_df: pd.DataFrame, scores: np.ndarray, horizon: int) -> float:
    target_col = f"target_outperform_spy_{horizon}m"
    y = test_df[target_col].astype(int)

    try:
        return roc_auc_score(y, scores)
    except ValueError:
        return np.nan


def train_one(horizon: int):
    print("")
    print("=" * 100)
    print(f"Training leadership LightGBM ranker | horizon={horizon}m")

    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    future_return_col = f"future_{horizon}m_return"
    future_spy_col = f"future_{horizon}m_spy_return"
    future_excess_col = f"future_{horizon}m_excess_return"
    target_col = f"target_outperform_spy_{horizon}m"

    df = df.dropna(
        subset=[
            future_return_col,
            future_spy_col,
            future_excess_col,
            target_col,
        ]
    ).copy()

    feature_cols = get_feature_columns(df)

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    df["ranking_label"] = make_ranking_label(df, future_return_col)

    train, val, test = time_based_split(df)

    print("Dataset shape:", df.shape)
    print("Feature count:", len(feature_cols))
    print("Ticker count:", df["ticker"].nunique())
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Train/Val/Test:", train.shape, val.shape, test.shape)
    print("Train ranking labels:")
    print(train["ranking_label"].value_counts(normalize=True).sort_index())

    safe_feature_names = [f"feature_{i}" for i in range(len(feature_cols))]

    X_train = train[feature_cols].copy()
    X_val = val[feature_cols].copy()
    X_test = test[feature_cols].copy()

    X_train.columns = safe_feature_names
    X_val.columns = safe_feature_names
    X_test.columns = safe_feature_names

    y_train = train["ranking_label"].astype(int)
    y_val = val["ranking_label"].astype(int)

    train_group = group_sizes_by_date(train)
    val_group = group_sizes_by_date(val)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=600,
        learning_rate=0.025,
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
            lgb.log_evaluation(period=50),
        ],
    )

    test_scores = ranker.predict(X_test)

    signal = {
        "dataset": DATASET_NAME,
        "horizon_months": horizon,
        "model": "lightgbm_lambdarank_leadership_features",
        "best_iteration": ranker.best_iteration_,
        "test_auc_vs_spy_outperform": classification_auc_from_scores(test, test_scores, horizon),
        "top5_avg_future_return": top_ranked_signal(test, test_scores, horizon, 5),
        "top10_avg_future_return": top_ranked_signal(test, test_scores, horizon, 10),
        "top25_avg_future_return": top_ranked_signal(test, test_scores, horizon, 25),
        "top50_avg_future_return": top_ranked_signal(test, test_scores, horizon, 50),
        "top5_avg_future_excess": top_ranked_excess_signal(test, test_scores, horizon, 5),
        "top10_avg_future_excess": top_ranked_excess_signal(test, test_scores, horizon, 10),
        "top25_avg_future_excess": top_ranked_excess_signal(test, test_scores, horizon, 25),
        "top50_avg_future_excess": top_ranked_excess_signal(test, test_scores, horizon, 50),
        "all_stock_avg_future_return": test[future_return_col].mean(),
        "spy_avg_future_return": test[future_spy_col].mean(),
    }

    prediction_output = test[
        [
            "date",
            "ticker",
            future_return_col,
            future_spy_col,
            future_excess_col,
            target_col,
            "ranking_label",
        ]
    ].copy()

    prediction_output["dataset"] = DATASET_NAME
    prediction_output["horizon_months"] = horizon
    prediction_output["model"] = "lightgbm_lambdarank_leadership_features"
    prediction_output["ranker_score"] = test_scores
    prediction_output["rank_by_date"] = prediction_output.groupby("date")[
        "ranker_score"
    ].rank(ascending=False, method="first")

    pred_path = f"outputs/tables/week18_leadership_ranker_predictions_full500_{horizon}m.csv"
    prediction_output.to_csv(pred_path, index=False)

    print("Saved predictions:", pred_path)
    print("Signal:")
    print(signal)

    return pd.DataFrame([signal])


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    all_signals = []

    for horizon in HORIZONS:
        signal = train_one(horizon)
        all_signals.append(signal)

    signals_df = pd.concat(all_signals, ignore_index=True)

    signals_path = "outputs/tables/week18_leadership_ranker_portfolio_signal.csv"
    report_path = "outputs/reports/week18_leadership_ranker_summary.txt"

    signals_df.to_csv(signals_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 18 Leadership Feature LightGBM Ranker Summary\n")
        f.write("=================================================\n\n")
        f.write("Goal:\n")
        f.write("Train LightGBM rankers using added sector and industry leadership features.\n\n")
        f.write("Portfolio signal:\n")
        f.write(signals_df.to_string(index=False))
        f.write("\n")

    print("")
    print("Saved:", signals_path)
    print("Saved:", report_path)
    print("")
    print(signals_df.to_string(index=False))


if __name__ == "__main__":
    main()