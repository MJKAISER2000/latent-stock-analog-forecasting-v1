import os
import json
import pandas as pd
import numpy as np

import lightgbm as lgb


DATASET_PATH = "data/processed/week15_full500_modeling_dataset.parquet"
HORIZON = 1


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


def time_based_split(df: pd.DataFrame):
    train = df[df["date"] < "2018-01-01"].copy()
    val = df[(df["date"] >= "2018-01-01") & (df["date"] < "2021-01-01")].copy()
    test = df[df["date"] >= "2021-01-01"].copy()
    return train, val, test


def categorize_feature(feature_name: str) -> str:
    name = feature_name.lower()

    if "future" in name or "target" in name:
        return "WARNING_possible_target_leakage"

    if "spy" in name or "bear_regime" in name or "correction_regime" in name or "crash_regime" in name:
        return "market_regime"

    if "ret_" in name or "return" in name:
        return "stock_return_momentum"

    if "vol" in name:
        return "volatility"

    if "drawdown" in name:
        return "drawdown"

    if "ma_" in name or "price_to_ma" in name:
        return "moving_average"

    if "sector" in name:
        return "sector_metadata"

    if "industry" in name:
        return "industry_metadata"

    if "exchange" in name:
        return "exchange_metadata"

    return "other"


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Loading dataset...")
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    future_return_col = f"future_{HORIZON}m_return"
    future_spy_col = f"future_{HORIZON}m_spy_return"
    future_excess_col = f"future_{HORIZON}m_excess_return"
    target_col = f"target_outperform_spy_{HORIZON}m"

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
    print("Train/Val/Test:", train.shape, val.shape, test.shape)

    safe_feature_names = [f"feature_{i}" for i in range(len(feature_cols))]
    safe_to_original = dict(zip(safe_feature_names, feature_cols))

    X_train = train[feature_cols].copy()
    X_val = val[feature_cols].copy()

    X_train.columns = safe_feature_names
    X_val.columns = safe_feature_names

    y_train = train["ranking_label"].astype(int)
    y_val = val["ranking_label"].astype(int)

    train_group = group_sizes_by_date(train)
    val_group = group_sizes_by_date(val)

    print("Training ranker for feature importance...")

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
            lgb.log_evaluation(period=50),
        ],
    )

    booster = ranker.booster_

    split_importance = booster.feature_importance(importance_type="split")
    gain_importance = booster.feature_importance(importance_type="gain")

    importance = pd.DataFrame(
        {
            "safe_feature_name": safe_feature_names,
            "feature": [safe_to_original[s] for s in safe_feature_names],
            "split_importance": split_importance,
            "gain_importance": gain_importance,
        }
    )

    importance["gain_importance_pct"] = importance["gain_importance"] / importance["gain_importance"].sum()
    importance["split_importance_pct"] = importance["split_importance"] / importance["split_importance"].sum()
    importance["feature_category"] = importance["feature"].apply(categorize_feature)

    importance = importance.sort_values("gain_importance", ascending=False).reset_index(drop=True)

    category_summary = (
        importance.groupby("feature_category")
        .agg(
            feature_count=("feature", "count"),
            total_gain=("gain_importance", "sum"),
            total_split=("split_importance", "sum"),
        )
        .reset_index()
        .sort_values("total_gain", ascending=False)
    )

    category_summary["gain_share"] = category_summary["total_gain"] / category_summary["total_gain"].sum()
    category_summary["split_share"] = category_summary["total_split"] / category_summary["total_split"].sum()

    possible_leakage = importance[
        importance["feature_category"] == "WARNING_possible_target_leakage"
    ].copy()

    importance_path = "outputs/tables/week19_ranker_feature_importance.csv"
    category_path = "outputs/tables/week19_ranker_feature_importance_by_category.csv"
    mapping_path = "outputs/tables/week19_ranker_feature_name_mapping.csv"
    report_path = "outputs/reports/week19_ranker_feature_importance_summary.txt"
    model_path = "models/week19_feature_importance_lgbm_ranker.txt"

    importance.to_csv(importance_path, index=False)
    category_summary.to_csv(category_path, index=False)

    pd.DataFrame(
        {
            "safe_feature_name": safe_feature_names,
            "original_feature_name": feature_cols,
        }
    ).to_csv(mapping_path, index=False)

    booster.save_model(model_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 19 Ranker Feature Importance Summary\n")
        f.write("========================================\n\n")
        f.write("Goal:\n")
        f.write("Inspect whether the Week 17 LightGBM ranker is using sensible features or suspicious leakage-like features.\n\n")
        f.write(f"Best iteration: {ranker.best_iteration_}\n")
        f.write(f"Feature count: {len(feature_cols)}\n\n")

        f.write("Top 40 features by gain:\n")
        f.write(
            importance[
                [
                    "feature",
                    "feature_category",
                    "gain_importance",
                    "gain_importance_pct",
                    "split_importance",
                ]
            ]
            .head(40)
            .to_string(index=False)
        )
        f.write("\n\n")

        f.write("Importance by feature category:\n")
        f.write(category_summary.to_string(index=False))
        f.write("\n\n")

        f.write("Possible leakage features found:\n")
        if len(possible_leakage) == 0:
            f.write("None detected by feature-name scan.\n")
        else:
            f.write(possible_leakage.to_string(index=False))
            f.write("\n")

    print("")
    print("Saved:", importance_path)
    print("Saved:", category_path)
    print("Saved:", mapping_path)
    print("Saved:", model_path)
    print("Saved:", report_path)

    print("")
    print("TOP 40 FEATURES BY GAIN")
    print(
        importance[
            [
                "feature",
                "feature_category",
                "gain_importance",
                "gain_importance_pct",
                "split_importance",
            ]
        ]
        .head(40)
        .to_string(index=False)
    )

    print("")
    print("IMPORTANCE BY CATEGORY")
    print(category_summary.to_string(index=False))

    print("")
    print("POSSIBLE LEAKAGE FEATURES")
    if len(possible_leakage) == 0:
        print("None detected by feature-name scan.")
    else:
        print(possible_leakage.to_string(index=False))


if __name__ == "__main__":
    main()