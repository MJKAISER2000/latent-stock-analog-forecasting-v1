from typing import Any

import lightgbm as lgb
import pandas as pd


def make_ranking_label(df: pd.DataFrame, return_col: str) -> pd.Series:
    """
    Convert future returns within each date into ordinal ranking labels:
    0 = bottom quintile, ..., 4 = top quintile.

    Assumes return_col has already been cleaned of NaNs.
    """

    labels = pd.Series(index=df.index, dtype=float)

    for date, group in df.groupby("date"):
        valid_group = group.dropna(subset=[return_col]).copy()

        if len(valid_group) == 0:
            continue

        ranks = valid_group[return_col].rank(method="first", pct=True)

        relevance = pd.cut(
            ranks,
            bins=[0.0, 0.20, 0.40, 0.60, 0.80, 1.0],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True,
        )

        labels.loc[valid_group.index] = relevance.astype(int)

    if labels.isna().sum() > 0:
        bad_count = int(labels.isna().sum())
        raise ValueError(
            f"Ranking label creation produced {bad_count} NaNs. "
            f"Check missing values in {return_col}."
        )

    return labels.astype(int)


def group_sizes_by_date(df: pd.DataFrame) -> list[int]:
    return df.groupby("date").size().tolist()


def get_ranker_params(config: dict[str, Any]) -> dict[str, Any]:
    params = config["model"]["ranker"].copy()

    return {
        "objective": params.get("objective", "lambdarank"),
        "metric": params.get("metric", "ndcg"),
        "boosting_type": "gbdt",
        "n_estimators": params.get("n_estimators", 500),
        "learning_rate": params.get("learning_rate", 0.03),
        "num_leaves": params.get("num_leaves", 31),
        "max_depth": params.get("max_depth", -1),
        "min_child_samples": params.get("min_child_samples", 20),
        "subsample": params.get("subsample", 1.0),
        "colsample_bytree": params.get("colsample_bytree", 1.0),
        "reg_alpha": params.get("reg_alpha", 0.1),
        "reg_lambda": params.get("reg_lambda", 1.0),
        "random_state": params.get("random_state", 42),
        "n_jobs": -1,
    }


def train_lambdarank_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    config: dict[str, Any],
    label_col: str = "ranking_label",
) -> lgb.LGBMRanker:
    """
    Train one LightGBM LambdaRank model using date groups.
    """

    if len(train_df) == 0:
        raise ValueError("train_df is empty.")

    if len(val_df) == 0:
        raise ValueError("val_df is empty.")

    safe_feature_names = [f"feature_{i}" for i in range(len(feature_cols))]

    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()

    X_train.columns = safe_feature_names
    X_val.columns = safe_feature_names

    y_train = train_df[label_col].astype(int)
    y_val = val_df[label_col].astype(int)

    train_group = group_sizes_by_date(train_df)
    val_group = group_sizes_by_date(val_df)

    ranker = lgb.LGBMRanker(**get_ranker_params(config))

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

    return ranker


def predict_ranker_scores(
    model: lgb.LGBMRanker,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Predict scores and within-date ranks.
    """

    safe_feature_names = [f"feature_{i}" for i in range(len(feature_cols))]

    X = df[feature_cols].copy()
    X.columns = safe_feature_names

    out = df[["date", "ticker"]].copy()
    out["ranker_score"] = model.predict(X)
    out["rank_by_date"] = out.groupby("date")["ranker_score"].rank(
        ascending=False,
        method="first",
    )

    return out


def train_predict_latest(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: dict[str, Any],
    signal_date: pd.Timestamp,
) -> tuple[lgb.LGBMRanker, pd.DataFrame]:
    """
    Train on all dates before signal_date and score the selected signal_date.

    Important:
    - Training rows must have known future_1m_return.
    - Prediction rows may not have future_1m_return yet.
    """

    working = df.copy()
    working["date"] = pd.to_datetime(working["date"])
    signal_date = pd.Timestamp(signal_date)

    train_all = working[working["date"] < signal_date].copy()
    predict_df = working[working["date"] == signal_date].copy()

    if len(train_all) == 0:
        raise ValueError(f"No training rows before signal_date={signal_date}")

    if len(predict_df) == 0:
        raise ValueError(f"No prediction rows for signal_date={signal_date}")

    # CRITICAL FIX:
    # Only train on rows where the future label is known.
    train_all = train_all.dropna(subset=["future_1m_return"]).copy()

    if len(train_all) == 0:
        raise ValueError(
            f"No usable training rows with future_1m_return before signal_date={signal_date}"
        )

    train_all["ranking_label"] = make_ranking_label(
        train_all,
        "future_1m_return",
    )

    train_dates = sorted(train_all["date"].unique())

    if len(train_dates) < 5:
        raise ValueError(
            f"Not enough training dates before signal_date={signal_date}. "
            f"Found only {len(train_dates)} dates."
        )

    val_start_idx = int(len(train_dates) * 0.80)
    val_dates = set(train_dates[val_start_idx:])

    train_fit = train_all[~train_all["date"].isin(val_dates)].copy()
    val = train_all[train_all["date"].isin(val_dates)].copy()

    if len(train_fit) == 0:
        raise ValueError("train_fit is empty after time split.")

    if len(val) == 0:
        raise ValueError("validation set is empty after time split.")

    model = train_lambdarank_model(
        train_df=train_fit,
        val_df=val,
        feature_cols=feature_cols,
        config=config,
    )

    predictions = predict_ranker_scores(
        model=model,
        df=predict_df,
        feature_cols=feature_cols,
    )

    return model, predictions