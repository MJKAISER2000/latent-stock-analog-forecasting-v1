import os
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


DATASETS = ["balanced450", "balanced900", "lowvol300", "midvol300", "highvol300"]
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


def evaluate_model(model, X, y, dataset_name: str, horizon: int, split_name: str) -> dict:
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    try:
        auc = roc_auc_score(y, prob)
    except ValueError:
        auc = np.nan

    return {
        "dataset": dataset_name,
        "horizon_months": horizon,
        "split": split_name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": auc,
    }


def top_ranked_signal(test_df: pd.DataFrame, probabilities: np.ndarray, horizon: int, top_n: int) -> float:
    temp = test_df.copy()
    temp["predicted_prob_outperform"] = probabilities

    return_col = f"future_{horizon}m_return"

    returns = []

    for date, group in temp.groupby("date"):
        top = group.sort_values("predicted_prob_outperform", ascending=False).head(top_n)
        returns.append(top[return_col].mean())

    return float(np.mean(returns))


def train_one(dataset_name: str, horizon: int):
    input_path = f"data/processed/week16_{dataset_name}_modeling_dataset.parquet"

    print("")
    print("=" * 90)
    print(f"Training Week 16 raw GB | dataset={dataset_name} | horizon={horizon}m")

    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    target_col = f"target_outperform_spy_{horizon}m"
    return_col = f"future_{horizon}m_return"
    spy_return_col = f"future_{horizon}m_spy_return"

    df = df.dropna(subset=[target_col, return_col, spy_return_col]).copy()

    feature_cols = get_feature_columns(df)

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    print("Dataset shape after target drop:", df.shape)
    print("Feature count:", len(feature_cols))
    print("Ticker count:", df["ticker"].nunique())
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Target balance:")
    print(df[target_col].value_counts(normalize=True))

    train, val, test = time_based_split(df)

    print("Split sizes:")
    print("Train:", train.shape)
    print("Validation:", val.shape)
    print("Test:", test.shape)

    X_train = train[feature_cols]
    y_train = train[target_col].astype(int)

    X_val = val[feature_cols]
    y_val = val[target_col].astype(int)

    X_test = test[feature_cols]
    y_test = test[target_col].astype(int)

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        random_state=42,
    )

    model.fit(X_train, y_train)

    metrics = []

    for split_name, X, y in [
        ("train", X_train, y_train),
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        result = evaluate_model(model, X, y, dataset_name, horizon, split_name)
        metrics.append(result)

        print(
            f"{split_name}: "
            f"accuracy={result['accuracy']:.3f}, "
            f"f1={result['f1']:.3f}, "
            f"auc={result['auc']:.3f}"
        )

    test_probs = model.predict_proba(X_test)[:, 1]

    signal = {
        "dataset": dataset_name,
        "horizon_months": horizon,
        "model": "raw_gradient_boosting",
        "top5_avg_future_return": top_ranked_signal(test, test_probs, horizon, 5),
        "top10_avg_future_return": top_ranked_signal(test, test_probs, horizon, 10),
        "top25_avg_future_return": top_ranked_signal(test, test_probs, horizon, 25),
        "top50_avg_future_return": top_ranked_signal(test, test_probs, horizon, 50),
        "all_stock_avg_future_return": test[return_col].mean(),
        "spy_avg_future_return": test[spy_return_col].mean(),
    }

    prediction_output = test[
        [
            "date",
            "ticker",
            return_col,
            spy_return_col,
            f"future_{horizon}m_excess_return",
            target_col,
        ]
    ].copy()

    prediction_output["dataset"] = dataset_name
    prediction_output["horizon_months"] = horizon
    prediction_output["model"] = "raw_gradient_boosting"
    prediction_output["predicted_prob_outperform"] = test_probs
    prediction_output["rank_by_date"] = prediction_output.groupby("date")[
        "predicted_prob_outperform"
    ].rank(ascending=False, method="first")

    pred_path = f"outputs/tables/week16_raw_gb_predictions_{dataset_name}_{horizon}m.csv"
    prediction_output.to_csv(pred_path, index=False)

    print("Saved predictions:", pred_path)
    print("Signal:", signal)

    return pd.DataFrame(metrics), pd.DataFrame([signal])


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    all_metrics = []
    all_signals = []

    for dataset_name in DATASETS:
        for horizon in HORIZONS:
            metrics, signal = train_one(dataset_name, horizon)
            all_metrics.append(metrics)
            all_signals.append(signal)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    signals_df = pd.concat(all_signals, ignore_index=True)

    metrics_path = "outputs/tables/week16_raw_gb_model_metrics.csv"
    signals_path = "outputs/tables/week16_raw_gb_portfolio_signal.csv"
    report_path = "outputs/reports/week16_raw_gb_model_summary.txt"

    metrics_df.to_csv(metrics_path, index=False)
    signals_df.to_csv(signals_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 16 Raw Gradient Boosting Model Summary\n")
        f.write("==========================================\n\n")
        f.write("Goal:\n")
        f.write(
            "Train raw-feature gradient boosting models on large balanced volatility universes "
            "for 1-month and 36-month horizons.\n\n"
        )
        f.write("Metrics:\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\nPortfolio signal:\n")
        f.write(signals_df.to_string(index=False))
        f.write("\n")

    print("")
    print("Saved:", metrics_path)
    print("Saved:", signals_path)
    print("Saved:", report_path)
    print("")
    print("Portfolio signals:")
    print(signals_df)


if __name__ == "__main__":
    main()