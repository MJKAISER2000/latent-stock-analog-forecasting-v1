import os
import yaml
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    non_feature_cols = [
        "date",
        "ticker",
        "future_12m_return",
        "future_12m_spy_return",
        "target_abs_direction",
        "target_outperform_spy",
    ]

    return [col for col in df.columns if col not in non_feature_cols]


def time_based_split(df: pd.DataFrame):
    """
    Uses a simple chronological split.

    Train: dates before 2018
    Validation: 2018 through 2020
    Test: 2021 onward

    This avoids random leakage across time.
    """
    train = df[df["date"] < "2018-01-01"].copy()
    val = df[(df["date"] >= "2018-01-01") & (df["date"] < "2021-01-01")].copy()
    test = df[df["date"] >= "2021-01-01"].copy()

    return train, val, test


def evaluate_classifier(model, X, y, split_name: str, model_name: str) -> dict:
    pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        prob = pred

    try:
        auc = roc_auc_score(y, prob)
    except ValueError:
        auc = np.nan

    results = {
        "model": model_name,
        "split": split_name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": auc,
    }

    return results


def top_ranked_portfolio_return(test_df: pd.DataFrame, probabilities: np.ndarray, top_n: int = 5) -> float:
    """
    Simple first-pass portfolio test.

    For each date:
    - rank stocks by predicted probability of outperforming SPY
    - take top N
    - average their future 12-month returns

    This is not the final backtester, just an early signal test.
    """
    temp = test_df.copy()
    temp["predicted_prob"] = probabilities

    monthly_returns = []

    for date, group in temp.groupby("date"):
        top = group.sort_values("predicted_prob", ascending=False).head(top_n)
        monthly_returns.append(top["future_12m_return"].mean())

    return float(np.mean(monthly_returns))


def run_baselines():
    config = load_config("configs/experiment_01.yaml")
    processed_dir = config["processed_data_dir"]

    input_path = os.path.join(processed_dir, "modeling_dataset.parquet")
    df = pd.read_parquet(input_path)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    feature_cols = get_feature_columns(df)

    print("Dataset shape:", df.shape)
    print("Number of features:", len(feature_cols))
    print("Date range:", df["date"].min(), "to", df["date"].max())

    train, val, test = time_based_split(df)

    print("")
    print("Split sizes:")
    print("Train:", train.shape)
    print("Validation:", val.shape)
    print("Test:", test.shape)

    X_train = train[feature_cols]
    y_train = train["target_outperform_spy"]

    X_val = val[feature_cols]
    y_val = val["target_outperform_spy"]

    X_test = test[feature_cols]
    y_test = test["target_outperform_spy"]

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            class_weight="balanced",
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            random_state=42,
        ),
    }

    all_results = []
    portfolio_results = []

    for model_name, model in models.items():
        print("")
        print("=" * 60)
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)

        for split_name, X, y in [
            ("train", X_train, y_train),
            ("validation", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            result = evaluate_classifier(model, X, y, split_name, model_name)
            all_results.append(result)

            print(
                f"{split_name}: "
                f"accuracy={result['accuracy']:.3f}, "
                f"f1={result['f1']:.3f}, "
                f"auc={result['auc']:.3f}"
            )

        test_probs = model.predict_proba(X_test)[:, 1]
        prediction_output = test[[
            "date",
            "ticker",
            "future_12m_return",
            "future_12m_spy_return",
            "target_outperform_spy",
        ]].copy()

        prediction_output["model"] = model_name
        prediction_output["predicted_prob_outperform"] = test_probs
        prediction_output["rank_by_date"] = prediction_output.groupby("date")[
            "predicted_prob_outperform"
        ].rank(ascending=False, method="first")

        prediction_path = f"outputs/tables/week3_predictions_{model_name}.csv"
        prediction_output.to_csv(prediction_path, index=False)
        print(f"Saved test predictions to: {prediction_path}")
        top5_return = top_ranked_portfolio_return(test, test_probs, top_n=5)
        top10_return = top_ranked_portfolio_return(test, test_probs, top_n=10)

        portfolio_results.append(
            {
                "model": model_name,
                "top5_avg_future_12m_return": top5_return,
                "top10_avg_future_12m_return": top10_return,
                "test_spy_avg_future_12m_return": test["future_12m_spy_return"].mean(),
            }
        )

        print(f"Top 5 average future 12m return: {top5_return:.3f}")
        print(f"Top 10 average future 12m return: {top10_return:.3f}")
        print(f"SPY average future 12m return: {test['future_12m_spy_return'].mean():.3f}")

    results_df = pd.DataFrame(all_results)
    portfolio_df = pd.DataFrame(portfolio_results)

    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    results_path = "outputs/tables/week3_baseline_metrics.csv"
    portfolio_path = "outputs/tables/week3_portfolio_signal.csv"

    results_df.to_csv(results_path, index=False)
    portfolio_df.to_csv(portfolio_path, index=False)

    print("")
    print("Saved baseline metrics to:", results_path)
    print("Saved portfolio signal results to:", portfolio_path)

    report_path = "outputs/reports/week3_baseline_summary.txt"

    with open(report_path, "w") as f:
        f.write("Week 3 Baseline Model Summary\n")
        f.write("=============================\n\n")
        f.write("Dataset shape:\n")
        f.write(str(df.shape) + "\n\n")
        f.write("Feature count:\n")
        f.write(str(len(feature_cols)) + "\n\n")
        f.write("Split sizes:\n")
        f.write(f"Train: {train.shape}\n")
        f.write(f"Validation: {val.shape}\n")
        f.write(f"Test: {test.shape}\n\n")
        f.write("Classification metrics:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        f.write("Simple top-ranked portfolio signal:\n")
        f.write(portfolio_df.to_string(index=False))
        f.write("\n")

    print("Saved summary report to:", report_path)


if __name__ == "__main__":
    run_baselines()