import os
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    non_feature_cols = [
        "date",
        "ticker",
        "long_name",

        "future_1m_return",
        "future_1m_spy_return",
        "future_1m_excess_return",
        "target_outperform_spy_1m",
        "target_top_quintile_1m",

        "future_12m_return_new",
        "future_12m_spy_return_new",
        "future_12m_excess_return",
        "target_outperform_spy_12m",
        "target_top_quintile_12m",

        "future_12m_return",
        "future_12m_spy_return",
        "target_abs_direction",
        "target_outperform_spy",
    ]

    return [c for c in df.columns if c not in non_feature_cols]


def time_based_split(df: pd.DataFrame):
    train = df[df["date"] < "2018-01-01"].copy()
    val = df[(df["date"] >= "2018-01-01") & (df["date"] < "2021-01-01")].copy()
    test = df[df["date"] >= "2021-01-01"].copy()
    return train, val, test


def evaluate_model(model, X, y, split_name: str, model_name: str) -> dict:
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    try:
        auc = roc_auc_score(y, prob)
    except ValueError:
        auc = np.nan

    return {
        "model": model_name,
        "split": split_name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": auc,
    }


def top_ranked_realized_return(test_df: pd.DataFrame, probabilities: np.ndarray, top_n: int) -> float:
    temp = test_df.copy()
    temp["predicted_prob_top_quintile_1m"] = probabilities

    monthly_returns = []

    for date, group in temp.groupby("date"):
        top = group.sort_values("predicted_prob_top_quintile_1m", ascending=False).head(top_n)
        monthly_returns.append(top["future_1m_return"].mean())

    return float(np.mean(monthly_returns))


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    input_path = "data/processed/week12_aligned_modeling_dataset.parquet"

    print("Loading aligned Week 12 dataset...")
    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    df = df.dropna(
        subset=[
            "future_1m_return",
            "future_1m_spy_return",
            "target_top_quintile_1m",
        ]
    )

    feature_cols = get_feature_columns(df)

    print("Dataset shape:", df.shape)
    print("Feature count:", len(feature_cols))
    print("Ticker count:", df["ticker"].nunique())
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Target balance:")
    print(df["target_top_quintile_1m"].value_counts(normalize=True))

    train, val, test = time_based_split(df)

    print("")
    print("Split sizes:")
    print("Train:", train.shape)
    print("Validation:", val.shape)
    print("Test:", test.shape)

    X_train = train[feature_cols]
    y_train = train["target_top_quintile_1m"]

    X_val = val[feature_cols]
    y_val = val["target_top_quintile_1m"]

    X_test = test[feature_cols]
    y_test = test["target_top_quintile_1m"]

    models = {
        "random_forest_top_quintile": RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            class_weight="balanced",
        ),
        "gradient_boosting_top_quintile": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            random_state=42,
        ),
    }

    all_metrics = []
    portfolio_rows = []

    for model_name, model in models.items():
        print("")
        print("=" * 70)
        print("Training:", model_name)

        model.fit(X_train, y_train)

        for split_name, X, y in [
            ("train", X_train, y_train),
            ("validation", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            result = evaluate_model(model, X, y, split_name, model_name)
            all_metrics.append(result)

            print(
                f"{split_name}: "
                f"accuracy={result['accuracy']:.3f}, "
                f"precision={result['precision']:.3f}, "
                f"recall={result['recall']:.3f}, "
                f"f1={result['f1']:.3f}, "
                f"auc={result['auc']:.3f}"
            )

        test_probs = model.predict_proba(X_test)[:, 1]

        top5 = top_ranked_realized_return(test, test_probs, top_n=5)
        top10 = top_ranked_realized_return(test, test_probs, top_n=10)
        top25 = top_ranked_realized_return(test, test_probs, top_n=25)
        top50 = top_ranked_realized_return(test, test_probs, top_n=50)

        portfolio_rows.append(
            {
                "model": model_name,
                "top5_avg_next_1m_return": top5,
                "top10_avg_next_1m_return": top10,
                "top25_avg_next_1m_return": top25,
                "top50_avg_next_1m_return": top50,
                "all_stock_avg_next_1m_return": test["future_1m_return"].mean(),
                "spy_avg_next_1m_return": test["future_1m_spy_return"].mean(),
            }
        )

        prediction_output = test[
            [
                "date",
                "ticker",
                "future_1m_return",
                "future_1m_spy_return",
                "future_1m_excess_return",
                "target_outperform_spy_1m",
                "target_top_quintile_1m",
            ]
        ].copy()

        prediction_output["model"] = model_name
        prediction_output["predicted_prob_top_quintile_1m"] = test_probs
        prediction_output["rank_by_date"] = prediction_output.groupby("date")[
            "predicted_prob_top_quintile_1m"
        ].rank(ascending=False, method="first")

        pred_path = f"outputs/tables/week12_top_quintile_predictions_{model_name}.csv"
        prediction_output.to_csv(pred_path, index=False)
        print("Saved predictions:", pred_path)

    metrics = pd.DataFrame(all_metrics)
    portfolio = pd.DataFrame(portfolio_rows)

    metrics_path = "outputs/tables/week12_top_quintile_metrics.csv"
    portfolio_path = "outputs/tables/week12_top_quintile_portfolio_signal.csv"
    report_path = "outputs/reports/week12_top_quintile_model_summary.txt"

    metrics.to_csv(metrics_path, index=False)
    portfolio.to_csv(portfolio_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 12 Top-Quintile Model Summary\n")
        f.write("==================================\n\n")
        f.write("Target:\n")
        f.write("target_top_quintile_1m\n\n")
        f.write("Interpretation:\n")
        f.write(
            "This model predicts whether a stock will be in the top 20% of next-month returns "
            "within the expanded universe. This target is more directly aligned with portfolio construction "
            "than binary SPY outperformance.\n\n"
        )
        f.write("Classification metrics:\n")
        f.write(metrics.to_string(index=False))
        f.write("\n\nPortfolio signal:\n")
        f.write(portfolio.to_string(index=False))
        f.write("\n")

    print("")
    print("Saved:", metrics_path)
    print("Saved:", portfolio_path)
    print("Saved:", report_path)
    print("")
    print("Portfolio signal:")
    print(portfolio)


if __name__ == "__main__":
    main()