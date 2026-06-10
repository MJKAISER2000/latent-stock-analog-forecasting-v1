import os
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


HORIZONS = [1, 3, 6, 12, 24, 36]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    non_feature_cols = [
        "date",
        "ticker",
        "long_name",

        # old targets
        "future_12m_return",
        "future_12m_spy_return",
        "target_abs_direction",
        "target_outperform_spy",

        # Week 12 targets
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

        # Week 13 horizon-specific targets
        "horizon_months",
        "future_h_return",
        "future_h_spy_return",
        "future_h_excess_return",
        "target_outperform_spy_h",
        "target_top_quintile_h",
    ]

    return [c for c in df.columns if c not in non_feature_cols]


def time_based_split(df: pd.DataFrame):
    train = df[df["date"] < "2018-01-01"].copy()
    val = df[(df["date"] >= "2018-01-01") & (df["date"] < "2021-01-01")].copy()
    test = df[df["date"] >= "2021-01-01"].copy()
    return train, val, test


def evaluate_model(model, X, y, split_name: str, horizon: int) -> dict:
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    try:
        auc = roc_auc_score(y, prob)
    except ValueError:
        auc = np.nan

    return {
        "horizon_months": horizon,
        "split": split_name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": auc,
    }


def top_ranked_signal(test_df: pd.DataFrame, probabilities: np.ndarray, top_n: int) -> float:
    temp = test_df.copy()
    temp["predicted_prob_outperform_h"] = probabilities

    returns = []

    for date, group in temp.groupby("date"):
        top = group.sort_values("predicted_prob_outperform_h", ascending=False).head(top_n)
        returns.append(top["future_h_return"].mean())

    return float(np.mean(returns))


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    features_path = "data/processed/week12_aligned_modeling_dataset.parquet"
    targets_path = "data/processed/week13_horizon_sweep_targets.parquet"

    print("Loading feature dataset...")
    features = pd.read_parquet(features_path)
    features["date"] = pd.to_datetime(features["date"])

    print("Loading horizon targets...")
    targets = pd.read_parquet(targets_path)
    targets["date"] = pd.to_datetime(targets["date"])

    all_metrics = []
    all_signals = []

    for horizon in HORIZONS:
        print("")
        print("=" * 80)
        print(f"Training horizon model: {horizon} months")

        targets_h = targets[targets["horizon_months"] == horizon].copy()

        df = features.merge(
            targets_h,
            on=["date", "ticker"],
            how="inner",
        )

        df = df.dropna(
            subset=[
                "future_h_return",
                "future_h_spy_return",
                "target_outperform_spy_h",
            ]
        )

        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        feature_cols = get_feature_columns(df)

        print("Dataset shape:", df.shape)
        print("Feature count:", len(feature_cols))
        print("Ticker count:", df["ticker"].nunique())
        print("Date range:", df["date"].min(), "to", df["date"].max())
        print("Target balance:")
        print(df["target_outperform_spy_h"].value_counts(normalize=True))

        train, val, test = time_based_split(df)

        print("Split sizes:")
        print("Train:", train.shape)
        print("Validation:", val.shape)
        print("Test:", test.shape)

        X_train = train[feature_cols]
        y_train = train["target_outperform_spy_h"]

        X_val = val[feature_cols]
        y_val = val["target_outperform_spy_h"]

        X_test = test[feature_cols]
        y_test = test["target_outperform_spy_h"]

        model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            random_state=42,
        )

        model.fit(X_train, y_train)

        for split_name, X, y in [
            ("train", X_train, y_train),
            ("validation", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            result = evaluate_model(model, X, y, split_name, horizon)
            all_metrics.append(result)

            print(
                f"{split_name}: "
                f"accuracy={result['accuracy']:.3f}, "
                f"f1={result['f1']:.3f}, "
                f"auc={result['auc']:.3f}"
            )

        test_probs = model.predict_proba(X_test)[:, 1]

        signal_row = {
            "horizon_months": horizon,
            "top5_avg_future_h_return": top_ranked_signal(test, test_probs, top_n=5),
            "top10_avg_future_h_return": top_ranked_signal(test, test_probs, top_n=10),
            "top25_avg_future_h_return": top_ranked_signal(test, test_probs, top_n=25),
            "top50_avg_future_h_return": top_ranked_signal(test, test_probs, top_n=50),
            "all_stock_avg_future_h_return": test["future_h_return"].mean(),
            "spy_avg_future_h_return": test["future_h_spy_return"].mean(),
        }

        all_signals.append(signal_row)

        prediction_output = test[
            [
                "date",
                "ticker",
                "horizon_months",
                "future_h_return",
                "future_h_spy_return",
                "future_h_excess_return",
                "target_outperform_spy_h",
                "target_top_quintile_h",
            ]
        ].copy()

        prediction_output["predicted_prob_outperform_h"] = test_probs
        prediction_output["rank_by_date"] = prediction_output.groupby("date")[
            "predicted_prob_outperform_h"
        ].rank(ascending=False, method="first")

        pred_path = f"outputs/tables/week13_predictions_horizon_{horizon}m.csv"
        prediction_output.to_csv(pred_path, index=False)

        print("Saved predictions:", pred_path)
        print("Signal row:")
        print(signal_row)

    metrics = pd.DataFrame(all_metrics)
    signals = pd.DataFrame(all_signals)

    metrics_path = "outputs/tables/week13_horizon_model_metrics.csv"
    signals_path = "outputs/tables/week13_horizon_portfolio_signal.csv"
    report_path = "outputs/reports/week13_horizon_model_summary.txt"

    metrics.to_csv(metrics_path, index=False)
    signals.to_csv(signals_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 13 Horizon Sweep Model Summary\n")
        f.write("===================================\n\n")
        f.write("Model:\n")
        f.write("GradientBoostingClassifier trained separately for each horizon.\n\n")
        f.write("Horizons tested:\n")
        f.write(str(HORIZONS))
        f.write("\n\nClassification metrics:\n")
        f.write(metrics.to_string(index=False))
        f.write("\n\nPortfolio signal by horizon:\n")
        f.write(signals.to_string(index=False))
        f.write("\n")

    print("")
    print("Saved:", metrics_path)
    print("Saved:", signals_path)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()