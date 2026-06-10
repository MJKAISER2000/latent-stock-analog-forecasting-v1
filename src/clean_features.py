import os
import yaml
import pandas as pd
import numpy as np


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clean_feature_dataset():
    config = load_config("configs/experiment_01.yaml")
    processed_dir = config["processed_data_dir"]

    input_path = os.path.join(processed_dir, "features.parquet")
    output_path = os.path.join(processed_dir, "modeling_dataset.parquet")

    print("Loading feature dataset...")
    df = pd.read_parquet(input_path)

    print("Original shape:", df.shape)

    # Sort for clean time ordering
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Drop rows missing the target
    df = df.dropna(subset=["target_outperform_spy", "target_abs_direction", "future_12m_return"])

    # One-hot encode sector
    df = pd.get_dummies(df, columns=["sector"], prefix="sector", drop_first=False)

    # Identify columns that are not model input features
    non_feature_cols = [
        "date",
        "ticker",
        "future_12m_return",
        "future_12m_spy_return",
        "target_abs_direction",
        "target_outperform_spy",
    ]

    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    # Convert feature columns to numeric where possible
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace infinite values with NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Drop columns that are almost entirely missing
    missing_fraction = df[feature_cols].isna().mean()
    keep_feature_cols = missing_fraction[missing_fraction < 0.40].index.tolist()

    dropped_cols = sorted(set(feature_cols) - set(keep_feature_cols))
    if dropped_cols:
        print("Dropped mostly-missing columns:")
        print(dropped_cols)

    feature_cols = keep_feature_cols

    # Fill remaining missing values with median of each feature
    for col in feature_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    # Keep final columns
    final_cols = non_feature_cols + feature_cols
    df = df[final_cols]

    print("Cleaned shape:", df.shape)
    print("Target balance:")
    print(df["target_outperform_spy"].value_counts(normalize=True))

    df.to_parquet(output_path)
    print(f"Saved cleaned modeling dataset to {output_path}")


if __name__ == "__main__":
    clean_feature_dataset()