import os
import pandas as pd
import numpy as np


def clean_aligned_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Drop rows missing the 1-month target, because Week 12 focuses first on monthly alignment.
    df = df.dropna(
        subset=[
            "future_1m_return",
            "future_1m_spy_return",
            "future_1m_excess_return",
            "target_outperform_spy_1m",
            "target_top_quintile_1m",
        ]
    )

    # Fill 12-month target columns only where possible later; keep NaNs for dates without 12m future data.
    # Models using 12m target will drop missing 12m rows separately.

    non_numeric_keep = ["date", "ticker", "long_name"]

    for col in df.columns:
        if col not in non_numeric_keep:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    # Do not impute labels/returns. Only impute feature columns.
    target_cols = [
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

    feature_cols = [
        c for c in df.columns
        if c not in non_numeric_keep and c not in target_cols
    ]

    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    features_path = "data/processed/expanded_modeling_dataset_with_metadata.parquet"
    targets_path = "data/processed/week12_horizon_targets.parquet"

    output_path = "data/processed/week12_aligned_modeling_dataset.parquet"
    report_path = "outputs/reports/week12_aligned_dataset_summary.txt"

    print("Loading metadata-enhanced expanded features...")
    features = pd.read_parquet(features_path)
    features["date"] = pd.to_datetime(features["date"])

    print("Loading horizon targets...")
    targets = pd.read_parquet(targets_path)
    targets["date"] = pd.to_datetime(targets["date"])

    # Rename 12m target returns from new target file to avoid collision with previous Week 2/9 12m columns.
    targets = targets.rename(
        columns={
            "future_12m_return": "future_12m_return_new",
            "future_12m_spy_return": "future_12m_spy_return_new",
        }
    )

    print("Merging features with horizon targets...")
    merged = features.merge(targets, on=["date", "ticker"], how="left")

    print("Cleaning aligned dataset...")
    clean = clean_aligned_dataset(merged)

    clean.to_parquet(output_path)

    lines = []
    lines.append("Week 12 Aligned Modeling Dataset Summary")
    lines.append("=======================================")
    lines.append("")
    lines.append("Goal:")
    lines.append("Merge metadata-enhanced features with aligned 1-month and 12-month prediction targets.")
    lines.append("")
    lines.append(f"Input features shape: {features.shape}")
    lines.append(f"Horizon targets shape: {targets.shape}")
    lines.append(f"Output aligned shape: {clean.shape}")
    lines.append(f"Missing values total: {clean.isna().sum().sum()}")
    lines.append(f"Ticker count: {clean['ticker'].nunique()}")
    lines.append(f"Date range: {clean['date'].min()} to {clean['date'].max()}")
    lines.append("")
    lines.append("1-month outperformance balance:")
    lines.append(str(clean["target_outperform_spy_1m"].value_counts(normalize=True)))
    lines.append("")
    lines.append("1-month top-quintile balance:")
    lines.append(str(clean["target_top_quintile_1m"].value_counts(normalize=True)))
    lines.append("")
    lines.append("12-month non-missing rows:")
    lines.append(str(clean["target_outperform_spy_12m"].notna().sum()))
    lines.append("")
    lines.append("Output file:")
    lines.append(output_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("")
    print("Saved:", output_path)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()