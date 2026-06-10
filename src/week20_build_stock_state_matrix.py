import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


INPUT_PATH = "data/processed/week15_full500_modeling_dataset.parquet"

OUTPUT_MATRIX_PATH = "data/processed/week20_stock_state_matrix.parquet"
OUTPUT_SCALED_MATRIX_PATH = "data/processed/week20_stock_state_matrix_scaled.parquet"
OUTPUT_METADATA_PATH = "data/processed/week20_stock_state_metadata.parquet"
OUTPUT_FEATURE_LIST_PATH = "outputs/tables/week20_stock_state_feature_list.csv"
OUTPUT_SCALER_STATS_PATH = "outputs/tables/week20_stock_state_scaler_stats.csv"
REPORT_PATH = "outputs/reports/week20_stock_state_matrix_summary.txt"


TARGET_AND_NON_FEATURE_COLS = [
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

    "ranking_label",
]


def is_bad_feature_name(col: str) -> bool:
    name = col.lower()

    if "future" in name:
        return True

    if "target" in name:
        return True

    if "ranking_label" in name:
        return True

    return False


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    print("Loading base modeling dataset...")
    df = pd.read_parquet(INPUT_PATH)

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    print("Input shape:", df.shape)
    print("Ticker count:", df["ticker"].nunique())
    print("Date range:", df["date"].min(), "to", df["date"].max())

    candidate_features = [
        c for c in df.columns
        if c not in TARGET_AND_NON_FEATURE_COLS
        and not is_bad_feature_name(c)
    ]

    numeric_features = []

    for col in candidate_features:
        converted = pd.to_numeric(df[col], errors="coerce")

        if converted.notna().sum() > 0:
            df[col] = converted
            numeric_features.append(col)

    print("Candidate feature count:", len(candidate_features))
    print("Numeric feature count:", len(numeric_features))

    if len(numeric_features) == 0:
        raise ValueError("No numeric features found.")

    metadata_cols = ["date", "ticker"]

    optional_metadata_cols = [
        "company",
        "future_1m_return",
        "future_1m_spy_return",
        "future_1m_excess_return",
        "target_outperform_spy_1m",
        "future_36m_return",
        "future_36m_spy_return",
        "future_36m_excess_return",
        "target_outperform_spy_36m",
    ]

    for col in optional_metadata_cols:
        if col in df.columns:
            metadata_cols.append(col)

    metadata = df[metadata_cols].copy().reset_index(drop=True)
    metadata["row_id"] = np.arange(len(metadata))

    X = df[numeric_features].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        print("Dropping all-NaN columns:", all_nan_cols)
        X = X.drop(columns=all_nan_cols)

    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0.0)

    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_cols:
        print("Dropping constant columns:", constant_cols)
        X = X.drop(columns=constant_cols)

    final_features = list(X.columns)

    leakage_cols = [c for c in final_features if is_bad_feature_name(c)]
    if leakage_cols:
        raise ValueError(f"Leakage-like features detected: {leakage_cols}")

    X = X.astype(float)

    if X.isna().sum().sum() > 0:
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaNs still present in feature matrix: {bad_cols}")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values still present in stock-state matrix.")

    print("Final feature count:", len(final_features))
    print("Final matrix shape:", X.shape)

    scaler = StandardScaler()
    X_scaled_np = scaler.fit_transform(X)
    X_scaled_np = np.nan_to_num(X_scaled_np, nan=0.0, posinf=0.0, neginf=0.0)

    X_scaled = pd.DataFrame(
        X_scaled_np,
        columns=final_features,
    )

    X_out = X.reset_index(drop=True).copy()
    X_out["row_id"] = metadata["row_id"]

    X_scaled_out = X_scaled.reset_index(drop=True).copy()
    X_scaled_out["row_id"] = metadata["row_id"]

    feature_list = pd.DataFrame(
        {
            "feature": final_features,
            "feature_index": np.arange(len(final_features)),
        }
    )

    scaler_stats = pd.DataFrame(
        {
            "feature": final_features,
            "mean": scaler.mean_,
            "scale": scaler.scale_,
            "var": scaler.var_,
        }
    )

    metadata.to_parquet(OUTPUT_METADATA_PATH, index=False)
    X_out.to_parquet(OUTPUT_MATRIX_PATH, index=False)
    X_scaled_out.to_parquet(OUTPUT_SCALED_MATRIX_PATH, index=False)
    feature_list.to_csv(OUTPUT_FEATURE_LIST_PATH, index=False)
    scaler_stats.to_csv(OUTPUT_SCALER_STATS_PATH, index=False)

    lines = []
    lines.append("Week 20 Stock-State Matrix Summary")
    lines.append("==================================")
    lines.append("")
    lines.append("Goal:")
    lines.append("Build a clean stock/month feature matrix for stock-level latent twin encoding.")
    lines.append("")
    lines.append(f"Input path: {INPUT_PATH}")
    lines.append(f"Input shape: {df.shape}")
    lines.append(f"Ticker count: {df['ticker'].nunique()}")
    lines.append(f"Date range: {df['date'].min()} to {df['date'].max()}")
    lines.append("")
    lines.append(f"Candidate feature count: {len(candidate_features)}")
    lines.append(f"Numeric feature count: {len(numeric_features)}")
    lines.append(f"Final feature count: {len(final_features)}")
    lines.append(f"Stock-state matrix shape: {X_out.shape}")
    lines.append("")
    lines.append("Leakage check:")
    lines.append("No future/target/ranking-label columns included.")
    lines.append("")
    lines.append("Outputs:")
    lines.append(f"- {OUTPUT_MATRIX_PATH}")
    lines.append(f"- {OUTPUT_SCALED_MATRIX_PATH}")
    lines.append(f"- {OUTPUT_METADATA_PATH}")
    lines.append(f"- {OUTPUT_FEATURE_LIST_PATH}")
    lines.append(f"- {OUTPUT_SCALER_STATS_PATH}")
    lines.append("")
    lines.append("Example features:")
    lines.append(", ".join(final_features[:80]))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("")
    print("\n".join(lines))
    print("")
    print("Saved:", OUTPUT_MATRIX_PATH)
    print("Saved:", OUTPUT_SCALED_MATRIX_PATH)
    print("Saved:", OUTPUT_METADATA_PATH)
    print("Saved:", OUTPUT_FEATURE_LIST_PATH)
    print("Saved:", OUTPUT_SCALER_STATS_PATH)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()