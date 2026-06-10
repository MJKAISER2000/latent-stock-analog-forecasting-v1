import os
import pandas as pd
import numpy as np


BASE_DATASET_PATH = "data/processed/week15_full500_modeling_dataset.parquet"
NEIGHBOR_FEATURES_PATH = "data/processed/week20_stock_latent_neighbor_features.parquet"
STOCK_METADATA_PATH = "data/processed/week20_stock_state_metadata.parquet"

OUTPUT_PATH = "data/processed/week20_full500_with_stock_latent_neighbors.parquet"
REPORT_PATH = "outputs/reports/week20_merge_neighbor_features_summary.txt"


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    print("Loading base modeling dataset...")
    base = pd.read_parquet(BASE_DATASET_PATH)
    base["date"] = pd.to_datetime(base["date"])
    base["ticker"] = base["ticker"].astype(str).str.strip().str.upper()
    base = base.sort_values(["date", "ticker"]).reset_index(drop=True)

    print("Loading stock-state metadata...")
    metadata = pd.read_parquet(STOCK_METADATA_PATH)
    metadata["date"] = pd.to_datetime(metadata["date"])
    metadata["ticker"] = metadata["ticker"].astype(str).str.strip().str.upper()

    print("Loading neighbor features...")
    neighbors = pd.read_parquet(NEIGHBOR_FEATURES_PATH)
    neighbors["date"] = pd.to_datetime(neighbors["date"])
    neighbors["ticker"] = neighbors["ticker"].astype(str).str.strip().str.upper()

    print("Base shape:", base.shape)
    print("Metadata shape:", metadata.shape)
    print("Neighbor shape:", neighbors.shape)

    if "row_id" not in metadata.columns:
        raise ValueError("row_id missing from metadata.")

    if "row_id" not in neighbors.columns:
        raise ValueError("row_id missing from neighbor features.")

    # Keep row_id mapping from metadata.
    row_map = metadata[["row_id", "date", "ticker"]].copy()

    base_with_row = base.merge(
        row_map,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )

    if base_with_row["row_id"].isna().sum() > 0:
        missing = base_with_row["row_id"].isna().sum()
        raise ValueError(f"Missing row_id for {missing} base rows.")

    neighbor_cols = [
        c for c in neighbors.columns
        if c not in ["date", "ticker"]
    ]

    merged = base_with_row.merge(
        neighbors[neighbor_cols],
        on="row_id",
        how="left",
        validate="one_to_one",
    )

    neighbor_feature_cols = [
        c for c in neighbors.columns
        if c not in ["row_id", "date", "ticker"]
    ]

    for col in neighbor_feature_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged[col] = merged[col].replace([np.inf, -np.inf], np.nan)

        median_val = merged[col].median()
        if pd.isna(median_val):
            median_val = 0.0

        merged[col] = merged[col].fillna(median_val)

    # row_id can stay for debugging but should later be excluded from training.
    merged = merged.sort_values(["date", "ticker"]).reset_index(drop=True)

    merged.to_parquet(OUTPUT_PATH)

    feature_summary = pd.DataFrame(
        {
            "neighbor_feature": neighbor_feature_cols,
            "missing_count": [merged[c].isna().sum() for c in neighbor_feature_cols],
            "mean": [merged[c].mean() for c in neighbor_feature_cols],
            "std": [merged[c].std() for c in neighbor_feature_cols],
            "min": [merged[c].min() for c in neighbor_feature_cols],
            "max": [merged[c].max() for c in neighbor_feature_cols],
        }
    )

    feature_summary_path = "outputs/tables/week20_neighbor_feature_summary.csv"
    feature_summary.to_csv(feature_summary_path, index=False)

    lines = []
    lines.append("Week 20 Merge Neighbor Features Summary")
    lines.append("======================================")
    lines.append("")
    lines.append("Goal:")
    lines.append("Merge stock-level latent nearest-neighbor features into the full modeling dataset.")
    lines.append("")
    lines.append(f"Base dataset path: {BASE_DATASET_PATH}")
    lines.append(f"Neighbor features path: {NEIGHBOR_FEATURES_PATH}")
    lines.append(f"Output path: {OUTPUT_PATH}")
    lines.append("")
    lines.append(f"Base shape: {base.shape}")
    lines.append(f"Output shape: {merged.shape}")
    lines.append(f"Neighbor feature count: {len(neighbor_feature_cols)}")
    lines.append("")
    lines.append("Neighbor features:")
    lines.append(", ".join(neighbor_feature_cols))
    lines.append("")
    lines.append("Feature summary:")
    lines.append(feature_summary.to_string(index=False))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("")
    print("\n".join(lines))
    print("")
    print("Saved:", OUTPUT_PATH)
    print("Saved:", feature_summary_path)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()