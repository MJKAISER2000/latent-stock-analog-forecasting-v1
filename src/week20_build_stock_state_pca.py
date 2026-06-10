import os
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


SCALED_MATRIX_PATH = "data/processed/week20_stock_state_matrix_scaled.parquet"
METADATA_PATH = "data/processed/week20_stock_state_metadata.parquet"

LATENT_OUTPUT_PATH = "data/processed/week20_stock_state_pca_latents.parquet"
MERGED_OUTPUT_PATH = "data/processed/week20_stock_state_pca_latents_with_metadata.parquet"

EXPLAINED_VARIANCE_PATH = "outputs/tables/week20_stock_state_pca_explained_variance.csv"
CLUSTER_SUMMARY_PATH = "outputs/tables/week20_stock_state_pca_cluster_summary.csv"
REPORT_PATH = "outputs/reports/week20_stock_state_pca_summary.txt"

N_COMPONENTS = 16
N_CLUSTERS = 12


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    print("Loading scaled stock-state matrix...")
    X = pd.read_parquet(SCALED_MATRIX_PATH)

    print("Loading metadata...")
    metadata = pd.read_parquet(METADATA_PATH)

    if "row_id" not in X.columns:
        raise ValueError("row_id missing from scaled matrix.")

    if "row_id" not in metadata.columns:
        raise ValueError("row_id missing from metadata.")

    X = X.sort_values("row_id").reset_index(drop=True)
    metadata = metadata.sort_values("row_id").reset_index(drop=True)

    if not X["row_id"].equals(metadata["row_id"]):
        raise ValueError("row_id alignment mismatch between X and metadata.")

    feature_cols = [c for c in X.columns if c != "row_id"]

    X_values = X[feature_cols].copy()
    X_values = X_values.apply(pd.to_numeric, errors="coerce")
    X_values = X_values.replace([np.inf, -np.inf], np.nan)
    X_values = X_values.fillna(0.0)
    X_values = X_values.astype(float)

    if not np.isfinite(X_values.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values in scaled stock-state matrix.")

    print("Scaled matrix shape:", X_values.shape)
    print("Metadata shape:", metadata.shape)
    print("Feature count:", len(feature_cols))

    n_components = min(N_COMPONENTS, X_values.shape[1])
    print("Using PCA components:", n_components)

    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X_values.to_numpy(dtype=float))

    latents = pd.DataFrame(
        {
            "row_id": X["row_id"].values,
        }
    )

    for i in range(n_components):
        latents[f"stock_pca_z{i+1}"] = Z[:, i]

    X_recon = pca.inverse_transform(Z)
    recon_error = np.mean((X_values.to_numpy(dtype=float) - X_recon) ** 2, axis=1)
    latents["stock_pca_reconstruction_error"] = recon_error

    print("Training KMeans clusters in stock latent space...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    latents["stock_pca_cluster"] = kmeans.fit_predict(Z)

    distances = kmeans.transform(Z)

    for i in range(N_CLUSTERS):
        latents[f"stock_pca_dist_cluster_{i}"] = distances[:, i]

    merged = metadata.merge(latents, on="row_id", how="inner")

    if len(merged) != len(metadata):
        raise ValueError("Merged latent metadata row count mismatch.")

    explained = pd.DataFrame(
        {
            "component": [f"stock_pca_z{i+1}" for i in range(n_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )

    cluster_rows = []

    for cluster, group in merged.groupby("stock_pca_cluster"):
        row = {
            "cluster": cluster,
            "rows": len(group),
            "unique_tickers": group["ticker"].nunique(),
            "first_date": group["date"].min(),
            "last_date": group["date"].max(),
            "avg_reconstruction_error": group[
                "stock_pca_reconstruction_error"
            ].mean(),
        }

        if "future_1m_return" in group.columns:
            row["avg_future_1m_return"] = group["future_1m_return"].mean()
            row["median_future_1m_return"] = group["future_1m_return"].median()

        if "future_1m_excess_return" in group.columns:
            row["avg_future_1m_excess_return"] = group[
                "future_1m_excess_return"
            ].mean()

        if "target_outperform_spy_1m" in group.columns:
            row["outperform_spy_1m_rate"] = group[
                "target_outperform_spy_1m"
            ].mean()

        cluster_rows.append(row)

    cluster_summary = pd.DataFrame(cluster_rows).sort_values("cluster")

    latents.to_parquet(LATENT_OUTPUT_PATH, index=False)
    merged.to_parquet(MERGED_OUTPUT_PATH, index=False)
    explained.to_csv(EXPLAINED_VARIANCE_PATH, index=False)
    cluster_summary.to_csv(CLUSTER_SUMMARY_PATH, index=False)

    lines = []
    lines.append("Week 20 Stock-State PCA Latent Twin Summary")
    lines.append("==========================================")
    lines.append("")
    lines.append("Goal:")
    lines.append("Compress each stock/month feature vector into a stock-level latent state.")
    lines.append("")
    lines.append(f"Input matrix shape: {X_values.shape}")
    lines.append(f"Feature count: {len(feature_cols)}")
    lines.append(f"PCA components: {n_components}")
    lines.append(f"KMeans clusters: {N_CLUSTERS}")
    lines.append("")
    lines.append("Explained variance:")
    lines.append(explained.to_string(index=False))
    lines.append("")
    lines.append("Cluster summary:")
    lines.append(cluster_summary.to_string(index=False))
    lines.append("")
    lines.append("Outputs:")
    lines.append(f"- {LATENT_OUTPUT_PATH}")
    lines.append(f"- {MERGED_OUTPUT_PATH}")
    lines.append(f"- {EXPLAINED_VARIANCE_PATH}")
    lines.append(f"- {CLUSTER_SUMMARY_PATH}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("")
    print("Saved:", LATENT_OUTPUT_PATH)
    print("Saved:", MERGED_OUTPUT_PATH)
    print("Saved:", EXPLAINED_VARIANCE_PATH)
    print("Saved:", CLUSTER_SUMMARY_PATH)
    print("Saved:", REPORT_PATH)

    print("")
    print("EXPLAINED VARIANCE")
    print(explained.to_string(index=False))

    print("")
    print("CLUSTER SUMMARY")
    print(cluster_summary.to_string(index=False))


if __name__ == "__main__":
    main()