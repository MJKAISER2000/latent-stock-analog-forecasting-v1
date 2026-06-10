import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


INPUT_PATH = "data/processed/week19_market_state_dataset.parquet"

LATENT_OUTPUT_PATH = "data/processed/week19_market_state_pca_latents.parquet"
MERGED_OUTPUT_PATH = "data/processed/week19_full500_with_market_pca_latents.parquet"
BASE_STOCK_DATASET_PATH = "data/processed/week15_full500_modeling_dataset.parquet"

REPORT_PATH = "outputs/reports/week19_market_state_pca_summary.txt"

N_COMPONENTS = 6
N_CLUSTERS = 4


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    print("Loading market-state dataset...")
    market = pd.read_parquet(INPUT_PATH)
    market["date"] = pd.to_datetime(market["date"])
    market = market.sort_values("date").reset_index(drop=True)

    feature_cols = [c for c in market.columns if c != "date"]

    X = market[feature_cols].copy()

    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are entirely missing.
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        print("Dropping all-NaN columns:", all_nan_cols)
        X = X.drop(columns=all_nan_cols)

    # Fill missing values safely.
    X = X.ffill().bfill()
    X = X.fillna(X.median())
    X = X.fillna(0.0)

    # Drop constant columns.
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_cols:
        print("Dropping constant columns:", constant_cols)
        X = X.drop(columns=constant_cols)

    # Final safety checks.
    if X.isna().sum().sum() > 0:
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaNs still present in columns: {bad_cols}")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Non-finite values still present in PCA matrix.")

    feature_cols = list(X.columns)

    print("Market-state shape:", market.shape)
    print("Feature count used for PCA:", len(feature_cols))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Extra safety after scaling.
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    Z = pca.fit_transform(X_scaled)

    latent = market[["date"]].copy()

    for i in range(N_COMPONENTS):
        latent[f"market_pca_z{i+1}"] = Z[:, i]

    # Reconstruction error = rough "unusual market state" feature.
    X_recon_scaled = pca.inverse_transform(Z)
    recon_error = np.mean((X_scaled - X_recon_scaled) ** 2, axis=1)
    latent["market_pca_reconstruction_error"] = recon_error

    # Cluster latent market states into rough regimes.
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    latent["market_pca_regime_cluster"] = kmeans.fit_predict(Z)

    distances = kmeans.transform(Z)

    for i in range(N_CLUSTERS):
        latent[f"market_pca_dist_cluster_{i}"] = distances[:, i]

    # Add interpretable columns for inspection only.
    inspect_cols = [
        "spy_ret_6m",
        "spy_ret_12m",
        "spy_drawdown",
        "equal_weight_ret_6m",
        "equal_weight_ret_12m",
        "equal_weight_drawdown",
        "sector_information_technology_ret_6m",
        "sector_information_technology_ret_12m",
        "sector_information_technology_minus_spy_6m",
        "sector_information_technology_minus_spy_12m",
        "pct_stocks_positive_ret_6m",
        "pct_stocks_positive_ret_12m",
        "pct_stocks_drawdown_worse_20",
        "pct_stocks_above_ma_12m",
    ]

    existing_inspect_cols = [c for c in inspect_cols if c in market.columns]

    latent_inspect = latent.merge(
        market[["date"] + existing_inspect_cols],
        on="date",
        how="left",
    )

    detail_rows = []

    for cluster, group in latent_inspect.groupby("market_pca_regime_cluster"):
        row = {
            "cluster": cluster,
            "months": len(group),
            "first_date": group["date"].min(),
            "last_date": group["date"].max(),
            "avg_reconstruction_error": group[
                "market_pca_reconstruction_error"
            ].mean(),
        }

        for col in existing_inspect_cols:
            row[f"avg_{col}"] = group[col].mean()

        detail_rows.append(row)

    cluster_detail = pd.DataFrame(detail_rows).sort_values("cluster")

    explained = pd.DataFrame(
        {
            "component": [f"market_pca_z{i+1}" for i in range(N_COMPONENTS)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )

    # Save latent market-state outputs.
    latent.to_parquet(LATENT_OUTPUT_PATH)
    latent_inspect.to_csv(
        "outputs/tables/week19_market_state_pca_latents_inspect.csv",
        index=False,
    )
    cluster_detail.to_csv(
        "outputs/tables/week19_market_state_pca_cluster_summary.csv",
        index=False,
    )
    explained.to_csv(
        "outputs/tables/week19_market_state_pca_explained_variance.csv",
        index=False,
    )

    # Merge latent market-state features into stock modeling dataset.
    print("Merging market PCA latents into stock modeling dataset...")
    stock_df = pd.read_parquet(BASE_STOCK_DATASET_PATH)
    stock_df["date"] = pd.to_datetime(stock_df["date"])

    merged = stock_df.merge(latent, on="date", how="left")

    latent_cols = [c for c in latent.columns if c != "date"]

    for col in latent_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged[col] = merged[col].fillna(merged[col].median())

    merged.to_parquet(MERGED_OUTPUT_PATH)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Week 19 Market-State PCA Latent Twin Summary\n")
        f.write("===========================================\n\n")
        f.write("Goal:\n")
        f.write(
            "Compress monthly market-state vectors into latent regime features.\n\n"
        )
        f.write(f"Input market-state shape: {market.shape}\n")
        f.write(f"PCA feature count used: {len(feature_cols)}\n")
        f.write(f"PCA components: {N_COMPONENTS}\n")
        f.write(f"Regime clusters: {N_CLUSTERS}\n\n")

        f.write("Explained variance:\n")
        f.write(explained.to_string(index=False))
        f.write("\n\n")

        f.write("Cluster summary:\n")
        f.write(cluster_detail.to_string(index=False))
        f.write("\n\n")

        f.write("Saved latent market state to:\n")
        f.write(LATENT_OUTPUT_PATH)
        f.write("\n\nSaved merged stock dataset to:\n")
        f.write(MERGED_OUTPUT_PATH)
        f.write("\n")

    print("")
    print("Saved:", LATENT_OUTPUT_PATH)
    print("Saved:", MERGED_OUTPUT_PATH)
    print("Saved:", REPORT_PATH)

    print("")
    print("EXPLAINED VARIANCE")
    print(explained.to_string(index=False))

    print("")
    print("CLUSTER SUMMARY")
    print(cluster_detail.to_string(index=False))


if __name__ == "__main__":
    main()