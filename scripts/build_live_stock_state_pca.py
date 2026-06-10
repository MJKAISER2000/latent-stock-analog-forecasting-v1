import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, ensure_output_dirs


CONFIG_PATH = "configs/final_model_config.yaml"

LIVE_BASE_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "live_full500_modeling_dataset.parquet"

LIVE_STATE_MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / "live_stock_state_matrix.parquet"
LIVE_STATE_MATRIX_SCALED_PATH = PROJECT_ROOT / "data" / "processed" / "live_stock_state_matrix_scaled.parquet"
LIVE_STATE_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "live_stock_state_metadata.parquet"

LIVE_PCA_LATENTS_PATH = PROJECT_ROOT / "data" / "processed" / "live_stock_state_pca_latents.parquet"
LIVE_PCA_LATENTS_WITH_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "live_stock_state_pca_latents_with_metadata.parquet"

FEATURE_LIST_PATH = PROJECT_ROOT / "outputs" / "tables" / "live_stock_state_feature_list.csv"
SCALER_STATS_PATH = PROJECT_ROOT / "outputs" / "tables" / "live_stock_state_scaler_stats.csv"
PCA_VARIANCE_PATH = PROJECT_ROOT / "outputs" / "tables" / "live_stock_state_pca_explained_variance.csv"
PCA_CLUSTER_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "live_stock_state_pca_cluster_summary.csv"

REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "live_stock_state_pca_report.txt"


N_COMPONENTS = 16
N_CLUSTERS = 4


NON_FEATURE_COLS = {
    "date",
    "ticker",
    "company",
    "sector",
    "industry",
    "row_id",

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
}


def is_leakage_column(col: str) -> bool:
    name = col.lower()

    if "future" in name:
        return True

    if "target" in name:
        return True

    if "ranking_label" in name:
        return True

    return False


def load_live_base_dataset() -> pd.DataFrame:
    if not LIVE_BASE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Live base dataset not found: {LIVE_BASE_DATASET_PATH}. "
            f"Run scripts/build_live_base_features.py first."
        )

    df = pd.read_parquet(LIVE_BASE_DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    return df


def select_state_feature_columns(df: pd.DataFrame) -> list[str]:
    candidate_cols = []

    for col in df.columns:
        if col in NON_FEATURE_COLS:
            continue

        if is_leakage_column(col):
            continue

        if col.startswith("neighbor_"):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            candidate_cols.append(col)

    # Remove obvious raw price scale because PCA should describe state, not price level.
    # Keep price_to_ma, returns, volatility, drawdown, sector dummies, industry dummies, etc.
    remove_cols = {
        "price",
        "spy_price",
    }

    feature_cols = [c for c in candidate_cols if c not in remove_cols]

    return feature_cols


def build_state_matrix(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_cols = ["date", "ticker", "company", "sector", "industry"]

    metadata = df[[c for c in metadata_cols if c in df.columns]].copy()

    X = df[feature_cols].copy()

    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    # Median impute using full available history. This is acceptable for a representation transform,
    # but later we can make this strictly walk-forward if needed.
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    X = X.fillna(0.0)

    return X, metadata


def scale_state_matrix(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled_array,
        columns=X.columns,
        index=X.index,
    )

    scaler_stats = pd.DataFrame(
        {
            "feature": X.columns,
            "mean": scaler.mean_,
            "scale": scaler.scale_,
        }
    )

    return X_scaled, scaler_stats


def build_pca_latents(
    X_scaled: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_components = min(N_COMPONENTS, X_scaled.shape[1])

    pca = PCA(n_components=n_components, random_state=42)
    z = pca.fit_transform(X_scaled)

    latent_cols = [f"stock_pca_z{i + 1}" for i in range(n_components)]

    latents = pd.DataFrame(z, columns=latent_cols, index=X_scaled.index)

    reconstruction = pca.inverse_transform(z)
    reconstruction_error = np.mean((X_scaled.values - reconstruction) ** 2, axis=1)

    latents["stock_pca_reconstruction_error"] = reconstruction_error

    variance = pd.DataFrame(
        {
            "component": latent_cols,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latents[latent_cols])

    latents["stock_pca_cluster"] = cluster_labels

    centers = kmeans.cluster_centers_

    for cluster_idx in range(N_CLUSTERS):
        center = centers[cluster_idx]
        diff = latents[latent_cols].values - center
        dist = np.sqrt(np.sum(diff ** 2, axis=1))
        latents[f"stock_pca_dist_cluster_{cluster_idx}"] = dist

    latents_with_metadata = pd.concat(
        [
            metadata.reset_index(drop=True),
            latents.reset_index(drop=True),
        ],
        axis=1,
    )

    cluster_summary = (
        latents_with_metadata.groupby("stock_pca_cluster")
        .agg(
            row_count=("ticker", "count"),
            ticker_count=("ticker", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            avg_reconstruction_error=("stock_pca_reconstruction_error", "mean"),
        )
        .reset_index()
    )

    return latents, latents_with_metadata, variance, cluster_summary


def write_report(
    df: pd.DataFrame,
    feature_cols: list[str],
    X: pd.DataFrame,
    X_scaled: pd.DataFrame,
    latents_with_metadata: pd.DataFrame,
    variance: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> None:
    lines = []
    lines.append("Latent Market Twin Live Stock-State PCA Report")
    lines.append("=============================================")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Input dataset: {LIVE_BASE_DATASET_PATH}")
    lines.append(f"Input shape: {df.shape}")
    lines.append(f"Input date range: {df['date'].min()} to {df['date'].max()}")
    lines.append(f"Input ticker count: {df['ticker'].nunique()}")
    lines.append("")
    lines.append(f"State matrix shape: {X.shape}")
    lines.append(f"Scaled matrix shape: {X_scaled.shape}")
    lines.append(f"Feature count: {len(feature_cols)}")
    lines.append("")
    lines.append("Top PCA explained variance:")
    lines.append(variance.head(20).to_string(index=False))
    lines.append("")
    lines.append("Cluster summary:")
    lines.append(cluster_summary.to_string(index=False))
    lines.append("")
    lines.append("Feature columns:")
    lines.append(", ".join(feature_cols))
    lines.append("")
    lines.append("Latent sample:")
    lines.append(latents_with_metadata.tail(30).to_string(index=False))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    config = load_config(CONFIG_PATH)
    ensure_output_dirs(config)

    os.makedirs(PROJECT_ROOT / "data" / "processed", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "outputs" / "tables", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "outputs" / "reports", exist_ok=True)

    df = load_live_base_dataset()

    print("")
    print("=" * 100)
    print("BUILDING LIVE STOCK-STATE PCA")
    print("=" * 100)
    print("Input shape:", df.shape)
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Ticker count:", df["ticker"].nunique())

    feature_cols = select_state_feature_columns(df)

    X, metadata = build_state_matrix(df, feature_cols)
    X_scaled, scaler_stats = scale_state_matrix(X)

    latents, latents_with_metadata, variance, cluster_summary = build_pca_latents(
        X_scaled=X_scaled,
        metadata=metadata,
    )

    feature_list = pd.DataFrame(
        {
            "feature": feature_cols,
            "feature_index": list(range(len(feature_cols))),
        }
    )

    X.to_parquet(LIVE_STATE_MATRIX_PATH)
    X_scaled.to_parquet(LIVE_STATE_MATRIX_SCALED_PATH)
    metadata.to_parquet(LIVE_STATE_METADATA_PATH)

    latents.to_parquet(LIVE_PCA_LATENTS_PATH)
    latents_with_metadata.to_parquet(LIVE_PCA_LATENTS_WITH_METADATA_PATH)

    feature_list.to_csv(FEATURE_LIST_PATH, index=False)
    scaler_stats.to_csv(SCALER_STATS_PATH, index=False)
    variance.to_csv(PCA_VARIANCE_PATH, index=False)
    cluster_summary.to_csv(PCA_CLUSTER_SUMMARY_PATH, index=False)

    write_report(
        df=df,
        feature_cols=feature_cols,
        X=X,
        X_scaled=X_scaled,
        latents_with_metadata=latents_with_metadata,
        variance=variance,
        cluster_summary=cluster_summary,
    )

    print("")
    print("=" * 100)
    print("LIVE STOCK-STATE PCA COMPLETE")
    print("=" * 100)
    print("Feature count:", len(feature_cols))
    print("State matrix shape:", X.shape)
    print("Latents shape:", latents_with_metadata.shape)
    print("")
    print("Saved state matrix:", LIVE_STATE_MATRIX_PATH)
    print("Saved scaled state matrix:", LIVE_STATE_MATRIX_SCALED_PATH)
    print("Saved metadata:", LIVE_STATE_METADATA_PATH)
    print("Saved latents:", LIVE_PCA_LATENTS_WITH_METADATA_PATH)
    print("Saved feature list:", FEATURE_LIST_PATH)
    print("Saved PCA variance:", PCA_VARIANCE_PATH)
    print("Saved report:", REPORT_PATH)
    print("")
    print("PCA VARIANCE")
    print(variance.to_string(index=False))
    print("")
    print("CLUSTER SUMMARY")
    print(cluster_summary.to_string(index=False))
    print("")
    print("LATENT SAMPLE")
    print(latents_with_metadata.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()