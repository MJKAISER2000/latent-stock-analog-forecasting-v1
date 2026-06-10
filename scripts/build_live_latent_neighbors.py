import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, ensure_output_dirs


CONFIG_PATH = "configs/final_model_config.yaml"

LIVE_BASE_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "live_full500_modeling_dataset.parquet"
LIVE_PCA_LATENTS_WITH_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "live_stock_state_pca_latents_with_metadata.parquet"

LIVE_NEIGHBOR_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "live_stock_latent_neighbor_features.parquet"
LIVE_FULL_WITH_NEIGHBORS_PATH = PROJECT_ROOT / "data" / "processed" / "live_full500_with_stock_latent_neighbors.parquet"

LIVE_NEIGHBOR_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "live_latent_neighbor_feature_summary.csv"
LIVE_NEIGHBOR_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "live_latent_neighbor_feature_report.txt"


N_NEIGHBORS = 50
MIN_HISTORY_ROWS = 250


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not LIVE_BASE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Live base dataset not found: {LIVE_BASE_DATASET_PATH}. "
            f"Run scripts/build_live_base_features.py first."
        )

    if not LIVE_PCA_LATENTS_WITH_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Live PCA latents not found: {LIVE_PCA_LATENTS_WITH_METADATA_PATH}. "
            f"Run scripts/build_live_stock_state_pca.py first."
        )

    base = pd.read_parquet(LIVE_BASE_DATASET_PATH)
    latents = pd.read_parquet(LIVE_PCA_LATENTS_WITH_METADATA_PATH)

    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    latents["date"] = pd.to_datetime(latents["date"]).dt.normalize()

    base["ticker"] = base["ticker"].astype(str).str.strip().str.upper()
    latents["ticker"] = latents["ticker"].astype(str).str.strip().str.upper()

    base = base.sort_values(["date", "ticker"]).reset_index(drop=True)
    latents = latents.sort_values(["date", "ticker"]).reset_index(drop=True)

    return base, latents


def get_latent_columns(latents: pd.DataFrame) -> list[str]:
    latent_cols = [
        c for c in latents.columns
        if c.startswith("stock_pca_z")
    ]

    if len(latent_cols) == 0:
        raise ValueError("No stock_pca_z columns found in latent file.")

    latent_cols = sorted(
        latent_cols,
        key=lambda x: int(x.replace("stock_pca_z", "")),
    )

    return latent_cols


def merge_outcomes_into_latents(
    base: pd.DataFrame,
    latents: pd.DataFrame,
) -> pd.DataFrame:
    outcome_cols = [
        "date",
        "ticker",
        "future_1m_return",
        "future_1m_spy_return",
        "future_1m_excess_return",
        "target_outperform_spy_1m",
        "ranking_label",
    ]

    available_cols = [c for c in outcome_cols if c in base.columns]

    merged = latents.merge(
        base[available_cols],
        on=["date", "ticker"],
        how="left",
    )

    return merged


def summarize_neighbors(neighbor_rows: pd.DataFrame, distances: np.ndarray) -> dict:
    valid = neighbor_rows.dropna(subset=["future_1m_return"]).copy()

    if len(valid) == 0:
        return {
            "neighbor_count": 0,
            "neighbor_distance_mean": np.nan,
            "neighbor_distance_median": np.nan,
            "neighbor_distance_min": np.nan,
            "neighbor_avg_future_1m_return": np.nan,
            "neighbor_median_future_1m_return": np.nan,
            "neighbor_avg_future_1m_excess_return": np.nan,
            "neighbor_outperform_spy_1m_rate": np.nan,
            "neighbor_positive_1m_return_rate": np.nan,
        }

    valid_distances = distances[neighbor_rows.index.get_indexer(valid.index)]

    return {
        "neighbor_count": int(len(valid)),
        "neighbor_distance_mean": float(np.nanmean(valid_distances)),
        "neighbor_distance_median": float(np.nanmedian(valid_distances)),
        "neighbor_distance_min": float(np.nanmin(valid_distances)),
        "neighbor_avg_future_1m_return": float(valid["future_1m_return"].mean()),
        "neighbor_median_future_1m_return": float(valid["future_1m_return"].median()),
        "neighbor_avg_future_1m_excess_return": float(valid["future_1m_excess_return"].mean()),
        "neighbor_outperform_spy_1m_rate": float((valid["future_1m_excess_return"] > 0).mean()),
        "neighbor_positive_1m_return_rate": float((valid["future_1m_return"] > 0).mean()),
    }


def build_neighbor_features(
    latent_outcomes: pd.DataFrame,
    latent_cols: list[str],
) -> pd.DataFrame:
    rows = []

    all_dates = sorted(latent_outcomes["date"].dropna().unique())

    print("")
    print("=" * 100)
    print("BUILDING LIVE LATENT-NEIGHBOR FEATURES")
    print("=" * 100)
    print("Dates:", len(all_dates))
    print("Rows:", len(latent_outcomes))
    print("Latent cols:", latent_cols)

    for idx, current_date in enumerate(all_dates):
        current_date = pd.Timestamp(current_date).normalize()

        current_rows = latent_outcomes[
            latent_outcomes["date"] == current_date
        ].copy()

        history = latent_outcomes[
            latent_outcomes["date"] < current_date
        ].copy()

        history = history.dropna(subset=latent_cols)

        # Only rows with known future outcomes should be used as historical analogs.
        history = history.dropna(subset=["future_1m_return"])

        if len(history) < MIN_HISTORY_ROWS:
            for _, row in current_rows.iterrows():
                rows.append(
                    {
                        "date": row["date"],
                        "ticker": row["ticker"],
                        "neighbor_count": 0,
                        "neighbor_distance_mean": np.nan,
                        "neighbor_distance_median": np.nan,
                        "neighbor_distance_min": np.nan,
                        "neighbor_avg_future_1m_return": np.nan,
                        "neighbor_median_future_1m_return": np.nan,
                        "neighbor_avg_future_1m_excess_return": np.nan,
                        "neighbor_outperform_spy_1m_rate": np.nan,
                        "neighbor_positive_1m_return_rate": np.nan,
                    }
                )

            continue

        X_hist = history[latent_cols].to_numpy(dtype=float)
        n_neighbors = min(N_NEIGHBORS, len(history))

        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="euclidean",
            algorithm="auto",
        )

        nn.fit(X_hist)

        X_current = current_rows[latent_cols].to_numpy(dtype=float)

        distances_matrix, indices_matrix = nn.kneighbors(X_current)

        for row_position, (_, current_row) in enumerate(current_rows.iterrows()):
            neighbor_indices = indices_matrix[row_position]
            distances = distances_matrix[row_position]

            neighbor_rows = history.iloc[neighbor_indices].copy()
            neighbor_rows.index = range(len(neighbor_rows))

            stats = summarize_neighbors(
                neighbor_rows=neighbor_rows,
                distances=distances,
            )

            rows.append(
                {
                    "date": current_row["date"],
                    "ticker": current_row["ticker"],
                    **stats,
                }
            )

        if idx % 12 == 0 or idx == len(all_dates) - 1:
            print(
                f"Processed date {idx + 1}/{len(all_dates)}: "
                f"{current_date.date()} | history rows={len(history)}"
            )

    features = pd.DataFrame(rows)
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    features["ticker"] = features["ticker"].astype(str).str.strip().str.upper()

    features = features.sort_values(["date", "ticker"]).reset_index(drop=True)

    return features


def merge_neighbor_features(
    base: pd.DataFrame,
    neighbor_features: pd.DataFrame,
) -> pd.DataFrame:
    merged = base.merge(
        neighbor_features,
        on=["date", "ticker"],
        how="left",
    )

    neighbor_cols = [
        c for c in neighbor_features.columns
        if c not in ["date", "ticker"]
    ]

    for col in neighbor_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    return merged


def build_summary(neighbor_features: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        c for c in neighbor_features.columns
        if c not in ["date", "ticker"]
    ]

    rows = []

    for col in numeric_cols:
        series = pd.to_numeric(neighbor_features[col], errors="coerce")

        rows.append(
            {
                "feature": col,
                "missing_count": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean()),
                "mean": float(series.mean(skipna=True)) if series.notna().any() else np.nan,
                "std": float(series.std(skipna=True)) if series.notna().any() else np.nan,
                "min": float(series.min(skipna=True)) if series.notna().any() else np.nan,
                "max": float(series.max(skipna=True)) if series.notna().any() else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False).reset_index(drop=True)


def write_report(
    base: pd.DataFrame,
    latent_outcomes: pd.DataFrame,
    neighbor_features: pd.DataFrame,
    merged: pd.DataFrame,
    summary: pd.DataFrame,
    latent_cols: list[str],
) -> None:
    latest_date = merged["date"].max()
    latest = merged[merged["date"] == latest_date].copy()

    lines = []
    lines.append("Latent Market Twin Live Latent-Neighbor Feature Report")
    lines.append("=====================================================")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Base dataset: {LIVE_BASE_DATASET_PATH}")
    lines.append(f"Latents: {LIVE_PCA_LATENTS_WITH_METADATA_PATH}")
    lines.append("")
    lines.append(f"Base shape: {base.shape}")
    lines.append(f"Latent outcomes shape: {latent_outcomes.shape}")
    lines.append(f"Neighbor feature shape: {neighbor_features.shape}")
    lines.append(f"Merged shape: {merged.shape}")
    lines.append("")
    lines.append(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    lines.append(f"Ticker count: {merged['ticker'].nunique()}")
    lines.append(f"Latent columns: {', '.join(latent_cols)}")
    lines.append("")
    lines.append("Neighbor feature summary:")
    lines.append(summary.to_string(index=False))
    lines.append("")
    lines.append(f"Latest date sample: {latest_date}")
    sample_cols = [
        "date",
        "ticker",
        "future_1m_return",
        "neighbor_count",
        "neighbor_distance_mean",
        "neighbor_avg_future_1m_return",
        "neighbor_avg_future_1m_excess_return",
        "neighbor_outperform_spy_1m_rate",
        "neighbor_positive_1m_return_rate",
    ]
    sample_cols = [c for c in sample_cols if c in latest.columns]
    lines.append(latest[sample_cols].head(80).to_string(index=False))

    with open(LIVE_NEIGHBOR_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    config = load_config(CONFIG_PATH)
    ensure_output_dirs(config)

    os.makedirs(PROJECT_ROOT / "data" / "processed", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "outputs" / "tables", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "outputs" / "reports", exist_ok=True)

    base, latents = load_inputs()
    latent_cols = get_latent_columns(latents)

    latent_outcomes = merge_outcomes_into_latents(
        base=base,
        latents=latents,
    )

    neighbor_features = build_neighbor_features(
        latent_outcomes=latent_outcomes,
        latent_cols=latent_cols,
    )

    merged = merge_neighbor_features(
        base=base,
        neighbor_features=neighbor_features,
    )

    summary = build_summary(neighbor_features)

    neighbor_features.to_parquet(LIVE_NEIGHBOR_FEATURES_PATH)
    merged.to_parquet(LIVE_FULL_WITH_NEIGHBORS_PATH)
    summary.to_csv(LIVE_NEIGHBOR_SUMMARY_PATH, index=False)

    write_report(
        base=base,
        latent_outcomes=latent_outcomes,
        neighbor_features=neighbor_features,
        merged=merged,
        summary=summary,
        latent_cols=latent_cols,
    )

    latest_date = merged["date"].max()
    latest = merged[merged["date"] == latest_date].copy()

    print("")
    print("=" * 100)
    print("LIVE LATENT-NEIGHBOR FEATURES COMPLETE")
    print("=" * 100)
    print("Neighbor feature shape:", neighbor_features.shape)
    print("Merged live dataset shape:", merged.shape)
    print("Date range:", merged["date"].min(), "to", merged["date"].max())
    print("Ticker count:", merged["ticker"].nunique())
    print("")
    print("Saved neighbor features:", LIVE_NEIGHBOR_FEATURES_PATH)
    print("Saved merged dataset:", LIVE_FULL_WITH_NEIGHBORS_PATH)
    print("Saved summary:", LIVE_NEIGHBOR_SUMMARY_PATH)
    print("Saved report:", LIVE_NEIGHBOR_REPORT_PATH)
    print("")
    print("NEIGHBOR SUMMARY")
    print(summary.to_string(index=False))
    print("")
    print("LATEST DATE SAMPLE")
    sample_cols = [
        "date",
        "ticker",
        "future_1m_return",
        "neighbor_count",
        "neighbor_distance_mean",
        "neighbor_avg_future_1m_return",
        "neighbor_avg_future_1m_excess_return",
        "neighbor_outperform_spy_1m_rate",
        "neighbor_positive_1m_return_rate",
    ]
    sample_cols = [c for c in sample_cols if c in latest.columns]
    print(latest[sample_cols].head(80).to_string(index=False))


if __name__ == "__main__":
    main()