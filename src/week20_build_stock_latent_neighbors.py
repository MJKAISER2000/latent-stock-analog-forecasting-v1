import os
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors


LATENT_METADATA_PATH = "data/processed/week20_stock_state_pca_latents_with_metadata.parquet"

OUTPUT_PATH = "data/processed/week20_stock_latent_neighbor_features.parquet"
REPORT_PATH = "outputs/reports/week20_stock_latent_neighbor_features_summary.txt"

K_NEIGHBORS = 50
MIN_HISTORY_ROWS = 500
LATENT_PREFIX = "stock_pca_z"


def get_latent_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith(LATENT_PREFIX)]

    def sort_key(col: str) -> int:
        return int(col.replace(LATENT_PREFIX, ""))

    return sorted(cols, key=sort_key)


def safe_mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() == 0:
        return np.nan
    return float(x.mean())


def safe_median(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() == 0:
        return np.nan
    return float(x.median())


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    print("Loading stock PCA latent metadata...")
    df = pd.read_parquet(LATENT_METADATA_PATH)

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    latent_cols = get_latent_cols(df)

    if len(latent_cols) == 0:
        raise ValueError("No stock PCA latent columns found.")

    print("Input shape:", df.shape)
    print("Latent columns:", latent_cols)
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Ticker count:", df["ticker"].nunique())

    required_cols = [
        "row_id",
        "date",
        "ticker",
        "future_1m_return",
        "future_1m_excess_return",
        "target_outperform_spy_1m",
    ]

    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    for col in latent_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[latent_cols] = df[latent_cols].replace([np.inf, -np.inf], np.nan)
    df[latent_cols] = df[latent_cols].fillna(0.0)

    dates = sorted(df["date"].unique())

    all_rows = []

    print("Building historical latent-neighbor features...")

    for i, current_date in enumerate(dates):
        current = df[df["date"] == current_date].copy()
        history = df[df["date"] < current_date].copy()

        if len(current) == 0:
            continue

        if len(history) < MIN_HISTORY_ROWS:
            # Not enough prior history. Fill with NaNs for this month.
            for _, row in current.iterrows():
                all_rows.append(
                    {
                        "row_id": row["row_id"],
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

        k = min(K_NEIGHBORS, len(history))

        X_hist = history[latent_cols].to_numpy(dtype=float)
        X_curr = current[latent_cols].to_numpy(dtype=float)

        nbrs = NearestNeighbors(
            n_neighbors=k,
            metric="euclidean",
            algorithm="auto",
        )

        nbrs.fit(X_hist)
        distances, indices = nbrs.kneighbors(X_curr)

        history_reset = history.reset_index(drop=True)

        for row_pos, (_, row) in enumerate(current.iterrows()):
            idx = indices[row_pos]
            dist = distances[row_pos]

            neighbor_rows = history_reset.iloc[idx].copy()

            future_returns = pd.to_numeric(
                neighbor_rows["future_1m_return"],
                errors="coerce",
            )

            future_excess = pd.to_numeric(
                neighbor_rows["future_1m_excess_return"],
                errors="coerce",
            )

            outperform = pd.to_numeric(
                neighbor_rows["target_outperform_spy_1m"],
                errors="coerce",
            )

            all_rows.append(
                {
                    "row_id": row["row_id"],
                    "date": row["date"],
                    "ticker": row["ticker"],

                    "neighbor_count": len(neighbor_rows),
                    "neighbor_distance_mean": float(np.mean(dist)),
                    "neighbor_distance_median": float(np.median(dist)),
                    "neighbor_distance_min": float(np.min(dist)),

                    "neighbor_avg_future_1m_return": safe_mean(future_returns),
                    "neighbor_median_future_1m_return": safe_median(future_returns),
                    "neighbor_avg_future_1m_excess_return": safe_mean(future_excess),
                    "neighbor_outperform_spy_1m_rate": safe_mean(outperform),
                    "neighbor_positive_1m_return_rate": float((future_returns > 0).mean()),
                }
            )

        if i % 12 == 0:
            print(f"Processed {i+1}/{len(dates)} dates | current_date={pd.Timestamp(current_date).date()} | history_rows={len(history)}")

    neighbor_features = pd.DataFrame(all_rows)

    neighbor_feature_cols = [
        c for c in neighbor_features.columns
        if c not in ["row_id", "date", "ticker"]
    ]

    for col in neighbor_feature_cols:
        neighbor_features[col] = pd.to_numeric(neighbor_features[col], errors="coerce")

    # Fill early-history NaNs with expanding/global medians.
    for col in neighbor_feature_cols:
        median_val = neighbor_features[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        neighbor_features[col] = neighbor_features[col].fillna(median_val)

    neighbor_features = neighbor_features.sort_values(["date", "ticker"]).reset_index(drop=True)

    neighbor_features.to_parquet(OUTPUT_PATH, index=False)

    lines = []
    lines.append("Week 20 Stock Latent Neighbor Features Summary")
    lines.append("==============================================")
    lines.append("")
    lines.append("Goal:")
    lines.append("For each stock/month, find similar historical stock latent states and summarize their future outcomes.")
    lines.append("")
    lines.append(f"Input path: {LATENT_METADATA_PATH}")
    lines.append(f"Output path: {OUTPUT_PATH}")
    lines.append(f"Input shape: {df.shape}")
    lines.append(f"Output shape: {neighbor_features.shape}")
    lines.append(f"Latent columns used: {len(latent_cols)}")
    lines.append(f"K neighbors: {K_NEIGHBORS}")
    lines.append(f"Minimum history rows before neighbor search: {MIN_HISTORY_ROWS}")
    lines.append("")
    lines.append("Feature columns:")
    lines.append(", ".join(neighbor_feature_cols))
    lines.append("")
    lines.append("Neighbor feature summary:")
    lines.append(neighbor_features[neighbor_feature_cols].describe().to_string())

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("")
    print("Saved:", OUTPUT_PATH)
    print("Saved:", REPORT_PATH)
    print("")
    print("OUTPUT SHAPE")
    print(neighbor_features.shape)
    print("")
    print("NEIGHBOR FEATURE SUMMARY")
    print(neighbor_features[neighbor_feature_cols].describe().to_string())


if __name__ == "__main__":
    main()