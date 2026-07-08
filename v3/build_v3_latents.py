"""Latent-dimension sweep: stock-state PCA + point-in-time neighbor features.

For each n_components in LATENT_DIMS:
1. Standardize the v3 state features (all numeric non-leakage features,
   now including the new v3_/v3m_ columns) and fit PCA on the full panel.
   NOTE: like v1, the PCA rotation is fit once on all months — a structural
   lookahead documented in the README. The neighbor *outcomes* below are
   strictly point-in-time.
2. For every month, find each stock's 50 nearest neighbors among all EARLIER
   stock-months with realized next-month returns, and summarize how those
   analogs performed (same statistics as v1).

Output per dim: data/processed/v3_neighbors_dim{K}.parquet
    (date, ticker, neighbor_* x 9)
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(PROJECT_ROOT, "data", "processed")
DATASET_PATH = os.path.join(DATA, "v3_modeling_dataset.parquet")

LATENT_DIMS = [4, 8, 16, 32]
N_NEIGHBORS = 50
MIN_HISTORY_ROWS = 250

NON_FEATURE_COLS = {"date", "ticker", "company", "row_id", "sector", "industry",
                    "price", "spy_price"}


def is_leakage_column(col: str) -> bool:
    name = col.lower()
    return "future" in name or "target" in name or "ranking_label" in name


def select_state_features(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in NON_FEATURE_COLS or is_leakage_column(col) or col.startswith("neighbor_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def summarize(neighbor_rows: pd.DataFrame, distances: np.ndarray) -> dict:
    return {
        "neighbor_count": int(len(neighbor_rows)),
        "neighbor_distance_mean": float(np.mean(distances)),
        "neighbor_distance_median": float(np.median(distances)),
        "neighbor_distance_min": float(np.min(distances)),
        "neighbor_avg_future_1m_return": float(neighbor_rows["future_1m_return"].mean()),
        "neighbor_median_future_1m_return": float(neighbor_rows["future_1m_return"].median()),
        "neighbor_avg_future_1m_excess_return": float(neighbor_rows["future_1m_excess_return"].mean()),
        "neighbor_outperform_spy_1m_rate": float((neighbor_rows["future_1m_excess_return"] > 0).mean()),
        "neighbor_positive_1m_return_rate": float((neighbor_rows["future_1m_return"] > 0).mean()),
    }


EMPTY = {
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


def build_neighbors_for_dim(
    meta: pd.DataFrame, latents: np.ndarray, dim: int
) -> pd.DataFrame:
    """meta: date/ticker/future returns aligned row-wise with `latents`."""
    dates = np.array(sorted(meta["date"].unique()))
    date_values = meta["date"].to_numpy()
    has_outcome = meta["future_1m_return"].notna().to_numpy()

    rows = []
    for current_date in dates:
        current_mask = date_values == current_date
        history_mask = (date_values < current_date) & has_outcome

        current_meta = meta[current_mask]

        if history_mask.sum() < MIN_HISTORY_ROWS:
            for _, r in current_meta.iterrows():
                rows.append({"date": r["date"], "ticker": r["ticker"], **EMPTY})
            continue

        nn = NearestNeighbors(n_neighbors=min(N_NEIGHBORS, int(history_mask.sum())))
        nn.fit(latents[history_mask])
        distances, indices = nn.kneighbors(latents[current_mask])

        history_meta = meta[history_mask].reset_index(drop=True)
        for pos, (_, r) in enumerate(current_meta.iterrows()):
            neighbor_rows = history_meta.iloc[indices[pos]]
            rows.append({
                "date": r["date"], "ticker": r["ticker"],
                **summarize(neighbor_rows, distances[pos]),
            })

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out


def main() -> None:
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    state_features = select_state_features(df)
    print(f"State features for PCA: {len(state_features)}", flush=True)

    X = df[state_features].copy()
    for col in state_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    X_scaled = StandardScaler().fit_transform(X)

    meta = df[["date", "ticker", "future_1m_return", "future_1m_excess_return"]].copy()

    for dim in LATENT_DIMS:
        started = time.time()
        pca = PCA(n_components=min(dim, X_scaled.shape[1]), random_state=42)
        latents = pca.fit_transform(X_scaled)
        explained = float(pca.explained_variance_ratio_.sum())

        neighbors = build_neighbors_for_dim(meta, latents, dim)
        out_path = os.path.join(DATA, f"v3_neighbors_dim{dim}.parquet")
        neighbors.to_parquet(out_path)

        print(f"dim={dim}: explained variance {explained:.1%}, "
              f"saved {out_path} {neighbors.shape} "
              f"({time.time() - started:.0f}s)", flush=True)

    print("LATENT SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
