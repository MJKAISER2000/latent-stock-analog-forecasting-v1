"""v3 walk-forward CV — retrain monthly, save FULL rankings per month.

Unlike the v1 backtest (which saved only the held portfolio), this saves every
stock's score for every out-of-sample month. Portfolio size, branch blending,
and hedging then become cheap post-processing (run_v3_grid.py) instead of
requiring a retrain per config.

Branches (each an independent walk-forward run):
    original_v1feats   v1 feature set only (baseline for "did new data help?")
    original_v3feats   v1 features + all new v3_/v3m_ features
    neighbor_dim4/8/16/32   the 9 latent-neighbor features per PCA dimension

Training per month is identical to v1/live: train on all months strictly
before t (80/20 time-ordered validation split, early stopping), score month t.
Feature imputation medians are computed on data <= t only.

Checkpointed per (branch, month) to outputs/v3_backtest/rankings_{branch}.csv.

Usage:
    python v3/run_v3_cv.py --branches original_v1feats original_v3feats
    python v3/run_v3_cv.py --branches neighbor_dim4 neighbor_dim8 ...
    python v3/run_v3_cv.py            (all branches)
"""

import argparse
import os
import sys
import time

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_config
from src.models.rankers import train_predict_latest
from src.features.feature_sets import is_leakage_like
from backtests.run_cv_backtest import point_in_time_impute, get_test_dates

DATA = os.path.join(PROJECT_ROOT, "data", "processed")
DATASET_PATH = os.path.join(DATA, "v3_modeling_dataset.parquet")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "v3_backtest")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "live_model_config.yaml")

LATENT_DIMS = [4, 8, 16, 32]
MIN_TRAIN_MONTHS = 36

NON_FEATURES = {"date", "ticker", "company", "row_id", "sector", "industry"}


def candidate_features(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in NON_FEATURES
        and not is_leakage_like(c)
        and not c.startswith("neighbor_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def branch_dataset_and_features(
    branch: str, dataset: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    if branch == "original_v1feats":
        feats = [c for c in candidate_features(dataset)
                 if not c.startswith(("v3_", "v3m_"))]
        return dataset, feats

    if branch == "original_v3feats":
        return dataset, candidate_features(dataset)

    if branch.startswith("neighbor_dim"):
        dim = int(branch.replace("neighbor_dim", ""))
        neighbors = pd.read_parquet(os.path.join(DATA, f"v3_neighbors_dim{dim}.parquet"))
        neighbors["ticker"] = neighbors["ticker"].astype(str).str.strip().str.upper()
        keep = ["date", "ticker", "future_1m_return", "future_1m_spy_return",
                "future_1m_excess_return"]
        merged = dataset[keep].merge(neighbors, on=["date", "ticker"], how="left")
        feats = [c for c in merged.columns if c.startswith("neighbor_")]
        return merged, feats

    raise ValueError(f"Unknown branch: {branch}")


def run_branch(branch: str, dataset: pd.DataFrame, config: dict) -> None:
    out_path = os.path.join(OUT_DIR, f"rankings_{branch}.csv")

    branch_df, feature_cols = branch_dataset_and_features(branch, dataset)
    test_dates = get_test_dates(branch_df, MIN_TRAIN_MONTHS)

    done: set[pd.Timestamp] = set()
    if os.path.exists(out_path):
        done = set(pd.to_datetime(
            pd.read_csv(out_path, usecols=["date"], parse_dates=["date"])["date"].unique()
        ))

    pending = [d for d in test_dates if d not in done]
    print(f"[{branch}] features={len(feature_cols)} months={len(test_dates)} "
          f"pending={len(pending)}", flush=True)

    for i, signal_date in enumerate(pending, start=1):
        started = time.time()
        visible = branch_df[branch_df["date"] <= signal_date].copy()
        visible = point_in_time_impute(visible, feature_cols)

        _, predictions = train_predict_latest(
            df=visible,
            feature_cols=feature_cols,
            config=config,
            signal_date=signal_date,
        )

        out = predictions[["date", "ticker", "ranker_score", "rank_by_date"]]
        out.to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)

        if i % 10 == 0 or i == len(pending):
            print(f"[{branch}] {i}/{len(pending)} {signal_date.date()} "
                  f"({time.time() - started:.0f}s/mo)", flush=True)

    print(f"[{branch}] DONE -> {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    all_branches = (["original_v1feats", "original_v3feats"]
                    + [f"neighbor_dim{d}" for d in LATENT_DIMS])
    parser.add_argument("--branches", nargs="*", default=all_branches)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    os.makedirs(OUT_DIR, exist_ok=True)
    config = load_config(CONFIG_PATH)

    dataset = pd.read_parquet(DATASET_PATH)
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset["ticker"] = dataset["ticker"].astype(str).str.strip().str.upper()
    dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)

    for branch in args.branches:
        run_branch(branch, dataset, config)

    print("ALL BRANCHES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
