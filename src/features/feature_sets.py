from typing import Any

import pandas as pd


BASE_NON_FEATURE_COLS = [
    "date",
    "ticker",
    "company",
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
]


def is_leakage_like(col: str) -> bool:
    name = col.lower()

    if "future" in name:
        return True

    if "target" in name:
        return True

    if "ranking_label" in name:
        return True

    return False


def get_all_candidate_features(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in BASE_NON_FEATURE_COLS
        and not is_leakage_like(c)
    ]


def get_original_feature_columns(df: pd.DataFrame) -> list[str]:
    features = get_all_candidate_features(df)

    return [
        c for c in features
        if not c.startswith("neighbor_")
    ]


def get_neighbor_feature_columns(df: pd.DataFrame) -> list[str]:
    features = get_all_candidate_features(df)

    return [
        c for c in features
        if c.startswith("neighbor_")
    ]


def get_feature_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set == "original_only":
        return get_original_feature_columns(df)

    if feature_set == "neighbor_only":
        return get_neighbor_feature_columns(df)

    if feature_set == "original_plus_neighbors":
        original = get_original_feature_columns(df)
        neighbor = get_neighbor_feature_columns(df)
        return original + neighbor

    raise ValueError(f"Unknown feature_set: {feature_set}")


def validate_no_leakage(feature_cols: list[str]) -> None:
    leaked = [c for c in feature_cols if is_leakage_like(c)]

    if leaked:
        raise ValueError(f"Leakage-like columns found in feature set: {leaked}")


def prepare_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()

    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0.0)

    return X


def print_feature_set_summary(df: pd.DataFrame, feature_set: str) -> list[str]:
    feature_cols = get_feature_columns(df, feature_set)
    validate_no_leakage(feature_cols)

    neighbor_cols = [c for c in feature_cols if c.startswith("neighbor_")]
    original_cols = [c for c in feature_cols if not c.startswith("neighbor_")]

    print("")
    print("=" * 80)
    print(f"FEATURE SET: {feature_set}")
    print("=" * 80)
    print("Total features:", len(feature_cols))
    print("Original features:", len(original_cols))
    print("Neighbor features:", len(neighbor_cols))

    if neighbor_cols:
        print("Neighbor feature columns:")
        print(neighbor_cols)

    return feature_cols