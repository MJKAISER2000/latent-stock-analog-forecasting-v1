import os
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    non_feature_cols = [
        "date",
        "ticker",
        "future_12m_return",
        "future_12m_spy_return",
        "target_abs_direction",
        "target_outperform_spy",
    ]
    return [col for col in df.columns if col not in non_feature_cols]


def build_sequences_for_ticker(
    ticker_df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int = 12,
):
    """
    Builds rolling sequences for one ticker.

    Example:
    If sequence_length = 12, then each sample uses the past 12 months of features
    to predict the target at the final month of the sequence.
    """

    ticker_df = ticker_df.sort_values("date").reset_index(drop=True)

    X_sequences = []
    y_outperform = []
    y_abs_direction = []
    future_returns = []
    future_spy_returns = []
    dates = []
    tickers = []

    features = ticker_df[feature_cols].values

    for end_idx in range(sequence_length - 1, len(ticker_df)):
        start_idx = end_idx - sequence_length + 1

        sequence = features[start_idx : end_idx + 1]

        row = ticker_df.iloc[end_idx]

        X_sequences.append(sequence)
        y_outperform.append(row["target_outperform_spy"])
        y_abs_direction.append(row["target_abs_direction"])
        future_returns.append(row["future_12m_return"])
        future_spy_returns.append(row["future_12m_spy_return"])
        dates.append(row["date"])
        tickers.append(row["ticker"])

    return (
        X_sequences,
        y_outperform,
        y_abs_direction,
        future_returns,
        future_spy_returns,
        dates,
        tickers,
    )


def build_sequence_dataset(sequence_length: int = 12):
    config = load_config("configs/experiment_01.yaml")
    processed_dir = config["processed_data_dir"]

    input_path = os.path.join(processed_dir, "modeling_dataset.parquet")
    output_path = os.path.join(processed_dir, f"sequence_dataset_{sequence_length}m.npz")
    meta_output_path = os.path.join(processed_dir, f"sequence_metadata_{sequence_length}m.csv")

    print("Loading modeling dataset...")
    df = pd.read_parquet(input_path)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    feature_cols = get_feature_columns(df)

    print("Feature count:", len(feature_cols))
    print("Sequence length:", sequence_length)

    # Scale features using only pre-2018 training data to avoid future leakage.
    train_mask = df["date"] < "2018-01-01"

    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, feature_cols])

    df[feature_cols] = scaler.transform(df[feature_cols])

    all_X = []
    all_y_outperform = []
    all_y_abs = []
    all_future_returns = []
    all_future_spy_returns = []
    all_dates = []
    all_tickers = []

    for ticker, ticker_df in df.groupby("ticker"):
        (
            X_sequences,
            y_outperform,
            y_abs_direction,
            future_returns,
            future_spy_returns,
            dates,
            tickers,
        ) = build_sequences_for_ticker(
            ticker_df=ticker_df,
            feature_cols=feature_cols,
            sequence_length=sequence_length,
        )

        all_X.extend(X_sequences)
        all_y_outperform.extend(y_outperform)
        all_y_abs.extend(y_abs_direction)
        all_future_returns.extend(future_returns)
        all_future_spy_returns.extend(future_spy_returns)
        all_dates.extend(dates)
        all_tickers.extend(tickers)

    X = np.array(all_X, dtype=np.float32)
    y_outperform = np.array(all_y_outperform, dtype=np.float32)
    y_abs = np.array(all_y_abs, dtype=np.float32)
    future_returns = np.array(all_future_returns, dtype=np.float32)
    future_spy_returns = np.array(all_future_spy_returns, dtype=np.float32)

    meta = pd.DataFrame(
        {
            "date": all_dates,
            "ticker": all_tickers,
            "future_12m_return": future_returns,
            "future_12m_spy_return": future_spy_returns,
            "target_outperform_spy": y_outperform,
            "target_abs_direction": y_abs,
        }
    )

    meta = meta.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Reorder X and labels to match sorted metadata
    sort_idx = np.lexsort((np.array(all_tickers), pd.to_datetime(all_dates).astype("int64")))
    X = X[sort_idx]
    y_outperform = y_outperform[sort_idx]
    y_abs = y_abs[sort_idx]
    future_returns = future_returns[sort_idx]
    future_spy_returns = future_spy_returns[sort_idx]

    np.savez_compressed(
        output_path,
        X=X,
        y_outperform=y_outperform,
        y_abs_direction=y_abs,
        future_12m_return=future_returns,
        future_12m_spy_return=future_spy_returns,
        feature_cols=np.array(feature_cols),
    )

    meta.to_csv(meta_output_path, index=False)

    print("Saved sequence dataset to:", output_path)
    print("Saved sequence metadata to:", meta_output_path)
    print("X shape:", X.shape)
    print("y_outperform shape:", y_outperform.shape)
    print("Date range:", meta["date"].min(), "to", meta["date"].max())
    print("Ticker count:", meta["ticker"].nunique())
    print("Target balance:")
    print(meta["target_outperform_spy"].value_counts(normalize=True))


if __name__ == "__main__":
    build_sequence_dataset(sequence_length=12)