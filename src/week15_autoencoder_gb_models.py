import os
import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


DATASETS = ["full500", "highvol100", "highvol200"]
HORIZONS = [1, 36]
LATENT_DIM = 24


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    non_feature_cols = [
        "date",
        "ticker",
        "company",

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
    ]

    return [c for c in df.columns if c not in non_feature_cols]


def time_based_split(df: pd.DataFrame):
    train = df[df["date"] < "2018-01-01"].copy()
    val = df[(df["date"] >= "2018-01-01") & (df["date"] < "2021-01-01")].copy()
    test = df[df["date"] >= "2021-01-01"].copy()
    return train, val, test


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(X_train, X_val, latent_dim: int, device: str):
    input_dim = X_train.shape[1]

    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=256,
        shuffle=True,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None
    patience = 30
    patience_counter = 0

    for epoch in range(1, 401):
        model.train()
        train_losses = []

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_x = val_tensor.to(device)
            val_reconstructed = model(val_x)
            val_loss = criterion(val_reconstructed, val_x).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 50 == 0:
            print(
                f"epoch={epoch}, "
                f"train_loss={np.mean(train_losses):.5f}, "
                f"val_loss={val_loss:.5f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    return model, best_val_loss


def encode_dataset(model, X, device: str) -> np.ndarray:
    model.eval()

    x_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        z = model.encode(x_tensor).cpu().numpy()

    return z


def evaluate_classifier(model, X, y, dataset_name: str, horizon: int, split_name: str) -> dict:
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    try:
        auc = roc_auc_score(y, prob)
    except ValueError:
        auc = np.nan

    return {
        "dataset": dataset_name,
        "horizon_months": horizon,
        "split": split_name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": auc,
    }


def top_ranked_signal(test_df: pd.DataFrame, probabilities: np.ndarray, horizon: int, top_n: int) -> float:
    temp = test_df.copy()
    temp["predicted_prob_outperform"] = probabilities

    return_col = f"future_{horizon}m_return"

    returns = []

    for date, group in temp.groupby("date"):
        top = group.sort_values("predicted_prob_outperform", ascending=False).head(top_n)
        returns.append(top[return_col].mean())

    return float(np.mean(returns))


def train_one(dataset_name: str, horizon: int, device: str):
    input_path = f"data/processed/week15_{dataset_name}_modeling_dataset.parquet"

    print("")
    print("=" * 90)
    print(f"Training AE + GB | dataset={dataset_name} | horizon={horizon}m")

    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    target_col = f"target_outperform_spy_{horizon}m"
    return_col = f"future_{horizon}m_return"
    spy_return_col = f"future_{horizon}m_spy_return"

    df = df.dropna(subset=[target_col, return_col, spy_return_col]).copy()

    feature_cols = get_feature_columns(df)

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    print("Dataset shape after target drop:", df.shape)
    print("Feature count:", len(feature_cols))
    print("Ticker count:", df["ticker"].nunique())
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Target balance:")
    print(df[target_col].value_counts(normalize=True))

    train, val, test = time_based_split(df)

    print("Split sizes:")
    print("Train:", train.shape)
    print("Validation:", val.shape)
    print("Test:", test.shape)

    X_train_raw = train[feature_cols].values
    X_val_raw = val[feature_cols].values
    X_test_raw = test[feature_cols].values

    y_train = train[target_col].astype(int).values
    y_val = val[target_col].astype(int).values
    y_test = test[target_col].astype(int).values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    ae, val_recon_loss = train_autoencoder(
        X_train=X_train,
        X_val=X_val,
        latent_dim=LATENT_DIM,
        device=device,
    )

    Z_train = encode_dataset(ae, X_train, device)
    Z_val = encode_dataset(ae, X_val, device)
    Z_test = encode_dataset(ae, X_test, device)

    clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        random_state=42,
    )

    clf.fit(Z_train, y_train)

    metrics = []

    for split_name, Z, y in [
        ("train", Z_train, y_train),
        ("validation", Z_val, y_val),
        ("test", Z_test, y_test),
    ]:
        result = evaluate_classifier(clf, Z, y, dataset_name, horizon, split_name)
        result["latent_dim"] = LATENT_DIM
        result["val_reconstruction_loss"] = val_recon_loss
        metrics.append(result)

        print(
            f"{split_name}: "
            f"accuracy={result['accuracy']:.3f}, "
            f"f1={result['f1']:.3f}, "
            f"auc={result['auc']:.3f}"
        )

    test_probs = clf.predict_proba(Z_test)[:, 1]

    signal = {
        "dataset": dataset_name,
        "horizon_months": horizon,
        "model": "autoencoder_latent_gb",
        "latent_dim": LATENT_DIM,
        "val_reconstruction_loss": val_recon_loss,
        "top5_avg_future_return": top_ranked_signal(test, test_probs, horizon, 5),
        "top10_avg_future_return": top_ranked_signal(test, test_probs, horizon, 10),
        "top25_avg_future_return": top_ranked_signal(test, test_probs, horizon, 25),
        "top50_avg_future_return": top_ranked_signal(test, test_probs, horizon, 50),
        "all_stock_avg_future_return": test[return_col].mean(),
        "spy_avg_future_return": test[spy_return_col].mean(),
    }

    prediction_output = test[
        [
            "date",
            "ticker",
            return_col,
            spy_return_col,
            f"future_{horizon}m_excess_return",
            target_col,
        ]
    ].copy()

    prediction_output["dataset"] = dataset_name
    prediction_output["horizon_months"] = horizon
    prediction_output["model"] = "autoencoder_latent_gb"
    prediction_output["predicted_prob_outperform"] = test_probs
    prediction_output["rank_by_date"] = prediction_output.groupby("date")[
        "predicted_prob_outperform"
    ].rank(ascending=False, method="first")

    pred_path = f"outputs/tables/week15_ae_gb_predictions_{dataset_name}_{horizon}m.csv"
    prediction_output.to_csv(pred_path, index=False)

    print("Saved predictions:", pred_path)
    print("Signal:", signal)

    return pd.DataFrame(metrics), pd.DataFrame([signal])


def main():
    set_seed(42)

    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    all_metrics = []
    all_signals = []

    for dataset_name in DATASETS:
        for horizon in HORIZONS:
            metrics, signal = train_one(dataset_name, horizon, device)
            all_metrics.append(metrics)
            all_signals.append(signal)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    signals_df = pd.concat(all_signals, ignore_index=True)

    metrics_path = "outputs/tables/week15_ae_gb_model_metrics.csv"
    signals_path = "outputs/tables/week15_ae_gb_portfolio_signal.csv"
    report_path = "outputs/reports/week15_ae_gb_model_summary.txt"

    metrics_df.to_csv(metrics_path, index=False)
    signals_df.to_csv(signals_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 15 Autoencoder + Gradient Boosting Model Summary\n")
        f.write("====================================================\n\n")
        f.write("Goal:\n")
        f.write(
            "Train autoencoder latent-feature gradient boosting models on full500, highvol100, "
            "and highvol200 universes for 1-month and 36-month horizons.\n\n"
        )
        f.write("Latent dimension:\n")
        f.write(str(LATENT_DIM))
        f.write("\n\nMetrics:\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\nPortfolio signal:\n")
        f.write(signals_df.to_string(index=False))
        f.write("\n")

    print("")
    print("Saved:", metrics_path)
    print("Saved:", signals_path)
    print("Saved:", report_path)
    print("")
    print("Portfolio signals:")
    print(signals_df)


if __name__ == "__main__":
    main()