import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    non_feature_cols = [
        "date",
        "ticker",
        "long_name",
        "future_12m_return",
        "future_12m_spy_return",
        "target_abs_direction",
        "target_outperform_spy",
    ]

    return [col for col in df.columns if col not in non_feature_cols]


def time_based_split(df: pd.DataFrame):
    train = df[df["date"] < "2018-01-01"].copy()
    val = df[(df["date"] >= "2018-01-01") & (df["date"] < "2021-01-01")].copy()
    test = df[df["date"] >= "2021-01-01"].copy()
    return train, val, test


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 20):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(X_train, X_val, latent_dim: int, device: str):
    input_dim = X_train.shape[1]

    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=128,
        shuffle=True,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_state = None
    patience = 25
    patience_counter = 0

    for epoch in range(1, 301):
        model.train()
        train_losses = []

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            x_hat = model(batch_x)
            loss = criterion(x_hat, batch_x)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_x = val_tensor.to(device)
            val_hat = model(val_x)
            val_loss = criterion(val_hat, val_x).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 25 == 0:
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


def evaluate_classifier(clf, Z, y, split_name: str) -> dict:
    pred = clf.predict(Z)
    prob = clf.predict_proba(Z)[:, 1]

    return {
        "split": split_name,
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, prob),
    }


def top_ranked_portfolio_signal(test_df: pd.DataFrame, probabilities: np.ndarray, top_n: int) -> float:
    temp = test_df.copy()
    temp["predicted_prob_outperform"] = probabilities

    returns = []

    for date, group in temp.groupby("date"):
        top = group.sort_values("predicted_prob_outperform", ascending=False).head(top_n)
        returns.append(top["future_12m_return"].mean())

    return float(np.mean(returns))


def main():
    set_seed(42)

    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    input_path = "data/processed/expanded_modeling_dataset_with_metadata.parquet"

    print("Loading expanded metadata-enhanced dataset...")
    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    feature_cols = get_feature_columns(df)

    print("Dataset shape:", df.shape)
    print("Feature count:", len(feature_cols))
    print("Ticker count:", df["ticker"].nunique())
    print("Date range:", df["date"].min(), "to", df["date"].max())

    train, val, test = time_based_split(df)

    print("")
    print("Split sizes:")
    print("Train:", train.shape)
    print("Validation:", val.shape)
    print("Test:", test.shape)

    X_train_raw = train[feature_cols].values
    X_val_raw = val[feature_cols].values
    X_test_raw = test[feature_cols].values

    y_train = train["target_outperform_spy"].values
    y_val = val["target_outperform_spy"].values
    y_test = test["target_outperform_spy"].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    latent_dim = 20

    print("")
    print("Training expanded autoencoder...")
    ae, val_recon_loss = train_autoencoder(
        X_train=X_train,
        X_val=X_val,
        latent_dim=latent_dim,
        device=device,
    )

    print("Validation reconstruction loss:", val_recon_loss)

    Z_train = encode_dataset(ae, X_train, device)
    Z_val = encode_dataset(ae, X_val, device)
    Z_test = encode_dataset(ae, X_test, device)

    print("Training gradient boosting classifier on latent features...")

    clf = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=3,
        random_state=42,
    )

    clf.fit(Z_train, y_train)

    metrics = pd.DataFrame(
        [
            evaluate_classifier(clf, Z_train, y_train, "train"),
            evaluate_classifier(clf, Z_val, y_val, "validation"),
            evaluate_classifier(clf, Z_test, y_test, "test"),
        ]
    )

    metrics["model"] = "expanded_autoencoder_latent20_gradient_boosting"
    metrics["latent_dim"] = latent_dim
    metrics["val_reconstruction_loss"] = val_recon_loss

    test_probs = clf.predict_proba(Z_test)[:, 1]

    portfolio_signal = pd.DataFrame(
        [
            {
                "model": "expanded_autoencoder_latent20_gradient_boosting",
                "top5_avg_future_12m_return": top_ranked_portfolio_signal(test, test_probs, top_n=5),
                "top10_avg_future_12m_return": top_ranked_portfolio_signal(test, test_probs, top_n=10),
                "top25_avg_future_12m_return": top_ranked_portfolio_signal(test, test_probs, top_n=25),
                "top50_avg_future_12m_return": top_ranked_portfolio_signal(test, test_probs, top_n=50),
                "all_stock_avg_future_12m_return": test["future_12m_return"].mean(),
                "spy_avg_future_12m_return": test["future_12m_spy_return"].mean(),
                "latent_dim": latent_dim,
                "val_reconstruction_loss": val_recon_loss,
            }
        ]
    )

    prediction_output = test[
        [
            "date",
            "ticker",
            "future_12m_return",
            "future_12m_spy_return",
            "target_outperform_spy",
            "target_abs_direction",
        ]
    ].copy()

    prediction_output["predicted_prob_outperform"] = test_probs
    prediction_output["rank_by_date"] = prediction_output.groupby("date")[
        "predicted_prob_outperform"
    ].rank(ascending=False, method="first")

    metrics_path = "outputs/tables/week11_expanded_autoencoder_metrics.csv"
    signal_path = "outputs/tables/week11_expanded_autoencoder_portfolio_signal.csv"
    predictions_path = "outputs/tables/week11_expanded_predictions_latent20_gb.csv"

    metrics.to_csv(metrics_path, index=False)
    portfolio_signal.to_csv(signal_path, index=False)
    prediction_output.to_csv(predictions_path, index=False)

    report_path = "outputs/reports/week11_expanded_autoencoder_prediction_summary.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 11 Expanded Autoencoder Prediction Summary\n")
        f.write("==============================================\n\n")
        f.write("Model:\n")
        f.write("expanded metadata-enhanced features -> autoencoder latent_dim=20 -> gradient boosting\n\n")
        f.write(f"Input dataset: {input_path}\n")
        f.write(f"Dataset shape: {df.shape}\n")
        f.write(f"Feature count: {len(feature_cols)}\n")
        f.write(f"Ticker count: {df['ticker'].nunique()}\n")
        f.write(f"Validation reconstruction loss: {val_recon_loss}\n\n")
        f.write("Classification metrics:\n")
        f.write(metrics.to_string(index=False))
        f.write("\n\nPortfolio signal using 12-month target:\n")
        f.write(portfolio_signal.to_string(index=False))
        f.write("\n\nOutput predictions:\n")
        f.write(predictions_path)
        f.write("\n")

    print("")
    print("Saved:", metrics_path)
    print("Saved:", signal_path)
    print("Saved:", predictions_path)
    print("Saved:", report_path)
    print("")
    print("Classification metrics:")
    print(metrics)
    print("")
    print("Portfolio signal:")
    print(portfolio_signal)


if __name__ == "__main__":
    main()