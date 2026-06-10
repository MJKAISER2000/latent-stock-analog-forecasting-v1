import os
import yaml
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def time_based_split(df: pd.DataFrame):
    train = df[df["date"] < "2018-01-01"].copy()
    val = df[(df["date"] >= "2018-01-01") & (df["date"] < "2021-01-01")].copy()
    test = df[df["date"] >= "2021-01-01"].copy()
    return train, val, test


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
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
            nn.Linear(128, input_dim),
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
        batch_size=64,
        shuffle=True,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_state = None
    patience = 20
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


def add_sector_column(df: pd.DataFrame) -> pd.DataFrame:
    sector_map = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "NVDA": "Technology",
        "GOOGL": "Communication",
        "AMZN": "Consumer Disc.",
        "META": "Communication",
        "JPM": "Financials",
        "XOM": "Energy",
        "UNH": "Healthcare",
        "WMT": "Consumer Staples",
        "PG": "Consumer Staples",
        "KO": "Consumer Staples",
        "HD": "Consumer Disc.",
        "COST": "Consumer Staples",
        "AVGO": "Technology",
        "LLY": "Healthcare",
        "TSLA": "Consumer Disc.",
        "V": "Financials",
        "MA": "Financials",
        "NFLX": "Communication",
    }

    df = df.copy()
    df["sector_label"] = df["ticker"].map(sector_map).fillna("Unknown")
    return df


def make_pca_dataframe(Z: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Z)

    out = meta.copy()
    out["pca_1"] = coords[:, 0]
    out["pca_2"] = coords[:, 1]

    print("PCA explained variance ratio:", pca.explained_variance_ratio_)

    return out


def make_tsne_dataframe(Z: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    perplexity = min(30, max(5, len(Z) // 20))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )

    coords = tsne.fit_transform(Z)

    out = meta.copy()
    out["tsne_1"] = coords[:, 0]
    out["tsne_2"] = coords[:, 1]

    return out


def scatter_by_category(df, x_col, y_col, category_col, title, output_path):
    plt.figure(figsize=(10, 7))

    categories = sorted(df[category_col].dropna().unique())

    for category in categories:
        subset = df[df[category_col] == category]
        plt.scatter(
            subset[x_col],
            subset[y_col],
            label=str(category),
            alpha=0.7,
            s=35,
        )

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved figure to {output_path}")


def scatter_by_numeric(df, x_col, y_col, value_col, title, output_path):
    plt.figure(figsize=(10, 7))

    points = plt.scatter(
        df[x_col],
        df[y_col],
        c=df[value_col],
        alpha=0.75,
        s=35,
    )

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.colorbar(points, label=value_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved figure to {output_path}")


def assign_return_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["future_return_bucket"] = pd.cut(
        df["future_12m_return"],
        bins=[-np.inf, -0.20, 0.0, 0.20, 0.50, np.inf],
        labels=[
            "big_loss",
            "small_loss",
            "modest_gain",
            "strong_gain",
            "huge_gain",
        ],
    )

    return df


def main():
    set_seed(42)

    config = load_config("configs/experiment_01.yaml")
    processed_dir = config["processed_data_dir"]

    df = pd.read_parquet(os.path.join(processed_dir, "modeling_dataset.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    feature_cols = get_feature_columns(df)

    train, val, test = time_based_split(df)

    X_train_raw = train[feature_cols].values
    X_val_raw = val[feature_cols].values
    X_test_raw = test[feature_cols].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    latent_dim = 20
    print(f"Training analysis autoencoder with latent_dim={latent_dim}")

    ae, val_loss = train_autoencoder(
        X_train=X_train,
        X_val=X_val,
        latent_dim=latent_dim,
        device=device,
    )

    print("Validation reconstruction loss:", val_loss)

    Z_test = encode_dataset(ae, X_test, device)

    meta_cols = [
        "date",
        "ticker",
        "future_12m_return",
        "future_12m_spy_return",
        "target_outperform_spy",
        "target_abs_direction",
    ]

    meta = test[meta_cols].copy()
    meta = add_sector_column(meta)
    meta = assign_return_bucket(meta)

    latent_cols = [f"z_{i}" for i in range(latent_dim)]
    latent_df = pd.DataFrame(Z_test, columns=latent_cols)
    latent_df = pd.concat([meta.reset_index(drop=True), latent_df], axis=1)

    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    latent_path = "outputs/tables/week5_latent_test_coordinates.csv"
    latent_df.to_csv(latent_path, index=False)
    print(f"Saved latent coordinates to {latent_path}")

    pca_df = make_pca_dataframe(Z_test, meta)
    tsne_df = make_tsne_dataframe(Z_test, meta)

    pca_path = "outputs/tables/week5_pca_coordinates.csv"
    tsne_path = "outputs/tables/week5_tsne_coordinates.csv"

    pca_df.to_csv(pca_path, index=False)
    tsne_df.to_csv(tsne_path, index=False)

    scatter_by_category(
        pca_df,
        "pca_1",
        "pca_2",
        "sector_label",
        "Latent Space PCA Colored by Sector",
        "outputs/figures/week5_pca_by_sector.png",
    )

    scatter_by_category(
        pca_df,
        "pca_1",
        "pca_2",
        "future_return_bucket",
        "Latent Space PCA Colored by Future 12M Return Bucket",
        "outputs/figures/week5_pca_by_future_return_bucket.png",
    )

    scatter_by_numeric(
        pca_df,
        "pca_1",
        "pca_2",
        "future_12m_return",
        "Latent Space PCA Colored by Future 12M Return",
        "outputs/figures/week5_pca_by_future_return_numeric.png",
    )

    scatter_by_category(
        tsne_df,
        "tsne_1",
        "tsne_2",
        "sector_label",
        "Latent Space t-SNE Colored by Sector",
        "outputs/figures/week5_tsne_by_sector.png",
    )

    scatter_by_category(
        tsne_df,
        "tsne_1",
        "tsne_2",
        "future_return_bucket",
        "Latent Space t-SNE Colored by Future 12M Return Bucket",
        "outputs/figures/week5_tsne_by_future_return_bucket.png",
    )

    scatter_by_numeric(
        tsne_df,
        "tsne_1",
        "tsne_2",
        "future_12m_return",
        "Latent Space t-SNE Colored by Future 12M Return",
        "outputs/figures/week5_tsne_by_future_return_numeric.png",
    )

    summary_path = "outputs/reports/week5_latent_space_summary.txt"

    with open(summary_path, "w") as f:
        f.write("Week 5 Latent Space Analysis Summary\n")
        f.write("====================================\n\n")
        f.write(f"Latent dimension: {latent_dim}\n")
        f.write(f"Validation reconstruction loss: {val_loss}\n")
        f.write(f"Test rows analyzed: {len(test)}\n\n")
        f.write("Average future return by sector:\n")
        f.write(
            meta.groupby("sector_label")["future_12m_return"]
            .mean()
            .sort_values(ascending=False)
            .to_string()
        )
        f.write("\n\n")
        f.write("Average future return by return bucket:\n")
        f.write(
            meta.groupby("future_return_bucket", observed=False)["future_12m_return"]
            .mean()
            .to_string()
        )
        f.write("\n\n")
        f.write("Outperformance rate by sector:\n")
        f.write(
            meta.groupby("sector_label")["target_outperform_spy"]
            .mean()
            .sort_values(ascending=False)
            .to_string()
        )
        f.write("\n")

    print(f"Saved summary report to {summary_path}")


if __name__ == "__main__":
    main()