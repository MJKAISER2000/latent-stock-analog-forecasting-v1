import os
import yaml
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


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
                f"latent_dim={latent_dim}, epoch={epoch}, "
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


def evaluate_latent_features(Z_train, y_train, Z_val, y_val, Z_test, y_test, model_name: str):
    if model_name == "logistic_regression":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif model_name == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    clf.fit(Z_train, y_train)

    rows = []

    for split_name, Z, y in [
        ("train", Z_train, y_train),
        ("validation", Z_val, y_val),
        ("test", Z_test, y_test),
    ]:
        pred = clf.predict(Z)
        prob = clf.predict_proba(Z)[:, 1]

        rows.append({
            "classifier": model_name,
            "split": split_name,
            "accuracy": accuracy_score(y, pred),
            "f1": f1_score(y, pred, zero_division=0),
            "auc": roc_auc_score(y, prob),
        })

    return pd.DataFrame(rows), clf


def top_ranked_portfolio_return(test_df: pd.DataFrame, probabilities: np.ndarray, top_n: int = 5) -> float:
    temp = test_df.copy()
    temp["predicted_prob"] = probabilities

    monthly_returns = []

    for date, group in temp.groupby("date"):
        top = group.sort_values("predicted_prob", ascending=False).head(top_n)
        monthly_returns.append(top["future_12m_return"].mean())

    return float(np.mean(monthly_returns))


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

    y_train = train["target_outperform_spy"].values
    y_val = val["target_outperform_spy"].values
    y_test = test["target_outperform_spy"].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Input feature dimension:", X_train.shape[1])

    latent_dims = [2, 5, 10, 20]
    all_results = []
    portfolio_results = []

    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    for latent_dim in latent_dims:
        print("")
        print("=" * 70)
        print(f"Training autoencoder with latent_dim={latent_dim}")

        ae, val_recon_loss = train_autoencoder(
            X_train=X_train,
            X_val=X_val,
            latent_dim=latent_dim,
            device=device,
        )

        Z_train = encode_dataset(ae, X_train, device)
        Z_val = encode_dataset(ae, X_val, device)
        Z_test = encode_dataset(ae, X_test, device)

        for classifier_name in ["logistic_regression", "gradient_boosting"]:
            metrics_df, clf = evaluate_latent_features(
                Z_train,
                y_train,
                Z_val,
                y_val,
                Z_test,
                y_test,
                classifier_name,
            )

            metrics_df["latent_dim"] = latent_dim
            metrics_df["val_reconstruction_loss"] = val_recon_loss
            all_results.append(metrics_df)

            test_probs = clf.predict_proba(Z_test)[:, 1]

            top5_return = top_ranked_portfolio_return(test, test_probs, top_n=5)
            top10_return = top_ranked_portfolio_return(test, test_probs, top_n=10)

            portfolio_results.append({
                "latent_dim": latent_dim,
                "classifier": classifier_name,
                "top5_avg_future_12m_return": top5_return,
                "top10_avg_future_12m_return": top10_return,
                "test_spy_avg_future_12m_return": test["future_12m_spy_return"].mean(),
                "val_reconstruction_loss": val_recon_loss,
            })

            prediction_output = test[
                [
                    "date",
                    "ticker",
                    "future_12m_return",
                    "future_12m_spy_return",
                    "target_outperform_spy",
                ]
            ].copy()

            prediction_output["latent_dim"] = latent_dim
            prediction_output["classifier"] = classifier_name
            prediction_output["predicted_prob_outperform"] = test_probs
            prediction_output["rank_by_date"] = prediction_output.groupby("date")[
                "predicted_prob_outperform"
            ].rank(ascending=False, method="first")

            pred_path = (
                f"outputs/tables/week4_predictions_latent{latent_dim}_{classifier_name}.csv"
            )
            prediction_output.to_csv(pred_path, index=False)

            print(
                f"latent_dim={latent_dim}, classifier={classifier_name}, "
                f"top5={top5_return:.3f}, top10={top10_return:.3f}"
            )

    results = pd.concat(all_results, ignore_index=True)
    portfolio = pd.DataFrame(portfolio_results)

    results_path = "outputs/tables/week4_autoencoder_metrics.csv"
    portfolio_path = "outputs/tables/week4_autoencoder_portfolio_signal.csv"

    results.to_csv(results_path, index=False)
    portfolio.to_csv(portfolio_path, index=False)

    report_path = "outputs/reports/week4_autoencoder_summary.txt"
    with open(report_path, "w") as f:
        f.write("Week 4 Autoencoder Latent Market Model Summary\n")
        f.write("=============================================\n\n")
        f.write(f"Input feature dimension: {X_train.shape[1]}\n")
        f.write(f"Device: {device}\n\n")
        f.write("Classification metrics:\n")
        f.write(results.to_string(index=False))
        f.write("\n\nPortfolio signal:\n")
        f.write(portfolio.to_string(index=False))
        f.write("\n")

    print("")
    print("Saved metrics to:", results_path)
    print("Saved portfolio signal to:", portfolio_path)
    print("Saved report to:", report_path)


if __name__ == "__main__":
    main()