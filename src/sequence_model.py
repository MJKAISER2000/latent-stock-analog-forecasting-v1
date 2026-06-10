import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GRUOutperformanceModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        output, hidden = self.gru(x)
        last_hidden = output[:, -1, :]
        logits = self.classifier(last_hidden).squeeze(1)
        return logits


def load_sequence_data(sequence_length: int = 12):
    data_path = f"data/processed/sequence_dataset_{sequence_length}m.npz"
    meta_path = f"data/processed/sequence_metadata_{sequence_length}m.csv"

    data = np.load(data_path, allow_pickle=True)
    meta = pd.read_csv(meta_path)
    meta["date"] = pd.to_datetime(meta["date"])

    X = data["X"].astype(np.float32)
    y = data["y_outperform"].astype(np.float32)

    return X, y, meta


def time_split_sequences(X, y, meta):
    train_mask = meta["date"] < "2018-01-01"
    val_mask = (meta["date"] >= "2018-01-01") & (meta["date"] < "2021-01-01")
    test_mask = meta["date"] >= "2021-01-01"

    X_train = X[train_mask.values]
    y_train = y[train_mask.values]
    meta_train = meta[train_mask].copy()

    X_val = X[val_mask.values]
    y_val = y[val_mask.values]
    meta_val = meta[val_mask].copy()

    X_test = X[test_mask.values]
    y_test = y[test_mask.values]
    meta_test = meta[test_mask].copy()

    return X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test


def make_loader(X, y, batch_size=64, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate_model(model, X, y, device, split_name):
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    preds = (probs >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = np.nan

    return {
        "split": split_name,
        "accuracy": accuracy_score(y, preds),
        "f1": f1_score(y, preds, zero_division=0),
        "auc": auc,
    }, probs


def train_gru_model(X_train, y_train, X_val, y_val, input_dim, device):
    model = GRUOutperformanceModel(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=1,
        dropout=0.10,
    ).to(device)

    train_loader = make_loader(X_train, y_train, batch_size=64, shuffle=True)

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count

    if pos_count > 0:
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)
    else:
        pos_weight = torch.tensor([1.0], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_auc = -np.inf
    best_state = None
    patience = 25
    patience_counter = 0

    for epoch in range(1, 251):
        model.train()
        train_losses = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_metrics, _ = evaluate_model(model, X_val, y_val, device, "validation")
        val_auc = val_metrics["auc"]

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 25 == 0:
            print(
                f"epoch={epoch}, "
                f"train_loss={np.mean(train_losses):.5f}, "
                f"val_accuracy={val_metrics['accuracy']:.3f}, "
                f"val_f1={val_metrics['f1']:.3f}, "
                f"val_auc={val_auc:.3f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_val_auc


def top_ranked_portfolio_return(test_meta: pd.DataFrame, probabilities: np.ndarray, top_n: int = 5) -> float:
    temp = test_meta.copy()
    temp["predicted_prob_outperform"] = probabilities

    returns = []

    for date, group in temp.groupby("date"):
        top = group.sort_values("predicted_prob_outperform", ascending=False).head(top_n)
        returns.append(top["future_12m_return"].mean())

    return float(np.mean(returns))


def save_predictions(test_meta, probs):
    pred = test_meta.copy()
    pred["predicted_prob_outperform"] = probs
    pred["rank_by_date"] = pred.groupby("date")["predicted_prob_outperform"].rank(
        ascending=False,
        method="first",
    )

    output_path = "outputs/tables/week7_sequence_predictions_gru.csv"
    pred.to_csv(output_path, index=False)

    print("Saved sequence model predictions to:", output_path)

    return pred


def plot_probability_histogram(pred):
    plt.figure(figsize=(9, 6))
    plt.hist(pred["predicted_prob_outperform"], bins=20)
    plt.title("Week 7 GRU Predicted Outperformance Probabilities")
    plt.xlabel("Predicted probability of SPY outperformance")
    plt.ylabel("Count")
    plt.tight_layout()

    output_path = "outputs/figures/week7_gru_probability_histogram.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print("Saved probability histogram to:", output_path)


def main():
    set_seed(42)

    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    sequence_length = 12

    print("Loading sequence dataset...")
    X, y, meta = load_sequence_data(sequence_length=sequence_length)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Date range:", meta["date"].min(), "to", meta["date"].max())

    (
        X_train,
        y_train,
        meta_train,
        X_val,
        y_val,
        meta_val,
        X_test,
        y_test,
        meta_test,
    ) = time_split_sequences(X, y, meta)

    print("")
    print("Split sizes:")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    input_dim = X_train.shape[2]

    model, best_val_auc = train_gru_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_dim=input_dim,
        device=device,
    )

    rows = []

    train_metrics, train_probs = evaluate_model(model, X_train, y_train, device, "train")
    val_metrics, val_probs = evaluate_model(model, X_val, y_val, device, "validation")
    test_metrics, test_probs = evaluate_model(model, X_test, y_test, device, "test")

    rows.extend([train_metrics, val_metrics, test_metrics])

    metrics = pd.DataFrame(rows)
    metrics["model"] = "gru_sequence_12m"
    metrics["best_val_auc"] = best_val_auc

    top5_return = top_ranked_portfolio_return(meta_test, test_probs, top_n=5)
    top10_return = top_ranked_portfolio_return(meta_test, test_probs, top_n=10)
    spy_return = meta_test["future_12m_spy_return"].mean()
    all_stock_return = meta_test["future_12m_return"].mean()

    portfolio = pd.DataFrame(
        [
            {
                "model": "gru_sequence_12m",
                "top5_avg_future_12m_return": top5_return,
                "top10_avg_future_12m_return": top10_return,
                "all_stock_avg_future_12m_return": all_stock_return,
                "spy_avg_future_12m_return": spy_return,
                "best_val_auc": best_val_auc,
            }
        ]
    )

    metrics_path = "outputs/tables/week7_sequence_model_metrics.csv"
    portfolio_path = "outputs/tables/week7_sequence_portfolio_signal.csv"

    metrics.to_csv(metrics_path, index=False)
    portfolio.to_csv(portfolio_path, index=False)

    pred = save_predictions(meta_test, test_probs)
    plot_probability_histogram(pred)

    report_path = "outputs/reports/week7_sequence_model_summary.txt"

    with open(report_path, "w") as f:
        f.write("Week 7 Sequence Model Summary\n")
        f.write("=============================\n\n")
        f.write("Model:\n")
        f.write("12-month feature sequence -> GRU -> SPY outperformance probability\n\n")
        f.write(f"Input shape: {X.shape}\n")
        f.write(f"Best validation AUC: {best_val_auc}\n\n")
        f.write("Classification metrics:\n")
        f.write(metrics.to_string(index=False))
        f.write("\n\nPortfolio signal:\n")
        f.write(portfolio.to_string(index=False))
        f.write("\n\nTop selected stocks by frequency:\n")
        top = pred[pred["rank_by_date"] <= 5]
        f.write(top["ticker"].value_counts().to_string())
        f.write("\n")

    print("")
    print("Saved metrics to:", metrics_path)
    print("Saved portfolio signal to:", portfolio_path)
    print("Saved report to:", report_path)
    print("")
    print("Classification metrics:")
    print(metrics)
    print("")
    print("Portfolio signal:")
    print(portfolio)


if __name__ == "__main__":
    main()