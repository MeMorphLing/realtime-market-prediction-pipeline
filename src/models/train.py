"""Training loop and multi-model experiment runner.

Runs RNN/LSTM/GRU experiments and logs everything to MLflow.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .gru import build_gru
from .lstm import build_lstm
from .rnn import build_rnn

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MLRUNS_DIR = Path("mlruns")
CHECKPOINTS_DIR = Path(os.getenv("CHECKPOINTS_DIR", "checkpoints"))
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Wrap numpy arrays in a PyTorch DataLoader."""
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    return DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    checkpoint_dir: Optional[Path] = None,
    model_name: str = "model",
) -> dict:
    """Train ``model`` and log metrics to MLflow.

    Args:
        model: PyTorch module producing logits of shape ``(batch, 1)``.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        epochs: Maximum number of epochs.
        lr: Adam learning rate.
        checkpoint_dir: Where to save the best checkpoint (defaults to ``mlruns/checkpoints``).
        model_name: Used in filenames and MLflow run tags.

    Returns:
        Dict with the best epoch's validation metrics.
    """
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINTS_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = _device()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    best_metrics: dict = {}
    best_path = ckpt_dir / f"{model_name}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * xb.size(0)
        train_loss = running_loss / max(1, len(train_loader.dataset))

        val_loss, val_acc, val_f1 = _evaluate(model, val_loader, criterion, device)
        logger.info(
            "[%s] epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_f1=%.4f",
            model_name, epoch, epochs, train_loss, val_loss, val_acc, val_f1,
        )

        try:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_f1": val_f1,
                },
                step=epoch,
            )
        except Exception:
            logger.exception("MLflow metric logging failed at epoch %d", epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
            }
            try:
                torch.save(model.state_dict(), best_path)
                logger.info("New best checkpoint saved to %s", best_path)
            except Exception:
                logger.exception("Failed to save checkpoint to %s", best_path)

    try:
        mlflow.pytorch.log_model(model, artifact_path=model_name)
    except Exception:
        logger.exception("MLflow model logging failed for %s", model_name)

    return best_metrics


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Compute validation loss/accuracy/F1 on a dataloader."""
    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(float(loss.item()) * xb.size(0))
            preds.extend((torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(int).flatten().tolist())
            targets.extend(yb.cpu().numpy().astype(int).flatten().tolist())
    n = max(1, len(loader.dataset))
    return (
        sum(losses) / n,
        float(accuracy_score(targets, preds)) if preds else 0.0,
        float(f1_score(targets, preds, zero_division=0)) if preds else 0.0,
    )


def run_all_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
) -> dict:
    """Train RNN, LSTM and GRU sequentially as separate MLflow runs.

    Args:
        X_train: Training windows of shape ``(N, T, F)``.
        y_train: Binary labels of shape ``(N,)``.
        X_val: Validation windows.
        y_val: Validation labels.
        epochs: Epochs per model.
        batch_size: Mini-batch size.

    Returns:
        Mapping ``{model_name: best_metrics_dict}``.
    """
    if X_train.size == 0 or X_val.size == 0:
        logger.warning("Empty training or validation data; skipping experiments")
        return {}

    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment("market_prediction")
    except Exception:
        logger.exception("MLflow setup failed; continuing without tracking server")

    input_size = X_train.shape[-1]
    train_loader = _to_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _to_loader(X_val, y_val, batch_size, shuffle=False)

    factories = {"rnn": build_rnn, "lstm": build_lstm, "gru": build_gru}
    results: dict = {}

    for name, factory in factories.items():
        logger.info("=== Training %s ===", name)
        try:
            with mlflow.start_run(run_name=name):
                model = factory(input_size=input_size)
                mlflow.log_params(
                    {
                        "model": name,
                        "input_size": input_size,
                        "epochs": epochs,
                        "batch_size": batch_size,
                    }
                )
                metrics = train_model(
                    model,
                    train_loader,
                    val_loader,
                    epochs=epochs,
                    model_name=name,
                )
                results[name] = metrics
        except Exception:
            logger.exception("Experiment for %s failed", name)
            results[name] = {"error": "training_failed"}

    return results


def _load_windows_from_dir(features_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load every ``*_windows.parquet`` from ``features_dir`` and stack them."""
    files = sorted(features_dir.glob("*_windows.parquet"))
    if not files:
        raise FileNotFoundError(f"No *_windows.parquet under {features_dir}")
    import pandas as pd

    Xs, ys = [], []
    for f in files:
        df = pd.read_parquet(f)
        windows = []
        for row in df["X"]:
            # row is a (window_size,) object array of (n_features,) arrays
            windows.append(np.stack([np.asarray(step, dtype=np.float32) for step in row]))
        Xs.append(np.stack(windows))
        ys.append(np.asarray(df["y"].tolist(), dtype=np.int64))
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def main() -> None:
    """DVC entry point. Loads features, splits 80/20, runs all experiments."""
    features_dir = Path(os.getenv("FEATURES_DIR", "data/features"))
    epochs = int(os.getenv("TRAIN_EPOCHS", "10"))
    batch_size = int(os.getenv("TRAIN_BATCH_SIZE", "32"))

    X, y = _load_windows_from_dir(features_dir)
    logger.info("Loaded %d windows of shape %s from %s", len(X), X.shape[1:], features_dir)

    split = int(len(X) * 0.8)
    results = run_all_experiments(
        X[:split], y[:split], X[split:], y[split:], epochs=epochs, batch_size=batch_size
    )
    logger.info("Best metrics: %s", results)


if __name__ == "__main__":
    main()
