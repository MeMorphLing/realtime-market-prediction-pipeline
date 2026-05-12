"""Test-set evaluation utilities."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MLRUNS_DIR = Path("mlruns")
REPORTS_DIR = Path("reports")


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> dict:
    """Run inference and compute classification + regression metrics.

    Args:
        model: Trained PyTorch model emitting logits of shape ``(batch, 1)``.
        test_loader: DataLoader yielding ``(features, label)`` tensors.

    Returns:
        Dict with ``accuracy, f1_score, precision, recall, rmse, confusion_matrix``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    preds: list[int] = []
    probs: list[float] = []
    targets: list[int] = []

    try:
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                p = torch.sigmoid(logits).cpu().numpy().flatten()
                probs.extend(p.tolist())
                preds.extend((p >= 0.5).astype(int).tolist())
                targets.extend(yb.cpu().numpy().astype(int).flatten().tolist())
    except Exception:
        logger.exception("Evaluation forward pass failed")
        raise

    if not preds:
        logger.warning("Empty test loader; returning zeroed metrics")
        return {
            "accuracy": 0.0,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "rmse": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    metrics = {
        "accuracy": float(accuracy_score(targets, preds)),
        "f1_score": float(f1_score(targets, preds, zero_division=0)),
        "precision": float(precision_score(targets, preds, zero_division=0)),
        "recall": float(recall_score(targets, preds, zero_division=0)),
        "rmse": float(np.sqrt(mean_squared_error(targets, probs))),
        "confusion_matrix": confusion_matrix(targets, preds).tolist(),
    }
    logger.info("Test metrics: %s", metrics)
    return metrics


def plot_metrics(metrics_dict: dict, output_path: Path | None = None) -> Path:
    """Render a comparison bar chart for several model results.

    Args:
        metrics_dict: Mapping ``{model_name: metrics_dict}``.
        output_path: Where to save the PNG. Defaults to ``mlruns/comparison.png``.

    Returns:
        The path the figure was written to.
    """
    out = Path(output_path) if output_path else MLRUNS_DIR / "comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    if not metrics_dict:
        logger.warning("Empty metrics dict; skipping plot")
        return out

    metric_keys = ["accuracy", "f1_score", "precision", "recall"]
    model_names = list(metrics_dict.keys())
    n_models = len(model_names)
    n_metrics = len(metric_keys)

    x = np.arange(n_metrics)
    bar_w = 0.8 / max(1, n_models)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(model_names):
        values = [float(metrics_dict[name].get(k, 0.0)) for k in metric_keys]
        ax.bar(x + i * bar_w, values, width=bar_w, label=name)

    ax.set_xticks(x + bar_w * (n_models - 1) / 2)
    ax.set_xticklabels(metric_keys)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison")
    ax.legend()

    try:
        fig.tight_layout()
        fig.savefig(out, dpi=120)
    except Exception:
        logger.exception("Failed to save comparison plot to %s", out)
    finally:
        plt.close(fig)

    logger.info("Comparison plot saved to %s", out)
    return out


def _load_windows_from_dir(features_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate every ``*_windows.parquet`` under ``features_dir``."""
    files = sorted(features_dir.glob("*_windows.parquet"))
    if not files:
        raise FileNotFoundError(f"No *_windows.parquet under {features_dir}")
    Xs, ys = [], []
    for f in files:
        df = pd.read_parquet(f)
        windows = []
        for row in df["X"]:
            windows.append(np.stack([np.asarray(step, dtype=np.float32) for step in row]))
        Xs.append(np.stack(windows))
        ys.append(np.asarray(df["y"].tolist(), dtype=np.int64))
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def main() -> None:
    """DVC entry point. Loads features + checkpoints, writes metrics.json + plot."""
    from src.models import build_gru, build_lstm, build_rnn

    features_dir = Path(os.getenv("FEATURES_DIR", "data/features"))
    checkpoints_dir = Path(os.getenv("CHECKPOINTS_DIR", "checkpoints"))
    reports_dir = Path(os.getenv("REPORTS_DIR", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    X, y = _load_windows_from_dir(features_dir)
    split = int(len(X) * 0.8)
    X_val, y_val = X[split:], y[split:]
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1),
        ),
        batch_size=32,
    )

    factories = {"rnn": build_rnn, "lstm": build_lstm, "gru": build_gru}
    all_metrics: dict[str, dict] = {}
    for name, factory in factories.items():
        ckpt = checkpoints_dir / f"{name}_best.pt"
        if not ckpt.exists():
            logger.warning("No checkpoint at %s — skipping %s", ckpt, name)
            continue
        model = factory(input_size=X.shape[-1])
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        all_metrics[name] = evaluate_model(model, loader)

    metrics_path = reports_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(all_metrics, fp, indent=2)
    logger.info("Wrote metrics to %s", metrics_path)

    if all_metrics:
        plot_metrics(all_metrics, output_path=reports_dir / "comparison.png")


if __name__ == "__main__":
    main()
