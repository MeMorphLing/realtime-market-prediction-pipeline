"""Test-set evaluation utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
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
from torch.utils.data import DataLoader

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MLRUNS_DIR = Path("mlruns")


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
