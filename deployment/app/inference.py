"""Model loading and prediction logic for the deployment image.

The container expects checkpoints under ``MODEL_DIR`` (default ``./model``).
Each file is named ``{name}_best.pt`` and contains a ``state_dict``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch

from .models import MODEL_FACTORIES

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "lstm")
INPUT_SIZE = int(os.getenv("MODEL_INPUT_SIZE", "9"))

_loaded: dict[str, torch.nn.Module] = {}


def load_models() -> dict[str, torch.nn.Module]:
    """Load every checkpoint found under ``MODEL_DIR``."""
    _loaded.clear()
    for name, factory in MODEL_FACTORIES.items():
        path = MODEL_DIR / f"{name}_best.pt"
        if not path.exists():
            logger.warning("No checkpoint for %s at %s — skipping", name, path)
            continue
        try:
            model = factory(input_size=INPUT_SIZE)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            _loaded[name] = model
            logger.info("Loaded %s from %s", name, path)
        except Exception:
            logger.exception("Failed to load %s from %s", name, path)
    return _loaded


def available_models() -> list[str]:
    """Return the names of models currently loaded in memory."""
    return sorted(_loaded.keys())


def predict(ticker: str, window_data: list[list[float]], model_name: Optional[str] = None) -> dict:
    """Run a forward pass with ``model_name`` (or the default) on a single window.

    Returns a dict with ``ticker, direction, confidence, model``.
    Raises ``LookupError`` if the requested model is not loaded, and
    ``ValueError`` if ``window_data`` has the wrong shape.
    """
    name = (model_name or DEFAULT_MODEL).lower()
    model = _loaded.get(name)
    if model is None:
        raise LookupError(
            f"Model '{name}' not loaded. Available: {available_models()}"
        )

    x = torch.tensor([window_data], dtype=torch.float32)
    if x.ndim != 3:
        raise ValueError(f"window_data must be 2-D, got shape {tuple(x.shape)}")

    with torch.no_grad():
        prob_up = float(torch.sigmoid(model(x)).item())

    direction = "up" if prob_up >= 0.5 else "down"
    confidence = prob_up if direction == "up" else 1.0 - prob_up
    return {
        "ticker": ticker.upper(),
        "direction": direction,
        "confidence": round(confidence, 4),
        "model": name,
    }
