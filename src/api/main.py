"""FastAPI prediction service.

Exposes endpoints for health, prediction, current sentiment and a model
registry view. The default model is ``LSTM``; checkpoints are loaded from
the local MLflow store on startup.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models import build_gru, build_lstm, build_rnn

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MLRUNS_DIR = Path(os.getenv("MLRUNS_DIR", "mlruns"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "lstm")
DEFAULT_INPUT_SIZE = int(os.getenv("MODEL_INPUT_SIZE", "9"))

MODEL_FACTORIES = {"rnn": build_rnn, "lstm": build_lstm, "gru": build_gru}

app = FastAPI(
    title="Market Prediction API",
    version="0.1.0",
    description="Real-time market direction prediction powered by sequential DL models.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    """Body schema for the ``/predict`` endpoint."""

    ticker: str = Field(..., min_length=1, max_length=12)
    window_data: list[list[float]] = Field(..., description="Shape (window_size, n_features)")


class PredictResponse(BaseModel):
    """Response schema for the ``/predict`` endpoint."""

    ticker: str
    direction: str
    confidence: float
    model: str


class SentimentResponse(BaseModel):
    """Response schema for the ``/sentiment/{ticker}`` endpoint."""

    ticker: str
    positive: float
    negative: float
    neutral: float
    last_updated: str


class ModelInfo(BaseModel):
    """A single entry returned by the ``/models`` endpoint."""

    name: str
    available: bool
    metrics: dict


_loaded_models: dict[str, torch.nn.Module] = {}


def _load_checkpoint(name: str, input_size: int = DEFAULT_INPUT_SIZE) -> Optional[torch.nn.Module]:
    """Load a checkpoint from ``mlruns/checkpoints/{name}_best.pt`` if present."""
    factory = MODEL_FACTORIES.get(name)
    if factory is None:
        logger.warning("Unknown model %s", name)
        return None

    ckpt_path = MLRUNS_DIR / "checkpoints" / f"{name}_best.pt"
    model = factory(input_size=input_size)
    if not ckpt_path.exists():
        logger.info("No checkpoint at %s; using untrained %s weights", ckpt_path, name)
        model.eval()
        return model

    try:
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        logger.info("Loaded %s weights from %s", name, ckpt_path)
        return model
    except Exception:
        logger.exception("Failed to load checkpoint at %s", ckpt_path)
        return None


@app.on_event("startup")
def _startup() -> None:
    """Pre-load every available model checkpoint at startup."""
    for name in MODEL_FACTORIES:
        model = _load_checkpoint(name)
        if model is not None:
            _loaded_models[name] = model
    logger.info("Loaded models: %s", list(_loaded_models.keys()))


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Predict the next-step direction for ``ticker``.

    The body must contain ``window_data`` shaped ``(window_size, n_features)``.
    """
    model_name = DEFAULT_MODEL
    model = _loaded_models.get(model_name)
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model '{model_name}' is not loaded")

    try:
        x = torch.tensor([payload.window_data], dtype=torch.float32)
        if x.ndim != 3:
            raise ValueError(f"window_data must be 2-D, got shape {tuple(x.shape)}")
        with torch.no_grad():
            logits = model(x)
            prob_up = float(torch.sigmoid(logits).item())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Prediction failed for %s", payload.ticker)
        raise HTTPException(status_code=500, detail="Prediction failed")

    direction = "up" if prob_up >= 0.5 else "down"
    confidence = prob_up if direction == "up" else 1.0 - prob_up
    return PredictResponse(
        ticker=payload.ticker.upper(),
        direction=direction,
        confidence=round(confidence, 4),
        model=model_name,
    )


@app.get("/sentiment/{ticker}", response_model=SentimentResponse)
def sentiment(ticker: str) -> SentimentResponse:
    """Return the most recent sentiment distribution for a ticker.

    This is a placeholder that returns balanced values when no processed
    sentiment store is available — wire it up to your own aggregator when
    running in production.
    """
    if not ticker or len(ticker) > 12:
        raise HTTPException(status_code=400, detail="Invalid ticker")

    return SentimentResponse(
        ticker=ticker.upper(),
        positive=0.4,
        negative=0.3,
        neutral=0.3,
        last_updated=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )


@app.get("/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    """List the models known to the API and their availability."""
    out: list[ModelInfo] = []
    for name in MODEL_FACTORIES:
        out.append(
            ModelInfo(
                name=name,
                available=name in _loaded_models,
                metrics={},  # populated by your evaluation pipeline
            )
        )
    return out
