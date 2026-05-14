"""FastAPI prediction service.

Exposes endpoints for health, prediction, current sentiment and a model
registry view. The default model is ``LSTM``; checkpoints are loaded from
the local MLflow store on startup.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.models import build_gru, build_lstm, build_rnn

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MLRUNS_DIR = Path(os.getenv("MLRUNS_DIR", "mlruns"))
CHECKPOINTS_DIR = Path(os.getenv("CHECKPOINTS_DIR", "checkpoints"))
FEATURES_DIR = PROJECT_ROOT / os.getenv("FEATURES_DIR", "data/features")
SENTIMENT_DIR = PROJECT_ROOT / os.getenv("SENTIMENT_DIR", "data/processed/sentiment")
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
    """Body schema for the ``/predict`` endpoint.

    ``window_data`` is optional — if omitted, the server loads the latest
    sliding window from ``data/features/{ticker}_windows.parquet``.
    """

    ticker: str = Field(..., min_length=1, max_length=12)
    window_data: Optional[list[list[float]]] = Field(
        default=None,
        description="Shape (window_size, n_features). If omitted, latest window is loaded server-side.",
    )


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

    ckpt_path = CHECKPOINTS_DIR / f"{name}_best.pt"
    legacy_path = MLRUNS_DIR / "checkpoints" / f"{name}_best.pt"
    if not ckpt_path.exists() and legacy_path.exists():
        ckpt_path = legacy_path
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


def _load_latest_window(ticker: str) -> Optional[np.ndarray]:
    """Load the most recent sliding window for a ticker from disk."""
    path = FEATURES_DIR / f"{ticker.upper()}_windows.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    # Each "X" row is a (window_size, n_features) nested sequence.
    last_row = df.iloc[-1]["X"]
    window = np.stack([np.asarray(step, dtype=np.float32) for step in last_row])
    return window


def _available_tickers() -> list[str]:
    """List every ticker that has a features parquet on disk."""
    if not FEATURES_DIR.exists():
        return []
    return sorted(p.stem.replace("_windows", "") for p in FEATURES_DIR.glob("*_windows.parquet"))


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Predict the next-step direction for ``ticker``.

    If ``window_data`` is omitted, the latest window is loaded from
    ``data/features/{ticker}_windows.parquet``.
    """
    model_name = DEFAULT_MODEL
    model = _loaded_models.get(model_name)
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model '{model_name}' is not loaded")

    ticker = payload.ticker.upper()
    window_data = payload.window_data
    if not window_data:
        loaded = _load_latest_window(ticker)
        if loaded is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No features for {ticker}. Run ingestion + feature build first, "
                    f"or pass window_data explicitly. Available: {_available_tickers()}"
                ),
            )
        window_data = loaded.tolist()

    try:
        x = torch.tensor([window_data], dtype=torch.float32)
        if x.ndim != 3:
            raise ValueError(f"window_data must be 2-D, got shape {tuple(x.shape)}")
        with torch.no_grad():
            logits = model(x)
            prob_up = float(torch.sigmoid(logits).item())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Prediction failed for %s", ticker)
        raise HTTPException(status_code=500, detail="Prediction failed")

    direction = "up" if prob_up >= 0.5 else "down"
    confidence = prob_up if direction == "up" else 1.0 - prob_up
    return PredictResponse(
        ticker=ticker,
        direction=direction,
        confidence=round(confidence, 4),
        model=model_name,
    )


def _aggregate_sentiment(ticker: str) -> Optional[tuple[dict[str, float], str]]:
    """Compute positive/neutral/negative ratios from processed sentiment files.

    Filters to rows whose text contains the ticker symbol (case-insensitive).
    Falls back to the global aggregate if the filter returns nothing.
    """
    if not SENTIMENT_DIR.exists():
        return None
    files = sorted(SENTIMENT_DIR.glob("*.parquet"))
    if not files:
        return None
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return None

    ticker_up = ticker.upper()
    filtered = df[df["text"].str.contains(rf"\b{ticker_up}\b", case=False, na=False, regex=True)]
    used = filtered if not filtered.empty else df

    counts = used["label"].value_counts(normalize=True).to_dict()
    ratios = {
        "positive": float(counts.get("positive", 0.0)),
        "negative": float(counts.get("negative", 0.0)),
        "neutral": float(counts.get("neutral", 0.0)),
    }

    if "timestamp" in used.columns and not used.empty:
        last = pd.to_datetime(used["timestamp"], utc=True, errors="coerce").max()
        last_iso = last.isoformat() if pd.notna(last) else datetime.now(tz=timezone.utc).isoformat()
    else:
        last_iso = datetime.now(tz=timezone.utc).isoformat()

    return ratios, last_iso


@app.get("/sentiment/{ticker}", response_model=SentimentResponse)
def sentiment(ticker: str) -> SentimentResponse:
    """Return the most recent sentiment distribution for a ticker."""
    if not ticker or len(ticker) > 12:
        raise HTTPException(status_code=400, detail="Invalid ticker")

    agg = _aggregate_sentiment(ticker)
    if agg is None:
        # Honest fallback when no sentiment has been scored yet.
        return SentimentResponse(
            ticker=ticker.upper(),
            positive=0.0,
            negative=0.0,
            neutral=0.0,
            last_updated=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        )

    ratios, last_iso = agg
    return SentimentResponse(
        ticker=ticker.upper(),
        positive=round(ratios["positive"], 4),
        negative=round(ratios["negative"], 4),
        neutral=round(ratios["neutral"], 4),
        last_updated=last_iso,
    )


@app.get("/tickers")
def list_tickers() -> dict:
    """Return all tickers with on-disk feature windows ready for prediction."""
    return {"tickers": _available_tickers()}


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


# Serve the dashboard at "/". Mounted last so API routes match first.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
    logger.info("Serving frontend from %s at /", _FRONTEND_DIR)
