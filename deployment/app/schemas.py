"""Pydantic request/response schemas for the inference API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Body for ``POST /predict``."""

    ticker: str = Field(..., min_length=1, max_length=12, examples=["AAPL"])
    window_data: list[list[float]] = Field(
        ...,
        description="Shape (window_size, n_features). Must already be normalized.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Override the default model: one of 'rnn', 'lstm', 'gru'.",
    )


class PredictResponse(BaseModel):
    """Body for ``POST /predict``."""

    ticker: str
    direction: str
    confidence: float
    model: str


class ModelInfo(BaseModel):
    """Single row of ``GET /models``."""

    name: str
    available: bool
