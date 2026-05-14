"""FastAPI application factory for the deployment image."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .inference import available_models, load_models, predict
from .schemas import ModelInfo, PredictRequest, PredictResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def create_app() -> FastAPI:
    """Build and configure the FastAPI instance."""
    app = FastAPI(
        title="Market Prediction — Deployment",
        version="1.0.0",
        description="Self-contained inference container. POST a window of features, get a direction.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:
        loaded = load_models()
        logger.info("Startup complete. Models loaded: %s", sorted(loaded.keys()))

    @app.get("/health")
    def health() -> dict:
        """Liveness probe."""
        return {"status": "ok", "models": available_models()}

    @app.get("/models", response_model=list[ModelInfo])
    def models() -> list[ModelInfo]:
        """List models discovered at startup."""
        loaded = available_models()
        return [
            ModelInfo(name=name, available=name in loaded)
            for name in ("rnn", "lstm", "gru")
        ]

    @app.post("/predict", response_model=PredictResponse)
    def do_predict(payload: PredictRequest) -> PredictResponse:
        """Predict next-step direction from a normalized feature window."""
        try:
            result = predict(payload.ticker, payload.window_data, payload.model)
        except LookupError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception:
            logger.exception("Prediction failed for %s", payload.ticker)
            raise HTTPException(status_code=500, detail="Prediction failed")
        return PredictResponse(**result)

    return app
