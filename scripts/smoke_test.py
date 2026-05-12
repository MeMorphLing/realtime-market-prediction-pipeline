"""End-to-end smoke test.

Generates synthetic OHLCV data, builds windows, trains RNN/LSTM/GRU
for a few epochs each, logs everything to a local MLflow file store,
runs evaluation + plots, and prints a summary.

Run from the project root:

    python scripts/smoke_test.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Use a local file-based MLflow store so the script works without a server.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["MLFLOW_TRACKING_URI"] = (PROJECT_ROOT / "mlruns").as_uri()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.evaluation.metrics import evaluate_model, plot_metrics  # noqa: E402
from src.features.time_series import build_time_series  # noqa: E402
from src.models import build_gru, build_lstm, build_rnn  # noqa: E402
from src.models.train import run_all_experiments  # noqa: E402
from src.sentiment.vader import classify_vader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("smoke")

TICKER = "SMOKE"
N_ROWS = 600
WINDOW = 20
EPOCHS = 10
BATCH_SIZE = 16


def make_synthetic_prices(ticker: str = TICKER, n: int = N_ROWS) -> pd.DataFrame:
    """Generate a deterministic random-walk OHLCV frame."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    walk = np.cumsum(rng.normal(0, 1, n)) + 100
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": walk + rng.normal(0, 0.2, n),
            "high": walk + rng.uniform(0, 0.5, n),
            "low": walk - rng.uniform(0, 0.5, n),
            "close": walk,
            "volume": rng.integers(1000, 10000, n),
            "ticker": ticker,
        }
    )
    return df


def step_sentiment() -> None:
    log.info("STEP 1 — VADER sentiment sanity check")
    samples = [
        "Stocks rally on strong earnings.",
        "Markets crash overnight on rate fears.",
        "Trading was flat ahead of the Fed meeting.",
    ]
    for row in classify_vader(samples):
        log.info("  %s", row)


def step_features() -> tuple[np.ndarray, np.ndarray]:
    log.info("STEP 2 — synthetic OHLCV → sliding windows")
    raw_dir = PROJECT_ROOT / "data" / "raw" / "prices"
    raw_dir.mkdir(parents=True, exist_ok=True)
    df = make_synthetic_prices()
    df.to_parquet(raw_dir / f"{TICKER}_smoke.parquet", index=False)
    X, y = build_time_series(TICKER, window_size=WINDOW)
    log.info("  X.shape=%s  y.shape=%s  pos_rate=%.2f", X.shape, y.shape, float(y.mean()))
    if len(X) == 0:
        raise RuntimeError("No windows produced — feature build failed")
    return X, y


def step_train(X: np.ndarray, y: np.ndarray) -> dict:
    log.info("STEP 3 — train RNN/LSTM/GRU (%d epochs each)", EPOCHS)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    log.info("  train=%d val=%d", len(X_train), len(X_val))
    return run_all_experiments(X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE)


def step_evaluate(X: np.ndarray, y: np.ndarray) -> dict:
    log.info("STEP 4 — load each best checkpoint and re-evaluate")
    split = int(len(X) * 0.8)
    X_val, y_val = X[split:], y[split:]

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1),
        ),
        batch_size=BATCH_SIZE,
    )

    factories = {"rnn": build_rnn, "lstm": build_lstm, "gru": build_gru}
    results: dict = {}
    candidate_dirs = [PROJECT_ROOT / "checkpoints", PROJECT_ROOT / "mlruns" / "checkpoints"]
    for name, factory in factories.items():
        ckpt = next((d / f"{name}_best.pt" for d in candidate_dirs if (d / f"{name}_best.pt").exists()), None)
        if ckpt is None:
            log.warning("  no checkpoint found for %s — skipping", name)
            continue
        model = factory(input_size=X.shape[-1])
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        results[name] = evaluate_model(model, val_loader)

    if results:
        out = plot_metrics(results)
        log.info("  comparison plot saved to %s", out)
    return results


def main() -> None:
    log.info("== Market Prediction smoke test ==")
    log.info("  project root: %s", PROJECT_ROOT)
    log.info("  MLflow URI:   %s", os.environ["MLFLOW_TRACKING_URI"])
    log.info("  CUDA:         %s", torch.cuda.is_available())

    step_sentiment()
    X, y = step_features()
    train_results = step_train(X, y)
    eval_results = step_evaluate(X, y)

    log.info("== Summary ==")
    for name in ("rnn", "lstm", "gru"):
        train = train_results.get(name, {})
        evalr = eval_results.get(name, {})
        log.info(
            "  %-4s  best_val_f1=%.3f  test_acc=%.3f  test_f1=%.3f  rmse=%.3f",
            name,
            float(train.get("val_f1", 0.0)),
            float(evalr.get("accuracy", 0.0)),
            float(evalr.get("f1_score", 0.0)),
            float(evalr.get("rmse", 0.0)),
        )
    log.info("Done. Inspect mlruns/, mlruns/checkpoints/, mlruns/comparison.png.")


if __name__ == "__main__":
    main()
