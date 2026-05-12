"""Build sliding-window time-series features.

Combines OHLCV bars with hourly-aggregated sentiment signals and yields
sliding windows ready to feed into a sequential model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

RAW_PRICES_DIR = Path("data/raw/prices")
PROCESSED_SENTIMENT_DIR = Path("data/processed/sentiment")
FEATURES_DIR = Path("data/features")

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "avg_sentiment",
    "positive_ratio",
    "negative_ratio",
    "article_count",
]


def _load_prices(ticker: str, prices_dir: Path) -> pd.DataFrame:
    """Concatenate every Parquet file matching ``{ticker}_*.parquet``."""
    files = sorted(prices_dir.glob(f"{ticker}_*.parquet"))
    if not files:
        logger.warning("No price files found for %s in %s", ticker, prices_dir)
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)


def _load_sentiment(sentiment_dir: Path) -> pd.DataFrame:
    """Load every processed sentiment Parquet, one row per text item."""
    if not sentiment_dir.exists():
        logger.warning("Sentiment directory %s does not exist", sentiment_dir)
        return pd.DataFrame(columns=["timestamp", "label", "score"])
    files = sorted(sentiment_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["timestamp", "label", "score"])
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.dropna(subset=["timestamp"])


def _aggregate_sentiment_hourly(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-document sentiment to hourly bins."""
    if sentiment_df.empty:
        return pd.DataFrame(
            columns=["timestamp", "avg_sentiment", "positive_ratio", "negative_ratio", "article_count"]
        )

    df = sentiment_df.copy()
    df["timestamp"] = df["timestamp"].dt.floor("1h")
    df["is_pos"] = (df["label"] == "positive").astype(int)
    df["is_neg"] = (df["label"] == "negative").astype(int)

    grouped = df.groupby("timestamp").agg(
        avg_sentiment=("score", "mean"),
        positive_ratio=("is_pos", "mean"),
        negative_ratio=("is_neg", "mean"),
        article_count=("score", "count"),
    )
    return grouped.reset_index()


def build_time_series(
    ticker: str,
    window_size: int = 30,
    prices_dir: Optional[Path] = None,
    sentiment_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding-window arrays for a single ticker.

    Args:
        ticker: Symbol to process.
        window_size: Number of timesteps per window.
        prices_dir: Override for the raw prices directory.
        sentiment_dir: Override for the processed sentiment directory.
        output_dir: Override for the features output directory.

    Returns:
        A tuple ``(X, y)`` where ``X`` has shape ``(N, window_size, n_features)``
        and ``y`` has shape ``(N,)`` with binary direction labels.
    """
    p_dir = Path(prices_dir) if prices_dir else RAW_PRICES_DIR
    s_dir = Path(sentiment_dir) if sentiment_dir else PROCESSED_SENTIMENT_DIR
    out_dir = Path(output_dir) if output_dir else FEATURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    prices = _load_prices(ticker, p_dir)
    if prices.empty:
        logger.warning("No price data for %s — returning empty arrays", ticker)
        return np.empty((0, window_size, len(FEATURE_COLUMNS))), np.empty((0,))

    prices["timestamp"] = prices["timestamp"].dt.floor("1h")
    sentiment = _aggregate_sentiment_hourly(_load_sentiment(s_dir))

    merged = prices.merge(sentiment, on="timestamp", how="left")
    for col in ["avg_sentiment", "positive_ratio", "negative_ratio", "article_count"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged["direction"] = (merged["close"].shift(-1) > merged["close"]).astype(int)
    merged = merged.dropna(subset=["direction"]).reset_index(drop=True)

    feature_matrix = merged[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets = merged["direction"].to_numpy(dtype=np.int64)

    if len(merged) <= window_size:
        logger.warning(
            "Not enough rows (%d) to build a single window of size %d", len(merged), window_size
        )
        return np.empty((0, window_size, len(FEATURE_COLUMNS))), np.empty((0,))

    windows: list[np.ndarray] = []
    labels: list[int] = []
    for i in range(len(merged) - window_size):
        windows.append(feature_matrix[i : i + window_size])
        labels.append(int(targets[i + window_size - 1]))

    X = np.stack(windows, axis=0)
    y = np.asarray(labels, dtype=np.int64)

    out_path = out_dir / f"{ticker}_windows.parquet"
    try:
        flat = pd.DataFrame(
            {
                "X": [w.tolist() for w in X],
                "y": y.tolist(),
            }
        )
        flat.to_parquet(out_path, index=False)
        logger.info("Saved %d windows to %s", len(X), out_path)
    except Exception:
        logger.exception("Failed to persist features to %s", out_path)

    return X, y


if __name__ == "__main__":
    build_time_series("AAPL")
