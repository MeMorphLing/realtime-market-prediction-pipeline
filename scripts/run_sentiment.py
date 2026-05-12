"""Score every raw text record with VADER and save the per-document labels.

Loads everything under ``data/raw/news`` and ``data/raw/social`` (parquet),
scores each text, and writes ``data/processed/sentiment/{source}_{date}.parquet``
with columns: ``timestamp, text, label, score, compound, source``.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sentiment.vader import classify_vader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("sentiment-runner")

RAW_NEWS = PROJECT_ROOT / "data" / "raw" / "news"
RAW_SOCIAL = PROJECT_ROOT / "data" / "raw" / "social"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "sentiment"


def _parse_timestamp(row: pd.Series) -> pd.Timestamp:
    """Pick the best available timestamp column on a row."""
    for key in ("published", "created_at", "created_utc"):
        if key in row and pd.notna(row[key]):
            value = row[key]
            try:
                if key == "created_utc":
                    return pd.Timestamp(float(value), unit="s", tz="UTC")
                return pd.Timestamp(value, tz="UTC")
            except Exception:
                try:
                    return pd.to_datetime(value, utc=True, errors="coerce")
                except Exception:
                    continue
    return pd.Timestamp(datetime.now(tz=timezone.utc))


def _text_from(row: pd.Series) -> str:
    """Pick the best available text column on a row."""
    parts = []
    for key in ("title", "summary", "text", "selftext"):
        v = row.get(key, "")
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return " ".join(parts).strip()


def process_directory(raw_dir: Path, source_default: str) -> pd.DataFrame:
    if not raw_dir.exists():
        log.warning("No directory %s — skipping", raw_dir)
        return pd.DataFrame()

    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        log.warning("No parquet files in %s", raw_dir)
        return pd.DataFrame()

    rows: list[dict] = []
    for f in files:
        df = pd.read_parquet(f)
        if df.empty:
            log.info("  %s is empty", f.name)
            continue
        log.info("  scoring %s (%d rows)", f.name, len(df))
        texts = [_text_from(r) for _, r in df.iterrows()]
        timestamps = [_parse_timestamp(r) for _, r in df.iterrows()]
        sources = df["source"].tolist() if "source" in df.columns else [source_default] * len(df)
        scored = classify_vader(texts)
        for ts, src, item in zip(timestamps, sources, scored):
            rows.append(
                {
                    "timestamp": ts,
                    "text": item["text"],
                    "label": item["label"],
                    "score": float(item["score"]),
                    "compound": float(item["compound"]),
                    "source": src,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")

    news_df = process_directory(RAW_NEWS, source_default="news")
    if not news_df.empty:
        out = PROCESSED / f"news_{today}.parquet"
        news_df.to_parquet(out, index=False)
        log.info(
            "Wrote %d news rows to %s | label counts: %s",
            len(news_df), out, news_df["label"].value_counts().to_dict(),
        )

    social_df = process_directory(RAW_SOCIAL, source_default="social")
    if not social_df.empty:
        out = PROCESSED / f"social_{today}.parquet"
        social_df.to_parquet(out, index=False)
        log.info(
            "Wrote %d social rows to %s | label counts: %s",
            len(social_df), out, social_df["label"].value_counts().to_dict(),
        )

    if news_df.empty and social_df.empty:
        log.warning("Nothing scored — populate data/raw/news or data/raw/social first")


if __name__ == "__main__":
    main()
