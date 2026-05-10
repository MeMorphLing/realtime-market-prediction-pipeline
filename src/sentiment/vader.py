"""VADER lexicon-based sentiment analysis.

Used as a fast complementary signal next to FinBERT.
"""

from __future__ import annotations

import logging
from typing import Optional

import nltk
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

_analyzer: Optional[SentimentIntensityAnalyzer] = None


def _ensure_lexicon() -> SentimentIntensityAnalyzer:
    """Download the VADER lexicon on first use and cache the analyzer."""
    global _analyzer
    if _analyzer is not None:
        return _analyzer

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        logger.info("Downloading VADER lexicon")
        try:
            nltk.download("vader_lexicon", quiet=True)
        except Exception:
            logger.exception("Failed to download VADER lexicon")
            raise

    _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def classify_vader(texts: list[str]) -> list[dict]:
    """Score a list of texts with VADER.

    The compound score is mapped to a categorical label using the standard
    thresholds: ``>0.05`` positive, ``<-0.05`` negative, otherwise neutral.

    Args:
        texts: Sentences or short documents to classify.

    Returns:
        A list of ``{text, label, score, compound}`` dicts.
    """
    if not texts:
        return []

    analyzer = _ensure_lexicon()

    results: list[dict] = []
    for text in texts:
        try:
            scores = analyzer.polarity_scores(text or "")
            compound = float(scores["compound"])
            if compound > 0.05:
                label = "positive"
            elif compound < -0.05:
                label = "negative"
            else:
                label = "neutral"
            results.append(
                {
                    "text": text,
                    "label": label,
                    "score": float(scores[label[:3]]) if label[:3] in scores else compound,
                    "compound": compound,
                }
            )
        except Exception:
            logger.exception("VADER scoring failed for one input")
            results.append({"text": text, "label": "neutral", "score": 0.0, "compound": 0.0})

    return results


if __name__ == "__main__":
    sample = ["Stocks soared after the announcement.", "Markets crashed overnight."]
    for row in classify_vader(sample):
        logger.info(row)
