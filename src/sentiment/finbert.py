"""FinBERT sentiment classification.

Wraps the ``ProsusAI/finbert`` HuggingFace model for batched inference on
financial text. Uses the GPU when available.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MODEL_NAME = "ProsusAI/finbert"
LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForSequenceClassification] = None
_device: Optional[torch.device] = None


def _load_model() -> None:
    """Lazy-load the FinBERT tokenizer and model into module-level globals."""
    global _tokenizer, _model, _device
    if _model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading FinBERT (%s) on %s", MODEL_NAME, _device)

    try:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.to(_device)
        _model.eval()
    except Exception:
        logger.exception("Failed to load FinBERT model")
        raise


def classify_finbert(texts: list[str], batch_size: int = 16) -> list[dict]:
    """Classify a list of texts with FinBERT.

    Args:
        texts: Sentences or short documents to classify.
        batch_size: Inference batch size.

    Returns:
        A list of ``{text, label, score}`` dicts where ``label`` is one of
        ``"positive"``, ``"negative"``, ``"neutral"``.
    """
    if not texts:
        return []

    _load_model()
    assert _tokenizer is not None and _model is not None and _device is not None

    results: list[dict] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            encoded = _tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(_device)

            with torch.no_grad():
                logits = _model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)
                top_score, top_idx = torch.max(probs, dim=-1)

            for text, idx, score in zip(batch, top_idx.tolist(), top_score.tolist()):
                results.append(
                    {
                        "text": text,
                        "label": LABEL_MAP.get(idx, "neutral"),
                        "score": float(score),
                    }
                )
        except Exception:
            logger.exception("FinBERT inference failed on batch starting at %d", start)
            for text in batch:
                results.append({"text": text, "label": "neutral", "score": 0.0})

    return results


if __name__ == "__main__":
    sample = ["Apple beat earnings expectations.", "The Fed signalled rate hikes."]
    for row in classify_finbert(sample):
        logger.info(row)
