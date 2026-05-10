"""Reuters Business News RSS ingestion."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import feedparser
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

REUTERS_FEED_URL = "http://feeds.reuters.com/reuters/businessNews"
RAW_NEWS_DIR = Path("data/raw/news")


def fetch_reuters_news(
    max_articles: int = 50,
    feed_url: str = REUTERS_FEED_URL,
    output_dir: Optional[Path] = None,
) -> list[dict]:
    """Pull the latest Reuters business news headlines via RSS.

    Args:
        max_articles: Maximum number of feed entries to keep.
        feed_url: RSS endpoint to query.
        output_dir: Optional override for the output directory.

    Returns:
        A list of ``{title, summary, published, source}`` dicts.
    """
    out_dir = Path(output_dir) if output_dir else RAW_NEWS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching Reuters RSS feed: %s", feed_url)

    try:
        feed = feedparser.parse(feed_url)
    except Exception:
        logger.exception("Failed to parse Reuters feed")
        raise

    if getattr(feed, "bozo", 0) and not feed.entries:
        logger.warning("Reuters feed returned no entries (bozo=%s)", feed.bozo)

    articles: list[dict] = []
    for entry in feed.entries[:max_articles]:
        articles.append(
            {
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "published": entry.get("published", ""),
                "source": "reuters",
            }
        )

    logger.info("Parsed %d Reuters articles", len(articles))

    today = datetime.utcnow().strftime("%Y%m%d")
    out_path = out_dir / f"reuters_{today}.parquet"

    try:
        pd.DataFrame(articles).to_parquet(out_path, index=False)
        logger.info("Saved Reuters news to %s", out_path)
    except Exception:
        logger.exception("Failed to persist Reuters news to %s", out_path)
        raise

    return articles


if __name__ == "__main__":
    fetch_reuters_news()
