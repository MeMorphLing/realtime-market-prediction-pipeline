"""Twitter / X ingestion via Tweepy.

Requires the environment variable ``TWITTER_BEARER_TOKEN`` (Twitter API v2).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import tweepy
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DEFAULT_QUERY = "(stock OR market OR trading) lang:en -is:retweet"
RAW_SOCIAL_DIR = Path("data/raw/social")


def _build_client() -> tweepy.Client:
    """Construct a Twitter v2 client using the bearer token."""
    token = os.getenv("TWITTER_BEARER_TOKEN")
    if not token:
        raise EnvironmentError("TWITTER_BEARER_TOKEN must be set in the environment")
    return tweepy.Client(bearer_token=token, wait_on_rate_limit=True)


def fetch_tweets(
    query: str = DEFAULT_QUERY,
    max_results: int = 100,
    output_dir: Optional[Path] = None,
) -> list[dict]:
    """Search recent tweets matching ``query``.

    Args:
        query: Twitter v2 search query.
        max_results: Maximum number of tweets to retrieve (10-100).
        output_dir: Optional override for the output directory.

    Returns:
        A list of dicts: ``{text, created_at, public_metrics, source}``.
    """
    out_dir = Path(output_dir) if output_dir else RAW_SOCIAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Searching tweets: %r (max_results=%d)", query, max_results)

    try:
        client = _build_client()
        response = client.search_recent_tweets(
            query=query,
            max_results=max(10, min(100, max_results)),
            tweet_fields=["created_at", "public_metrics", "lang"],
        )
    except Exception:
        logger.exception("Twitter search failed")
        raise

    tweets: list[dict] = []
    for tweet in response.data or []:
        tweets.append(
            {
                "text": tweet.text,
                "created_at": str(tweet.created_at) if tweet.created_at else None,
                "public_metrics": dict(tweet.public_metrics or {}),
                "source": "twitter",
            }
        )

    logger.info("Collected %d tweets", len(tweets))

    today = datetime.utcnow().strftime("%Y%m%d")
    out_path = out_dir / f"twitter_{today}.parquet"

    try:
        pd.DataFrame(tweets).to_parquet(out_path, index=False)
        logger.info("Saved tweets to %s", out_path)
    except Exception:
        logger.exception("Failed to persist tweets to %s", out_path)
        raise

    return tweets


if __name__ == "__main__":
    fetch_tweets()
