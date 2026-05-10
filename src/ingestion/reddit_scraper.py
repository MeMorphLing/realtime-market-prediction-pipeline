"""Reddit ingestion using PRAW.

Credentials are loaded from environment variables:
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import praw
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DEFAULT_SUBREDDITS = ["wallstreetbets", "investing", "stocks", "finance"]
RAW_SOCIAL_DIR = Path("data/raw/social")


def _build_client() -> praw.Reddit:
    """Construct a read-only PRAW client from environment variables."""
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "market-bot/1.0")

    if not client_id or not client_secret:
        raise EnvironmentError(
            "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set in the environment"
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
    )


def fetch_reddit_posts(
    subreddits: Optional[list[str]] = None,
    limit: int = 100,
    output_dir: Optional[Path] = None,
) -> list[dict]:
    """Fetch the hottest posts from a list of subreddits.

    Args:
        subreddits: Subreddit names. Defaults to a curated finance set.
        limit: Maximum posts per subreddit.
        output_dir: Optional override for the output directory.

    Returns:
        A list of post dicts with text, score and metadata.
    """
    subs = subreddits or DEFAULT_SUBREDDITS
    out_dir = Path(output_dir) if output_dir else RAW_SOCIAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching Reddit posts from: %s (limit=%d)", subs, limit)

    try:
        reddit = _build_client()
    except Exception:
        logger.exception("Could not build Reddit client")
        raise

    posts: list[dict] = []
    for sub_name in subs:
        try:
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.hot(limit=limit):
                posts.append(
                    {
                        "title": post.title,
                        "selftext": post.selftext or "",
                        "score": int(post.score),
                        "created_utc": float(post.created_utc),
                        "subreddit": sub_name,
                        "source": "reddit",
                    }
                )
        except Exception:
            logger.exception("Failed to fetch r/%s", sub_name)
            continue

    logger.info("Collected %d Reddit posts", len(posts))

    today = datetime.utcnow().strftime("%Y%m%d")
    out_path = out_dir / f"reddit_{today}.parquet"

    try:
        pd.DataFrame(posts).to_parquet(out_path, index=False)
        logger.info("Saved Reddit data to %s", out_path)
    except Exception:
        logger.exception("Failed to persist Reddit data to %s", out_path)
        raise

    return posts


if __name__ == "__main__":
    fetch_reddit_posts()
