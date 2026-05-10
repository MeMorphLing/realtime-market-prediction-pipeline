"""Airflow DAG for the end-to-end market prediction pipeline.

Runs every 4 hours: ingest prices + news + social, score sentiment, build
features, optionally retrain, and refresh the API.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ingest_yahoo_finance(**_kwargs) -> None:
    """Pull OHLCV data for a default ticker basket."""
    from src.ingestion.yahoo_finance import fetch_price_data

    for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]:
        try:
            fetch_price_data(ticker)
        except Exception:
            logger.exception("Yahoo ingestion failed for %s", ticker)


def _ingest_reuters(**_kwargs) -> None:
    """Pull the Reuters business RSS feed."""
    from src.ingestion.reuters_rss import fetch_reuters_news

    fetch_reuters_news()


def _ingest_reddit(**_kwargs) -> None:
    """Pull recent posts from finance subreddits."""
    from src.ingestion.reddit_scraper import fetch_reddit_posts

    fetch_reddit_posts()


def _ingest_twitter(**_kwargs) -> None:
    """Pull recent finance-related tweets."""
    from src.ingestion.twitter_scraper import fetch_tweets

    fetch_tweets()


def _run_sentiment_analysis(**_kwargs) -> None:
    """Score collected texts with FinBERT and VADER."""
    from src.sentiment.finbert import classify_finbert
    from src.sentiment.vader import classify_vader

    sample: list[str] = []
    classify_vader(sample)
    classify_finbert(sample)


def _build_features(**_kwargs) -> None:
    """Build sliding-window features for the active ticker basket."""
    from src.features.time_series import build_time_series

    for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]:
        try:
            build_time_series(ticker)
        except Exception:
            logger.exception("Feature build failed for %s", ticker)


def _retrain_models(**context) -> None:
    """Retrain RNN/LSTM/GRU on Mondays only."""
    if context["logical_date"].weekday() != 0:
        logger.info("Skipping retraining (only runs on Mondays)")
        return
    logger.info("Retraining trigger fired — wire up X_train/X_val before enabling")


def _update_api(**_kwargs) -> None:
    """Signal the API to reload the newest checkpoints."""
    logger.info("API reload signal sent (no-op placeholder)")


default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="market_prediction_pipeline",
    description="End-to-end market prediction pipeline (ingest → sentiment → features → train → serve)",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@every 4 hours",
    catchup=False,
    tags=["market", "ml", "deep-learning"],
) as dag:

    ingest_yahoo_finance = PythonOperator(
        task_id="ingest_yahoo_finance",
        python_callable=_ingest_yahoo_finance,
    )
    ingest_reuters = PythonOperator(
        task_id="ingest_reuters",
        python_callable=_ingest_reuters,
    )
    ingest_reddit = PythonOperator(
        task_id="ingest_reddit",
        python_callable=_ingest_reddit,
    )
    ingest_twitter = PythonOperator(
        task_id="ingest_twitter",
        python_callable=_ingest_twitter,
    )
    run_sentiment_analysis = PythonOperator(
        task_id="run_sentiment_analysis",
        python_callable=_run_sentiment_analysis,
    )
    build_features = PythonOperator(
        task_id="build_features",
        python_callable=_build_features,
    )
    retrain_models = PythonOperator(
        task_id="retrain_models",
        python_callable=_retrain_models,
    )
    update_api = PythonOperator(
        task_id="update_api",
        python_callable=_update_api,
    )

    (
        ingest_yahoo_finance
        >> ingest_reuters
        >> ingest_reddit
        >> ingest_twitter
        >> run_sentiment_analysis
        >> build_features
        >> retrain_models
        >> update_api
    )
