FROM apache/airflow:2.8.0-python3.10

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Project Python deps the DAG's PythonOperators need at runtime.
# Pinned ranges match the project's requirements.txt.
RUN pip install --no-cache-dir \
    yfinance>=0.2.40 \
    feedparser>=6.0.10 \
    praw>=7.7.1 \
    tweepy>=4.14.0 \
    pandas>=2.1.0 \
    pyarrow>=14.0.0 \
    numpy>=1.26.0 \
    nltk>=3.8.1 \
    scikit-learn>=1.3.0 \
    python-dotenv>=1.0.0 \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.0/constraints-3.10.txt" \
    || pip install --no-cache-dir \
        yfinance feedparser praw tweepy pandas pyarrow numpy \
        nltk scikit-learn python-dotenv

# Torch + transformers are large and only needed for the (Monday-only) retrain
# task. Install CPU build to keep the image lean.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.2

RUN pip install --no-cache-dir \
    transformers>=4.36.0 \
    mlflow>=2.9.0
