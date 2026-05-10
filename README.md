# Real-Time Market Movement Prediction System

End-to-end pipeline for predicting next-period **price direction** (up / down)
of equities using OHLCV data combined with news + social-media sentiment, fed
into sequential deep-learning models (RNN / LSTM / GRU). The system covers
ingestion, feature engineering, training (with MLflow tracking), an
Airflow-orchestrated retraining schedule, a FastAPI prediction service, and a
minimal dark-themed dashboard.

## Architecture

```
+------------------+     +-----------------+     +-------------------+
|  Yahoo Finance   |     | Reuters RSS     |     | Reddit / Twitter  |
+--------+---------+     +--------+--------+     +---------+---------+
         |                        |                        |
         v                        v                        v
+----------------------------------------------------------------+
|                    src/ingestion/*  (DVC stage)               |
+----------------------------------------------------------------+
                                 |
                                 v
+----------------------------------------------------------------+
|        src/sentiment/  (FinBERT + VADER)  →  data/processed   |
+----------------------------------------------------------------+
                                 |
                                 v
+----------------------------------------------------------------+
|     src/features/time_series  →  sliding windows in data/features
+----------------------------------------------------------------+
                                 |
                                 v
+----------------------------------------------------------------+
|     src/models/{rnn,lstm,gru} + train.py  →  MLflow runs      |
+----------------------------------------------------------------+
                                 |
                                 v
+----------------------+     +-----------------------------+
| src/api  (FastAPI)   | <-- |  mlruns/  (registry)        |
+----------+-----------+     +-----------------------------+
           |
           v
+----------------------+
|  frontend/  (HTML)   |
+----------------------+

Airflow DAG (dags/market_pipeline_dag.py) runs every 4h end-to-end.
DVC (dvc.yaml) versions raw, processed, and feature data.
```

## Project layout

```
market-prediction/
├── .github/workflows/ci.yml    # lint + tests + docker smoke test
├── dags/                       # Airflow DAGs
├── data/                       # raw / processed / features (DVC-tracked)
├── src/
│   ├── ingestion/              # yahoo, reuters, reddit, twitter
│   ├── sentiment/              # finbert, vader
│   ├── features/               # time_series window builder
│   ├── models/                 # rnn, lstm, gru, train
│   ├── evaluation/             # metrics + plots
│   └── api/                    # FastAPI service
├── frontend/                   # static dashboard
├── notebooks/eda.ipynb         # exploratory analysis
├── docker-compose.yml          # api + mlflow + airflow
├── Dockerfile                  # API image
├── dvc.yaml                    # pipeline stages
└── requirements.txt
```

## Setup

```bash
git clone <your-fork-url> market-prediction
cd market-prediction
cp .env.example .env             # fill in Reddit / Twitter credentials
docker compose up --build        # api → :8000, mlflow → :5000, airflow → :8080
```

For local (non-Docker) development:

```bash
python -m venv .venv
source .venv/bin/activate        # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running individual components

| Component       | Command                                               |
|-----------------|-------------------------------------------------------|
| Yahoo ingest    | `python -m src.ingestion.yahoo_finance`               |
| Reuters ingest  | `python -m src.ingestion.reuters_rss`                 |
| Reddit ingest   | `python -m src.ingestion.reddit_scraper`              |
| Twitter ingest  | `python -m src.ingestion.twitter_scraper`             |
| Build features  | `python -m src.features.time_series`                  |
| Train models    | `python -m src.models.train`                          |
| API (local)     | `uvicorn src.api.main:app --reload --port 8000`       |
| Frontend        | open `frontend/index.html` in a browser               |
| DVC pipeline    | `dvc repro`                                           |

## API endpoints

| Method | Path                  | Description                                   |
|--------|-----------------------|-----------------------------------------------|
| GET    | `/health`             | Liveness probe                                |
| POST   | `/predict`            | `{ticker, window_data}` → direction + confidence |
| GET    | `/sentiment/{ticker}` | Latest sentiment distribution                 |
| GET    | `/models`             | Available trained models + metadata           |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"ticker":"AAPL","window_data":[[0,0,0,0,0,0,0,0,0]]}'
```

## MLflow

```
http://localhost:5000
```

Local backend store lives in `./mlruns`. Each model gets its own run inside the
`market_prediction` experiment.

## Airflow

```
http://localhost:8080
```

Default credentials are printed by `airflow standalone` on first boot
(`standalone_admin_password.txt`). The DAG `market_prediction_pipeline` runs
every 4 hours; retraining is gated to Mondays.

## Environment variables

See `.env.example`:

- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`
- `TWITTER_BEARER_TOKEN`
- `MLFLOW_TRACKING_URI`
- `AIRFLOW_UID`

## Testing

```bash
pytest tests/
flake8 src/ --max-line-length=120
```
