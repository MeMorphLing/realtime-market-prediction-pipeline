# Deployment — Inference container

Self-contained image for serving the trained RNN / LSTM / GRU models.

```
deployment/
├── Dockerfile
├── main.py                 # uvicorn entry point
├── requirements.txt
├── app/
│   ├── api.py              # FastAPI app + routes
│   ├── inference.py        # checkpoint loading + predict()
│   ├── models.py           # model class definitions
│   └── schemas.py          # Pydantic request/response
└── model/
    ├── rnn_best.pt
    ├── lstm_best.pt
    └── gru_best.pt
```

## Build & run locally

```bash
cd deployment
docker build -t market-prediction:latest .
docker run --rm -p 8000:8000 market-prediction:latest
```

## Endpoints

```
GET  /health            → {"status":"ok","models":[...]}
GET  /models            → [{"name":"rnn","available":true}, ...]
POST /predict           → {ticker, direction, confidence, model}
GET  /docs              → Swagger UI
```

### Predict request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "window_data": [[0,0,0,0,0,0,0,0,0], ...],
    "model": "lstm"
  }'
```

`window_data` must be shape `(window_size, 9)` and **already normalized** the
same way the training pipeline normalizes (per-ticker z-score over OHLCV +
sentiment features). The trained models expect 20-step windows.

## Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | `/app/model` | Where checkpoints live |
| `DEFAULT_MODEL` | `lstm` | Used when request doesn't specify `model` |
| `MODEL_INPUT_SIZE` | `9` | Number of features per timestep |

## CI/CD

`.github/workflows/cd.yml` builds and pushes this image to GitHub Container
Registry on every push to `main` that touches `deployment/**`.
