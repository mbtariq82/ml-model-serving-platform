# ML Model Serving Platform

Training implementation of a model-serving platform. It includes a model
registry, deterministic local inference runtime, A/B experiments, monitoring,
drift detection, and retraining triggers.

## Features

- Register pickle or ONNX model artifacts with tags and runtime parameters.
- Manage model stages through `dev -> staging -> production -> archived`.
- Warm models on demand and serve single or batch prediction requests.
- Split experiment traffic between versions, collect ground truth, evaluate
  absolute error, and auto-promote significant winners.
- Track prediction distribution, detect mean-shift drift or confidence
  degradation, and queue retraining.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
uvicorn ml_model_serving_platform.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API docs.

## Test

```bash
pytest
```

## Key endpoints

- `POST /models`
- `POST /models/{model_name}/versions/{version}/stage`
- `POST /models/{model_name}/predict`
- `POST /models/{model_name}/batch-predict`
- `POST /experiments`
- `POST /experiments/{experiment_id}/predict`
- `GET /experiments/{experiment_id}/evaluation`
- `POST /experiments/{experiment_id}/auto-promote`
- `POST /models/{model_name}/versions/{version}/drift`
- `POST /models/{model_name}/versions/{version}/retraining`
