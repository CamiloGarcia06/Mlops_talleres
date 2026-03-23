# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLOps workshop (taller) demonstrating a full ML experiment lifecycle using MLflow, JupyterLab, MinIO, PostgreSQL, and FastAPI — all orchestrated with Docker Compose.

## Commands

### Start the full stack
```bash
docker compose up --build
```

### Start in background
```bash
docker compose up -d --build
```

### Stop and remove volumes (full reset)
```bash
docker compose down -v
```

### Rebuild a single service
```bash
docker compose up --build jupyter
docker compose up --build api
```

### View logs for a service
```bash
docker compose logs -f mlflow
docker compose logs -f api
```

### Local Python environment (uv)
```bash
uv sync           # install dependencies from pyproject.toml
uv run python     # run python in the venv
```

## Architecture

### Services (docker-compose.yml)

| Service | Port | Purpose |
|---|---|---|
| `mlflow-server` | 5000 | MLflow Tracking Server + artifact proxy |
| `jupyterlab` | 8888 | JupyterLab (token: `taller2026`) |
| `inference-api` | 8000 | FastAPI inference server |
| `minio` | 9000/9001 | S3-compatible artifact storage (console: 9001) |
| `mlflow-db` | 5432 | PostgreSQL for MLflow metadata |
| `data-db` | 5433 | PostgreSQL for Wine dataset |

### Startup dependency chain
`mlflow-db` + `data-db` + `minio` → `minio-setup` (creates bucket) → `mlflow-server` → `jupyter` + `api`

### Two separate databases
- **mlflow-db** (`mlflow_metadata`): stores MLflow experiment runs, metrics, parameters. User: `mlflow` / `mlflow_secret`.
- **data-db** (`ml_datasets`): stores the Wine dataset. User: `datauser` / `data_secret`. Tables initialized via `init-scripts/init_data_db.sql`: `wine_raw`, `wine_processed`, `experiment_log`.

### Artifact storage
MLflow is configured with `--serve-artifacts`, meaning it proxies all artifact access through itself (port 5000) rather than requiring direct MinIO access. Artifacts are stored in the `mlflow-artifacts` S3 bucket on MinIO.

### Dockerfiles
Both `api/Dockerfile` and `jupyter/Dockerfile` use `uv` (from `ghcr.io/astral-sh/uv`) for fast dependency installation instead of pip.

### Notebook workflow (`jupyter/notebooks/01_experimentos_mlflow.ipynb`)
The single notebook covers the full ML workflow:
1. Load Wine dataset (sklearn) → save to `wine_raw` in data-db
2. Normalize features → save to `wine_processed`
3. Run 23 TensorFlow/Keras experiments with varied architectures and hyperparameters
4. Log each run to MLflow (metrics, params, model, artifacts)
5. Promote the best model to **Production** stage in MLflow Model Registry

### FastAPI inference service (`api/main.py`)
Loads the Production-staged model from MLflow Model Registry at startup and exposes a prediction endpoint. Swagger UI available at `http://localhost:8000/docs`.

## Key Environment Variables (set in docker-compose.yml)

| Variable | Value | Used by |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://mlflow-server:5000` | jupyter, api |
| `MLFLOW_S3_ENDPOINT_URL` | `http://minio:9000` | mlflow-server, jupyter, api |
| `AWS_ACCESS_KEY_ID` | `minioadmin` | mlflow-server, jupyter, api |
| `AWS_SECRET_ACCESS_KEY` | `minioadmin123` | mlflow-server, jupyter, api |
| `DATA_DB_URI` | `postgresql://datauser:data_secret@data-db:5432/ml_datasets` | jupyter |
| `JUPYTER_TOKEN` | `taller2026` | jupyter |
