# MLOps Taller — MLflow

Taller de MLOps que demuestra el ciclo de vida completo de un experimento de ML usando MLflow, JupyterLab, MinIO, PostgreSQL y FastAPI — todo orquestado con Docker Compose.

## Comandos

### Levantar el stack completo
```bash
docker compose up --build
```

### Levantar en segundo plano
```bash
docker compose up -d --build
```

### Apagar y eliminar volúmenes (reset completo)
```bash
docker compose down -v
```

### Reconstruir un servicio individual
```bash
docker compose up --build jupyter
docker compose up --build api
```

### Ver logs de un servicio
```bash
docker compose logs -f mlflow
docker compose logs -f api
```

### Entorno Python local (uv)
```bash
uv sync           # instala dependencias desde pyproject.toml
uv run python     # ejecuta python en el venv
```

## Arquitectura

### Servicios (docker-compose.yml)

| Servicio | Puerto | Propósito |
|---|---|---|
| `mlflow-server` | 5001 | MLflow Tracking Server + proxy de artefactos |
| `jupyterlab` | 8888 | JupyterLab (token: `taller2026`) |
| `inference-api` | 8000 | Servidor de inferencia FastAPI |
| `minio` | 9000/9001 | Almacenamiento de artefactos compatible con S3 (consola: 9001) |
| `mlflow-db` | 5432 | PostgreSQL para metadata de MLflow |
| `data-db` | 5433 | PostgreSQL para el dataset Wine |

### Cadena de dependencias de arranque
`mlflow-db` + `data-db` + `minio` → `minio-setup` (crea el bucket) → `mlflow-server` → `jupyter` + `api`

### Dos bases de datos separadas
- **mlflow-db** (`mlflow_metadata`): almacena los runs, métricas y parámetros de MLflow. Usuario: `mlflow` / `mlflow_secret`.
- **data-db** (`ml_datasets`): almacena el dataset Wine. Usuario: `datauser` / `data_secret`. Tablas inicializadas en `init-scripts/init_data_db.sql`: `wine_raw`, `wine_processed`, `experiment_log`.

### Almacenamiento de artefactos
MLflow está configurado con `--serve-artifacts`, lo que significa que actúa como proxy de acceso a los artefactos (puerto 5000) sin requerir acceso directo a MinIO. Los artefactos se guardan en el bucket `mlflow-artifacts` de MinIO.

### Dockerfiles
Tanto `api/Dockerfile` como `jupyter/Dockerfile` usan `uv` (desde `ghcr.io/astral-sh/uv`) para instalar dependencias de forma más rápida que pip.

### Flujo del notebook (`jupyter/notebooks/01_experimentos_mlflow.ipynb`)
El notebook cubre el flujo completo de ML:
1. Cargar dataset Wine (sklearn) → guardar en `wine_raw` en data-db
2. Normalizar features → guardar en `wine_processed`
3. Entrenar 3 modelos simples (DecisionTree, RandomForest, LogisticRegression) y registrar métricas en MLflow
4. Registrar los 3 modelos en el MLflow Model Registry y promover el mejor a **Production**

### Servicio de inferencia FastAPI (`api/main.py`)
Al iniciar, carga todas las versiones registradas en el MLflow Model Registry. Expone endpoints para predecir con cualquier modelo por nombre y recargar modelos sin reiniciar. Swagger UI disponible en `http://localhost:8000/docs`.

## Variables de entorno clave (definidas en docker-compose.yml)

| Variable | Valor | Usado por |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://mlflow-server:5000` | jupyter, api |
| `MLFLOW_S3_ENDPOINT_URL` | `http://minio:9000` | mlflow-server, jupyter, api |
| `AWS_ACCESS_KEY_ID` | `minioadmin` | mlflow-server, jupyter, api |
| `AWS_SECRET_ACCESS_KEY` | `minioadmin123` | mlflow-server, jupyter, api |
| `DATA_DB_URI` | `postgresql://datauser:data_secret@data-db:5432/ml_datasets` | jupyter |
| `JUPYTER_TOKEN` | `taller2026` | jupyter |
