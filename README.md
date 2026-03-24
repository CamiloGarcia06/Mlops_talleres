# Taller MLflow — MLOps con Docker

Taller de MLOps que demuestra el ciclo de vida completo de un experimento de Machine Learning usando **MLflow**, **JupyterLab**, **MinIO**, **PostgreSQL** y **FastAPI** — todo orquestado con Docker Compose.

## Requisitos previos

- [Docker](https://docs.docker.com/get-docker/) y [Docker Compose](https://docs.docker.com/compose/install/) instalados
- Puertos disponibles: `5002`, `5434`, `5435`, `8001`, `8889`, `9002`, `9003`

## Inicio rapido

```bash
# 1. Clonar el repositorio
git clone <url-del-repo> && cd Mlops_talleres

# 2. Levantar todos los servicios
docker compose up --build -d

# 3. Esperar a que todos los servicios estén healthy (~1-2 min)
docker compose ps

# 4. Abrir JupyterLab y ejecutar el notebook
#    http://localhost:8889  (token: taller2026)

# 5. Ejecutar todas las celdas del notebook 01_experimentos_mlflow.ipynb

# 6. Probar la API de inferencia
#    http://localhost:8001/docs
```

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                        docker-compose                           │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │ mlflow-db│    │ data-db  │    │  minio   │                  │
│  │ postgres │    │ postgres │    │  S3-like  │                  │
│  │ :5434    │    │ :5435    │    │ :9002/:9003│                 │
│  └────┬─────┘    └────┬─────┘    └─────┬─────┘                 │
│       │               │           ┌────┴────┐                  │
│       │               │           │minio-setup│ (init bucket)  │
│       │               │           └────┬────┘                  │
│       └───────┬───────┘                │                       │
│               │                        │                       │
│         ┌─────┴────────────────────────┴──┐                    │
│         │        mlflow-server            │                    │
│         │        :5002                    │                    │
│         └──────┬──────────────┬───────────┘                    │
│                │              │                                 │
│      ┌─────────┴──┐    ┌─────┴──────┐                          │
│      │ jupyterlab │    │inference-api│                          │
│      │ :8889      │    │ :8001       │                          │
│      └────────────┘    └─────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

## Servicios

| Servicio | Contenedor | Puerto local | Descripcion |
|---|---|---|---|
| `mlflow-db` | `mlflow-db` | `5434` | PostgreSQL exclusivo para metadata de MLflow |
| `data-db` | `data-db` | `5435` | PostgreSQL exclusivo para datos del modelo |
| `minio` | `minio` | `9002` (API) / `9003` (Consola) | Object storage S3-compatible para artefactos |
| `minio-setup` | `minio-setup` | — | Tarea init: crea el bucket `mlflow-artifacts` |
| `mlflow` | `mlflow-server` | `5002` | MLflow Tracking Server |
| `jupyter` | `jupyterlab` | `8889` | JupyterLab con librerias de ML |
| `api` | `inference-api` | `8001` | FastAPI para inferencia en tiempo real |

## URLs de acceso

| Servicio | URL |
|---|---|
| MLflow UI | http://localhost:5002 |
| JupyterLab | http://localhost:8889 (token: `taller2026`) |
| MinIO Console | http://localhost:9003 (user: `minioadmin` / pass: `minioadmin123`) |
| API Swagger | http://localhost:8001/docs |

## Bases de datos

El taller usa **dos bases de datos PostgreSQL separadas**:

### mlflow-db (metadata de MLflow)
- **Base de datos:** `mlflow_metadata`
- **Usuario:** `mlflow` / `mlflow_secret`
- **Puerto externo:** `5434`
- Almacena: experimentos, runs, metricas, parametros

### data-db (datos del modelo)
- **Base de datos:** `ml_datasets`
- **Usuario:** `datauser` / `data_secret`
- **Puerto externo:** `5435`
- Tablas (creadas por `init-scripts/init_data_db.sql`):

| Tabla | Descripcion |
|---|---|
| `wine_raw` | Datos crudos del dataset Wine de sklearn |
| `wine_processed` | Features normalizados con StandardScaler, separados en train/test |
| `experiment_log` | Log de cada experimento con hiperparametros y metricas en formato JSONB |

## Notebook: 01_experimentos_mlflow.ipynb

El notebook ejecuta el flujo completo de experimentacion:

### Paso 1 — Carga de datos
Carga el dataset Wine de sklearn (178 muestras, 13 features, 3 clases) y lo guarda en la tabla `wine_raw` de `data-db`.

### Paso 2 — Procesamiento
Lee desde `wine_raw`, normaliza los features con `StandardScaler`, divide en train/test (80/20) y guarda en `wine_processed`.

### Paso 3 — Experimentacion (24 ejecuciones)
Entrena **24 modelos** variando hiperparametros en 6 familias:

| Modelo | Variaciones | Hiperparametros |
|---|---|---|
| DecisionTree | 4 | `max_depth`, `min_samples_split` |
| RandomForest | 4 | `n_estimators`, `max_depth` |
| GradientBoosting | 4 | `n_estimators`, `learning_rate` |
| SVM (SVC) | 4 | `C`, `kernel` |
| KNN | 4 | `n_neighbors`, `weights` |
| LogisticRegression | 4 | `C`, `solver` |

Cada ejecucion registra en MLflow:
- Hiperparametros del modelo (`mlflow.log_params`)
- Metricas: `accuracy` y `f1_macro` (`mlflow.log_metrics`)
- Modelo serializado con signature (`mlflow.sklearn.log_model`)
- Log en tabla `experiment_log` de `data-db`

### Paso 4 — Model Registry
Registra los 24 modelos en el **MLflow Model Registry** bajo el nombre `wine-classifier-production` y promueve el mejor (por accuracy) al stage **Production**.

## API de inferencia

La API (`api/main.py`) carga al iniciar todas las versiones registradas del modelo desde MLflow y expone:

| Endpoint | Metodo | Descripcion |
|---|---|---|
| `/health` | GET | Estado de la API y cantidad de modelos cargados |
| `/models` | GET | Lista los nombres de modelos disponibles |
| `/predict` | POST | Realiza prediccion con un modelo especifico |
| `/reload` | POST | Recarga modelos desde MLflow sin reiniciar |

### Ejemplo de prediccion

```bash
# Ver modelos disponibles
curl http://localhost:8001/models

# Predecir (reemplazar model_name con uno de los disponibles)
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "rf_100_depthNone",
    "alcohol": 13.0,
    "malic_acid": 2.0,
    "ash": 2.3,
    "alcalinity_of_ash": 20.0,
    "magnesium": 100.0,
    "total_phenols": 2.5,
    "flavanoids": 2.5,
    "nonflavanoid_phenols": 0.3,
    "proanthocyanins": 1.5,
    "color_intensity": 5.0,
    "hue": 1.0,
    "od280_od315_diluted_wines": 3.0,
    "proline": 1000.0
  }'
```

## Estructura del proyecto

```
.
├── docker-compose.yml                          # Orquestacion de los 7 servicios
├── mlflow/
│   └── Dockerfile                              # MLflow server con psycopg2 y boto3
├── jupyter/
│   ├── Dockerfile                              # JupyterLab con librerias de ML
│   ├── requirements.txt                        # Dependencias: mlflow, sklearn, sqlalchemy, etc.
│   └── notebooks/
│       └── 01_experimentos_mlflow.ipynb         # Notebook principal del taller
├── api/
│   ├── Dockerfile                              # FastAPI con uvicorn
│   ├── main.py                                 # Servidor de inferencia
│   └── requirements.txt                        # Dependencias: fastapi, mlflow, sklearn
├── init-scripts/
│   └── init_data_db.sql                        # DDL para tablas wine_raw, wine_processed, experiment_log
└── README.md
```

## Comandos utiles

```bash
# Levantar todo
docker compose up --build -d

# Ver estado de los servicios
docker compose ps

# Ver logs de un servicio
docker compose logs -f mlflow
docker compose logs -f jupyter
docker compose logs -f api

# Reconstruir un servicio individual
docker compose up --build -d jupyter

# Apagar todo
docker compose down

# Apagar y eliminar volumenes (reset completo de datos)
docker compose down -v
```

## Cadena de dependencias

```
mlflow-db ─────┐
               ├──→ mlflow-server ──┬──→ jupyterlab
minio ──→ minio-setup ─┘            └──→ inference-api
data-db ────────────────────────────────→ jupyterlab
```

Los servicios usan healthchecks para garantizar que las dependencias estan listas antes de arrancar. El orden completo de inicio es:

1. `mlflow-db`, `data-db` y `minio` arrancan en paralelo
2. `minio-setup` espera a que `minio` este healthy, crea el bucket y termina
3. `mlflow-server` espera a `mlflow-db` (healthy) y `minio-setup` (completed)
4. `jupyterlab` espera a `mlflow-server` y `data-db`
5. `inference-api` espera a `mlflow-server`

## Variables de entorno

| Variable | Valor | Usado por |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://mlflow-server:5000` | jupyter, api |
| `MLFLOW_S3_ENDPOINT_URL` | `http://minio:9000` | mlflow, jupyter, api |
| `AWS_ACCESS_KEY_ID` | `minioadmin` | mlflow, jupyter, api |
| `AWS_SECRET_ACCESS_KEY` | `minioadmin123` | mlflow, jupyter, api |
| `DATA_DB_URI` | `postgresql://datauser:data_secret@data-db:5432/ml_datasets` | jupyter |
| `JUPYTER_TOKEN` | `taller2026` | jupyter |
