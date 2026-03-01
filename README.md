# Penguins ML Pipeline

Pipeline de Machine Learning orquestado con **Apache Airflow** para clasificar especies de pinguinos ([Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/)). Los modelos entrenados se sirven mediante una **API FastAPI** para inferencia en tiempo real.

## Arquitectura

```
                         docker-compose.yml
                                |
       +------------+-----------+-----------+------------+
       |            |                       |            |
  db-airflow    db-data (5433)        airflow (8080)   api (8989)
  PostgreSQL    PostgreSQL            Webserver +       FastAPI
  (metadata)    (penguins_db)         Scheduler
                     |                    |                |
                     |       penguins_pipeline DAG         |
                     |    [clear → load → preprocess       |
                     |              → train]               |
                     |                    |                |
                     |                    v                |
                     |              /models/ <-------------+
                     |           (volumen compartido)
                     v
               penguins_raw
               penguins_clean
```

**Flujo:**
1. El DAG de Airflow carga el dataset, lo preprocesa y lo almacena en PostgreSQL
2. Entrena 3 clasificadores y guarda los modelos como `.joblib` en un volumen compartido
3. La API carga los modelos y expone un endpoint de prediccion

---

## Requisitos

- [Docker](https://docs.docker.com/get-docker/) con Docker Compose
- [Task](https://taskfile.dev/) (opcional, simplifica los comandos)

No necesitas Python instalado localmente; todo corre dentro de los contenedores.

---

## Inicio rapido

```bash
task up              # construir y levantar todos los servicios
task dag:unpause     # activar el DAG
task dag:trigger     # ejecutar el pipeline de entrenamiento
```

O sin Task:

```bash
docker compose up -d --build
docker compose exec airflow-webserver airflow dags unpause penguins_pipeline
docker compose exec airflow-webserver airflow dags trigger penguins_pipeline
```

Una vez completado el DAG:

```bash
task models          # verificar que los modelos estan cargados en la API
task predict         # hacer una prediccion de prueba
```

---

## Servicios

| Servicio | Puerto | Descripcion |
|----------|--------|-------------|
| `airflow-webserver` | 8080 | UI de Airflow (usuario: `airflow`, password: `airflow`) |
| `airflow-scheduler` | - | Ejecuta los DAGs programados |
| `db-airflow` | - | PostgreSQL para metadata de Airflow |
| `db-data` | 5433 | PostgreSQL con los datos de penguins |
| `api` | 8989 | FastAPI para inferencia (docs en `/docs`) |

---

## DAG: penguins_pipeline

Pipeline de 4 tareas secuenciales:

| Tarea | Descripcion |
|-------|-------------|
| `clear_database` | Elimina tablas `penguins_raw` y `penguins_clean` |
| `load_raw_data` | Carga el dataset Palmer Penguins en `penguins_raw` |
| `preprocess_data` | Limpia datos, elimina nulos y guarda en `penguins_clean` |
| `train_model` | Entrena 3 clasificadores y los guarda en `/models/` |

**Preprocesamiento:**
- Features numericos: imputacion por mediana + escalado estandar
- Features categoricos: imputacion por moda + one-hot encoding
- Split: 80/20 estratificado

---

## Modelos

| Modelo | Algoritmo |
|--------|-----------|
| `logistic_regression` | LogisticRegression (max_iter=1000) |
| `random_forest` | RandomForestClassifier (n_estimators=100) |
| `gradient_boosting` | GradientBoostingClassifier (n_estimators=100) |

Los modelos se guardan como pipelines completos de scikit-learn (preprocesamiento + clasificador) en formato `.joblib`.

---

## API - Endpoints

### `GET /health`

```bash
curl http://localhost:8989/health
```
```json
{"status": "ok", "models_loaded": 3}
```

### `GET /models`

```bash
curl http://localhost:8989/models
```
```json
{"available": ["gradient_boosting", "logistic_regression", "random_forest"]}
```

### `POST /predict`

| Campo | Tipo | Ejemplo |
|-------|------|---------|
| `bill_length_mm` | float | 39.1 |
| `bill_depth_mm` | float | 18.7 |
| `flipper_length_mm` | float | 181.0 |
| `body_mass_g` | float | 3750.0 |
| `island` | string | `"Torgersen"`, `"Biscoe"`, `"Dream"` |
| `sex` | string | `"male"`, `"female"` |
| `model_name` | string | `"random_forest"` |

```bash
curl -X POST http://localhost:8989/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0,
    "island": "Torgersen",
    "sex": "male",
    "model_name": "random_forest"
  }'
```
```json
{"species": "Adelie", "model_used": "random_forest"}
```

### `POST /reload`

Recarga los modelos desde el volumen compartido sin reiniciar el servicio. Util despues de re-ejecutar el DAG de entrenamiento.

```bash
curl -X POST http://localhost:8989/reload
```
```json
{"status": "reloaded", "models_loaded": 3}
```

### Errores

| Codigo | Causa |
|--------|-------|
| 404 | `model_name` no existe |
| 422 | Campos faltantes o tipos invalidos |
| 503 | No hay modelos cargados |

---

## Comandos Task

### Lifecycle

| Comando | Descripcion |
|---------|-------------|
| `task up` | Construir y levantar todos los servicios |
| `task down` | Detener y eliminar servicios |
| `task restart` | Reiniciar servicios |
| `task ps` | Ver estado de los contenedores |
| `task clean` | Detener servicios y eliminar volumenes (reset completo) |

### Logs y Shell

| Comando | Descripcion |
|---------|-------------|
| `task logs` | Logs de todos los servicios |
| `task logs:api` | Logs de la API |
| `task logs:airflow` | Logs del webserver y scheduler |
| `task shell:api` | Shell en el contenedor de la API |
| `task shell:airflow` | Shell en el contenedor de Airflow |
| `task shell:db` | Consola psql en la base de datos |

### API

| Comando | Descripcion |
|---------|-------------|
| `task health` | Verificar estado de la API |
| `task models` | Listar modelos disponibles |
| `task reload` | Recargar modelos en la API |
| `task predict` | Prediccion de ejemplo (default: random_forest) |
| `task predict -- logistic_regression` | Prediccion con modelo especifico |

### Base de datos

| Comando | Descripcion |
|---------|-------------|
| `task db:tables` | Listar tablas en penguins_db |
| `task db:count` | Contar registros por tabla |
| `task db:preview` | Ver primeras 5 filas de penguins_clean |
| `task db:query -- 'SELECT ...'` | Ejecutar SQL personalizado |

### Airflow

| Comando | Descripcion |
|---------|-------------|
| `task dag:trigger` | Disparar el pipeline de entrenamiento |
| `task dag:status` | Ver ultimas ejecuciones del DAG |
| `task dag:unpause` | Activar el DAG |

---

## Estructura del proyecto

```
.
├── docker-compose.yml              # Orquestacion de servicios
├── Taskfile.yml                    # Task runner
├── airflow/
│   ├── Dockerfile                  # Imagen Airflow con dependencias ML
│   └── requirements.txt            # pandas, scikit-learn, palmerpenguins, etc.
├── dags/
│   └── penguins_pipeline.py        # DAG de entrenamiento
├── api/
│   ├── Dockerfile                  # Imagen API
│   ├── pyproject.toml              # Dependencias de la API
│   └── app/
│       ├── main.py                 # Endpoints FastAPI
│       └── schemas.py              # Modelos Pydantic
├── jupyter/
│   └── notebooks/
│       └── train_models.ipynb      # Notebook de experimentacion
└── models/                         # Volumen compartido (archivos .joblib)
```

---

## Tecnologias

- **Apache Airflow** - Orquestacion del pipeline ML
- **PostgreSQL** - Almacenamiento de datos y metadata
- **FastAPI** - API de inferencia
- **scikit-learn** - Entrenamiento de modelos
- **Docker Compose** - Orquestacion de servicios
- **Task** - Task runner
- **Palmer Penguins** - Dataset de clasificacion
