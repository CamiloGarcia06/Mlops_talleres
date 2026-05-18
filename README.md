# MLOps Diabetes — Proyecto 2

Plataforma MLOps end-to-end sobre Kubernetes para predecir reingreso hospitalario de pacientes diabéticos usando el dataset **Diabetes 130-US Hospitals (1999-2008)** (~101k registros).

El sistema cubre el ciclo completo: ingesta incremental por lotes → almacenamiento crudo → preprocesamiento → entrenamiento + experimentación → registro y promoción automática del modelo → servicio de inferencia → UI → observabilidad → pruebas de carga.

## Tabla de contenidos

- [Arquitectura](#arquitectura)
- [Componentes](#componentes)
- [Stack técnico](#stack-técnico)
- [Quickstart desde cero](#quickstart-desde-cero) — guía completa para alguien que no es el desarrollador
- [Flujo del DAG](#flujo-del-dag)
- [API de inferencia](#api-de-inferencia)
- [Modelo de datos](#modelo-de-datos)
- [Observabilidad y pruebas de carga](#observabilidad-y-pruebas-de-carga)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Variables de configuración](#variables-de-configuración)
- [Operación local](#operación-local)
- [Decisiones técnicas](#decisiones-técnicas)
- [Solución de problemas](#solución-de-problemas)

## Arquitectura

```mermaid
flowchart LR
    subgraph K8s["Kubernetes (namespace mlops)"]
        subgraph Training["Fase de entrenamiento"]
            CSV[(Diabetes.csv<br/>101k filas)]
            AF["Airflow<br/>(LocalExecutor)"]
            PG[("PostgreSQL<br/>raw / clean / inference")]
            ML["MLflow Server"]
            MN["MinIO<br/>(artifact store)"]
            CSV --> AF
            AF -->|"15k filas/run"| PG
            AF -->|"params, metrics,<br/>model + signature"| ML
            ML -.->|"metadata"| PG
            ML -.->|"artifacts"| MN
        end
        subgraph Inference["Fase de inferencia"]
            API["FastAPI<br/>/predict /metrics"]
            UI["Streamlit"]
            LC["Locust<br/>(load testing)"]
            UI -->|"HTTP POST"| API
            LC -->|"carga concurrente"| API
            API -->|"alias champion"| ML
            API -->|"log inferencia"| PG
        end
        subgraph Obs["Observabilidad"]
            PR["Prometheus"]
            GR["Grafana"]
            PR -->|"scrape /metrics"| API
            GR -->|"query"| PR
        end
    end
```

**Dos fases claramente separadas:**

- **Entrenamiento (offline):** Airflow ejecuta un DAG diario que carga 15k filas nuevas, preprocesa todo el acumulado, entrena LR + RF, registra ambos en MLflow y promueve automáticamente al ganador por F1.
- **Inferencia (online):** la API resuelve el modelo `champion` en MLflow al arrancar (con cache de 5 min y recarga manual vía `POST /reload-model`), responde predicciones a la UI y registra cada llamada en `inference.predictions`.

## Componentes

| Componente | Tecnología | Imagen DockerHub | Propósito |
|---|---|---|---|
| Orquestador | Apache Airflow 3.1.8 | `dandiazc/mlops-airflow:v1.2.0` | DAG de 9 tareas; embebe `pipeline/` y el CSV |
| BD relacional | PostgreSQL 16 | `postgres:16-alpine` (oficial) | Schemas `raw`, `clean`, `inference` + DB `mlflow`, `airflow` |
| Artifact store | MinIO | `quay.io/minio/minio` (oficial) | Bucket `mlflow-artifacts` |
| Registro ML | MLflow 2.13.0 | `dandiazc/mlops-mlflow:v0.1.0` | Backend PG + artifacts S3-compatible |
| API inferencia | FastAPI + uvicorn | `dandiazc/mlops-api:v0.3.0` | `/health /predict /model-info /metrics /reload-model` |
| UI | Streamlit | `dandiazc/mlops-ui:v0.1.0` | Formulario clínico → POST `/predict` |
| Pruebas de carga | Locust (master+workers) | `locustio/locust` (oficial) | Escenario contra `/predict` |
| Métricas | Prometheus | `prom/prometheus` (oficial) | Scrape vía anotaciones del pod |
| Dashboards | Grafana | `grafana/grafana` (oficial) | 1 dashboard con latencias p50/p95/p99 + CPU/mem |

## Stack técnico

| Capa | Versiones |
|---|---|
| Runtime | Python 3.11 (API/UI), Python 3.12 (Airflow base) |
| ML | scikit-learn 1.4.2, pandas 2.x |
| Models | LogisticRegression + RandomForestClassifier (ambos con `class_weight=balanced`) |
| Persistencia | PostgreSQL 16 (StatefulSet + PVC) |
| Object store | MinIO (StatefulSet + PVC) |
| Despliegue | kustomize para todo excepto Airflow (Helm chart oficial `apache-airflow/airflow`) |

## Quickstart desde cero

Esta sección está pensada para alguien que **nunca ha visto el repo** y quiere levantar todo el sistema en su máquina.

### Prerrequisitos

| Software | Versión mínima | Para qué |
|---|---|---|
| Docker Desktop | 4.x | Construir imágenes + correr Kubernetes local |
| Kubernetes local | cualquiera | `kind`, `minikube`, `microk8s` o Docker Desktop K8s. Recomendado: Docker Desktop con K8s activado |
| `kubectl` | 1.27+ | Cliente de Kubernetes |
| `helm` | 3.x | Para instalar Airflow |
| Cuenta DockerHub (opcional) | — | Solo si vas a construir y publicar imágenes propias |
| PowerShell 5.1+ o Bash | — | Para correr los scripts del repo |

Verificación rápida:

```powershell
docker info     # debe responder sin error
kubectl version --client
helm version
kubectl get nodes   # debe listar al menos un nodo
```

### Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/CamiloGarcia06/Mlops_talleres.git
cd Mlops_talleres
git checkout DanielDiaz/proyecto2
```

### Paso 2 — (Opcional) Construir imágenes propias

Si solo quieres consumir las imágenes publicadas en DockerHub, **salta este paso**. Las imágenes referenciadas en los manifiestos ya son públicas.

Si quieres construir las tuyas, **sustituye `dandiazc` por tu usuario** en los manifiestos (`k8s/*/deployment.yaml`, `airflow/values/values-local.yaml`) y luego:

```bash
# Build context = repo root para todas
docker build -f docker/api/Dockerfile     -t TU_USUARIO/mlops-api:v0.3.0 .
docker build -f docker/ui/Dockerfile      -t TU_USUARIO/mlops-ui:v0.1.0 .
docker build -f docker/mlflow/Dockerfile  -t TU_USUARIO/mlops-mlflow:v0.1.0 .
docker build -f airflow/Dockerfile        -t TU_USUARIO/mlops-airflow:v1.2.0 .

docker push TU_USUARIO/mlops-api:v0.3.0
docker push TU_USUARIO/mlops-ui:v0.1.0
docker push TU_USUARIO/mlops-mlflow:v0.1.0
docker push TU_USUARIO/mlops-airflow:v1.2.0
```

### Paso 3 — Desplegar fundaciones (Postgres + MinIO)

```bash
kubectl apply -k k8s/foundations
kubectl -n mlops rollout status statefulset/postgres
kubectl -n mlops rollout status statefulset/minio
kubectl -n mlops logs job/minio-bootstrap   # confirma "bucket created: mlflow-artifacts"
```

### Paso 4 — Desplegar MLflow

```bash
kubectl apply -k k8s/mlflow
kubectl -n mlops rollout status deployment/mlflow
```

### Paso 5 — Desplegar Airflow

```bash
helm repo add apache-airflow https://airflow.apache.org
helm repo update
helm upgrade --install airflow apache-airflow/airflow \
  -n mlops -f airflow/values/values-local.yaml
```

Espera ~2 minutos a que estén `Running` todos los pods (`scheduler`, `api-server`, `dag-processor`, `triggerer`):

```bash
kubectl -n mlops get pods -l release=airflow -w
```

### Paso 6 — Desplegar API, UI y observabilidad

```bash
kubectl apply -k k8s/api
kubectl apply -k k8s/ui
kubectl apply -k k8s/prometheus
kubectl apply -k k8s/grafana
kubectl apply -k k8s/locust
```

> **Nota:** la API arrancará en `CrashLoopBackOff` o fallará el readiness probe hasta que el DAG haya promovido un modelo `champion`. Es esperado en esta etapa.

### Paso 7 — Abrir todos los port-forwards

En PowerShell (Windows):

```powershell
.\scripts\port-forward-all.ps1
```

Esto expone localmente todos los servicios. Quick links:

| Servicio | URL | Credenciales |
|---|---|---|
| Airflow UI | http://localhost:8080 | `admin / admin` |
| MLflow UI | http://localhost:5000 | — |
| MinIO Console | http://localhost:9001 | `minioadmin / minioadmin123` |
| API (docs Swagger) | http://localhost:8000/docs | — |
| Streamlit UI | http://localhost:8501 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | `admin / mlops2026` |
| Locust UI | http://localhost:8089 | — |
| PostgreSQL | `localhost:5432` | `mlops_user / mlops_pass_2026` (db: `mlops`) |

### Paso 8 — Disparar el DAG por primera vez

1. Entra a Airflow UI: http://localhost:8080
2. Busca el DAG `diabetes_mlops_pipeline` y enciéndelo (toggle)
3. Click en **Trigger DAG** (▶️)
4. Espera 2-3 minutos a que todas las tareas estén en verde

Cada ejecución carga los siguientes 15k registros del CSV. Para procesar los 101k completos, dispara el DAG **7 veces** o deja que el schedule diario haga su trabajo.

### Paso 9 — Validar end-to-end

Tras la primera ejecución exitosa del DAG:

```bash
# Modelo registrado y promovido
curl http://localhost:5000/api/2.0/mlflow/registered-models/get?name=diabetes-classifier

# La API ya debería estar Healthy
curl http://localhost:8000/health
curl http://localhost:8000/model-info

# Hacer una predicción desde la UI
# Abre http://localhost:8501, click "Cargar valores de ejemplo (130-US)", luego "Predecir"
```

### Paso 10 — Probar la carga

1. Abre Locust: http://localhost:8089
2. Configura `Number of users = 50`, `Spawn rate = 5`, `Host = http://api.mlops.svc.cluster.local:8000`
3. Start
4. En otra pestaña abre Grafana (http://localhost:3000) y observa las gráficas de latencia y rate

## Flujo del DAG

```mermaid
flowchart LR
    A[t_migrate] --> B[t_check_source]
    B --> C[t_load_batch<br/>15k filas]
    C --> D[t_quality]
    D --> E[t_preprocess]
    E --> F[t_split<br/>70/15/15]
    F --> G[t_train<br/>LR + RF]
    G --> H[t_compare<br/>vs champion]
    H --> I[t_promote<br/>set alias]
```

| Tarea | Módulo | Qué hace | Idempotencia |
|---|---|---|---|
| `t_migrate` | `pipeline/db/migrations.py` | `CREATE SCHEMA/TABLE IF NOT EXISTS` para raw/clean/inference | ✓ (IF NOT EXISTS) |
| `t_check_source` | `pipeline/ingest.py:check_source` | Valida que el CSV exista en `/opt/airflow/data/Diabetes.csv` | ✓ |
| `t_load_batch` | `pipeline/ingest.py:load_batch` | Cuenta filas en raw → usa como offset → carga las siguientes 15k | ✓ (`row_hash UNIQUE`) |
| `t_quality` | `pipeline/quality.py` | Valida columnas y tipos | ✓ |
| `t_preprocess` | `pipeline/preprocess.py` | One-hot encoding (sobre **todo** lo acumulado), imputación, persiste como JSONB | ✓ (upsert por `row_hash`) |
| `t_split` | `pipeline/split.py` | Split estratificado 70/15/15 sobre `clean.diabetes_clean` | ✓ (re-asigna `split`) |
| `t_train` | `pipeline/train.py` | Entrena LR + RF con `class_weight=balanced`, logea params/metrics/model con signature en MLflow | ✓ (cada run es nuevo) |
| `t_compare` | `pipeline/promote.py:compare` | Compara F1 del candidato vs champion actual | ✓ |
| `t_promote` | `pipeline/promote.py:promote` | Si gana, mueve alias `champion` a la versión nueva | ✓ |

**Lógica del cursor de batches** ([pipeline/ingest.py](pipeline/ingest.py)): cada ejecución cuenta cuántas filas existen ya en `raw.diabetes_raw` con el `source_file` actual, usa ese número como `skiprows` y lee las siguientes 15k. Cuando el CSV se agota (~7ma ejecución), `load_batch` reporta `inserted=0` sin fallar.

**Promoción automática** ([pipeline/promote.py:51](pipeline/promote.py#L51)): si no existe champion → se promueve el candidato; si existe → solo se promueve si `candidate.F1 > champion.F1`.

## API de inferencia

Implementada con FastAPI ([api/main.py](api/main.py)).

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/health` | Liveness/readiness simple |
| GET | `/model-info` | Nombre, versión, alias y `loaded_at` del modelo en caché |
| POST | `/predict` | Recibe `{"features": {...}}`, retorna `{prediction, score, model_name, model_version, model_alias, request_id, processing_time_ms}` |
| POST | `/reload-model` | Fuerza recarga del champion desde MLflow (útil tras una promoción) |
| GET | `/metrics` | Métricas Prometheus (HTTP + custom de inferencia) |

**Estrategia de carga del modelo:**

1. **Pre-carga en startup** vía `lifespan` de FastAPI ([api/main.py:30-38](api/main.py#L30-L38)) → evita cold-start en la primera petición.
2. **Cache en memoria con TTL** ([api/model_loader.py](api/model_loader.py)) — default 300s configurable vía `MODEL_CACHE_TTL_SECONDS`. Tras el TTL, la siguiente petición refresca el modelo desde MLflow.
3. **Recarga manual** vía `POST /reload-model` para propagación inmediata tras una promoción.

**Alineación dinámica de features** ([api/main.py:85-105](api/main.py#L85-L105)): la UI envía 141 features one-hot, pero el modelo puede esperar menos (si fue entrenado con menos batches del CSV). `_align_features()` lee el `signature` que MLflow guarda con el modelo, reindexea el DataFrame al esquema esperado (rellena con 0 lo faltante, descarta lo desconocido) y predice sin romper. Fallback: si no hay signature, intenta `feature_names_in_` del sklearn nativo.

**Persistencia de inferencias** ([api/db.py](api/db.py)): cada `/predict` inserta una fila en `inference.predictions` con:

```
request_id (UUID) | created_at | input_payload (JSONB) | prediction
                    score | model_name | model_version | latency_ms
```

## Modelo de datos

Tres schemas en PostgreSQL con responsabilidades separadas ([pipeline/db/migrations.py](pipeline/db/migrations.py)):

### `raw.diabetes_raw`

| Columna | Tipo | Descripción |
|---|---|---|
| `row_hash` | `TEXT PK` | SHA-1 de la fila — garantiza idempotencia |
| `batch_id` | `TEXT` | UUID del batch que la cargó |
| `load_timestamp` | `TIMESTAMPTZ` | Cuándo se cargó |
| `source_file` | `TEXT` | Path del CSV origen |
| `status` | `TEXT` | `loaded` por defecto |
| `payload` | `JSONB` | La fila original sin transformar |

### `clean.diabetes_clean`

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | `BIGSERIAL PK` | — |
| `row_hash` | `TEXT FK → raw` | Trazabilidad al dato crudo |
| `batch_id` | `TEXT` | — |
| `processed_at` | `TIMESTAMPTZ` | — |
| `split` | `TEXT` | `train` / `val` / `test` |
| `features` | `JSONB` | One-hot + numéricas imputadas |
| `target` | `INT` | Binarizado: `1` = readmitido en <30 días |

### `inference.predictions`

| Columna | Tipo | Descripción |
|---|---|---|
| `request_id` | `UUID PK` | Generado por la API |
| `created_at` | `TIMESTAMPTZ` | — |
| `input_payload` | `JSONB` | Features que llegaron al endpoint |
| `prediction` | `INT` | 0 o 1 |
| `score` | `DOUBLE` | Probabilidad clase positiva (si `predict_proba` existe) |
| `model_name` | `TEXT` | `diabetes-classifier` |
| `model_version` | `TEXT` | Versión MLflow que sirvió la predicción |
| `latency_ms` | `DOUBLE` | Tiempo de procesamiento |

**Por qué JSONB para `features` y `payload`:** el esquema del dataset puede variar (Pima 8 features vs. 130-US 50+ features) y `pd.get_dummies` produce un número variable de columnas según los valores categóricos presentes. JSONB tolera esa variabilidad sin DDL.

## Observabilidad y pruebas de carga

### Métricas expuestas por la API

Vía `prometheus-fastapi-instrumentator` ([api/main.py:43-45](api/main.py#L43-L45)) + custom counters ([api/metrics.py](api/metrics.py)):

| Métrica | Tipo | Etiquetas |
|---|---|---|
| `http_requests_total` | Counter | `method`, `handler`, `status` |
| `http_request_duration_seconds` | Histogram | `method`, `handler` |
| `predictions_total` | Counter | `prediction` (valor 0 o 1) |
| `inference_latency_seconds` | Histogram | — |
| `model_info` | Gauge | `name`, `version`, `alias` |

### Scrape de Prometheus

Configurado por anotaciones en el pod ([k8s/api/deployment.yaml:17-20](k8s/api/deployment.yaml#L17-L20)):

```yaml
prometheus.io/scrape: "true"
prometheus.io/port: "8000"
prometheus.io/path: "/metrics"
```

### Dashboard de Grafana

Source of truth: [observability/dashboards/api.json](observability/dashboards/api.json) (sincronizado a `k8s/grafana/configmap-dashboard-json.yaml`).

Paneles incluidos:

- Requests totales y RPS
- Latencias p50 / p95 / p99
- Tasa de errores (4xx + 5xx / total)
- Predicciones por clase
- CPU y memoria del pod de la API

### Locust

Workers desplegados en el cluster ([k8s/locust/](k8s/locust/)). El payload del escenario está en [k8s/locust/configmap.yaml](k8s/locust/configmap.yaml) (141 features one-hot, valores representativos).

> ⚠️ **Mantener sincronizados:** `k8s/locust/configmap.yaml` (cluster) y `loadtest/locustfile.py` (run local) contienen el mismo payload. Si cambia el esquema de features hay que actualizar ambos.

**Cómo correr una prueba:**

1. http://localhost:8089
2. Number of users = 50, spawn rate = 5
3. Host = `http://api.mlops.svc.cluster.local:8000`
4. Click **Start swarming**
5. Observar Grafana en paralelo para ver el efecto

## Estructura del repositorio

```
.
├── README.md
├── CLAUDE.md                       # Guía para asistente Claude (gitignored)
├── airflow/
│   ├── Dockerfile                  # Imagen propia: base apache/airflow + pipeline/ + CSV
│   ├── dags/diabetes_training_dag.py
│   ├── requirements.txt
│   └── values/values-local.yaml    # Helm values
├── api/                            # Código FastAPI
│   ├── main.py                     # endpoints + _align_features()
│   ├── model_loader.py             # cache con TTL del modelo MLflow
│   ├── db.py                       # log de inferencias a PG
│   ├── metrics.py                  # custom Prometheus counters
│   ├── schemas.py                  # Pydantic models
│   └── requirements.txt
├── ui/                             # Streamlit
│   ├── app.py                      # formulario + mapeo 17 fields → 141 features
│   └── requirements.txt
├── pipeline/                       # Lógica de entrenamiento (sin deps de Airflow)
│   ├── config.py
│   ├── db/
│   ├── ingest.py                   # cursor por offset, batches de 15k
│   ├── quality.py
│   ├── preprocess.py               # one-hot, imputación, filtro alta cardinalidad
│   ├── split.py                    # 70/15/15 estratificado
│   ├── train.py                    # LR + RF + infer_signature
│   ├── promote.py                  # alias champion
│   └── cli.py                      # entry point para CLI
├── docker/
│   ├── api/Dockerfile
│   ├── ui/Dockerfile
│   └── mlflow/Dockerfile
├── k8s/
│   ├── foundations/                # postgres + minio (kustomize)
│   ├── mlflow/
│   ├── api/
│   ├── ui/
│   ├── prometheus/
│   ├── grafana/
│   └── locust/
├── observability/
│   ├── dashboards/api.json         # Grafana dashboard (source of truth)
│   └── REPORT.md
├── loadtest/locustfile.py          # locust local (mantener sync con k8s/locust/configmap.yaml)
├── scripts/
│   └── port-forward-all.ps1        # Levanta forwards de los 10 servicios
└── data/data/Diabetes.csv          # Dataset (gitignored si es grande)
```

## Variables de configuración

Toda la configuración runtime pasa por env vars resueltas en `pipeline/config.py` o vía ConfigMap/Secret en k8s.

### Pipeline / DAG

| Variable | Default in-cluster | Para qué |
|---|---|---|
| `PG_DSN` | `postgresql://mlops_user:...@postgres-service.mlops:5432/mlops` | Conexión a Postgres |
| `MLFLOW_TRACKING_URI` | `http://mlflow-service.mlops:5000` | MLflow |
| `MLFLOW_S3_ENDPOINT_URL` | `http://minio-service.mlops:9000` | MinIO como S3 |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | `minioadmin / minioadmin123` | MinIO creds |
| `SOURCE_CSV` | `/opt/airflow/data/Diabetes.csv` | CSV fuente |
| `BATCH_SIZE` | `15000` | Cap obligatorio por enunciado |
| `RANDOM_SEED` | `42` | Reproducibilidad |
| `MLFLOW_EXPERIMENT` | `diabetes-classification` | — |
| `MLFLOW_MODEL_NAME` | `diabetes-classifier` | Nombre del registered model |
| `CHAMPION_ALIAS` | `champion` | Alias para el modelo productivo |
| `PRIMARY_METRIC` | `f1` | Métrica para seleccionar champion |

### API

Define en [k8s/api/configmap.yaml](k8s/api/configmap.yaml) + [k8s/api/secret.yaml](k8s/api/secret.yaml):

| Variable | Default | Para qué |
|---|---|---|
| `MODEL_CACHE_TTL_SECONDS` | `300` | TTL del cache del modelo en memoria |
| `MLFLOW_TRACKING_URI` | (idem) | — |
| `MLFLOW_S3_ENDPOINT_URL` | (idem) | — |
| `AWS_*` | (idem) | Para descargar artefactos de MinIO |
| `PG_DSN` | (idem) | Para log de inferencias |

### UI

| Variable | Default | Para qué |
|---|---|---|
| `API_URL` | `http://api.mlops.svc.cluster.local:8000` | Endpoint de la API |
| `API_CLIENT_TIMEOUT_SECONDS` | `30` | Timeout HTTP del cliente |

## Operación local

### Levantar todos los port-forwards de una vez

```powershell
.\scripts\port-forward-all.ps1
```

Detenerlos:

```powershell
Get-Job | Stop-Job | Remove-Job
```

### Reiniciar la API tras un cambio de imagen

```bash
kubectl -n mlops rollout restart deployment/api
kubectl -n mlops rollout status deployment/api
```

### Forzar a la API a recargar el modelo

Después de una promoción manual o tras correr el DAG, sin esperar el TTL:

```bash
curl -X POST http://localhost:8000/reload-model
```

### Inspeccionar el estado del pipeline

```sql
-- Cuántos batches se han cargado
SELECT batch_id, COUNT(*) FROM raw.diabetes_raw GROUP BY 1 ORDER BY 1;

-- Distribución de target por split
SELECT split, target, COUNT(*)
FROM clean.diabetes_clean GROUP BY 1, 2 ORDER BY 1, 2;

-- Cuántas features tiene el modelo actual
SELECT COUNT(DISTINCT k)
FROM clean.diabetes_clean, jsonb_object_keys(features) k;

-- Últimas inferencias
SELECT created_at, prediction, score, model_version, latency_ms
FROM inference.predictions ORDER BY created_at DESC LIMIT 20;
```

### Resetear los datos (sin tocar MLflow)

```bash
kubectl -n mlops exec postgres-0 -- psql -U mlops_user -d mlops -c \
  "TRUNCATE raw.diabetes_raw CASCADE; TRUNCATE clean.diabetes_clean CASCADE; TRUNCATE inference.predictions;"
```

### Resetear MLflow Model Registry

Vía API REST de MLflow:

```bash
curl -X DELETE http://localhost:5000/api/2.0/mlflow/registered-models/delete \
  -H "Content-Type: application/json" \
  -d '{"name":"diabetes-classifier"}'
```

## Decisiones técnicas

### Métrica principal: F1

Dataset clínicamente desbalanceado (~11% positivos en 130-US). **Accuracy es engañosa**: un modelo que prediga siempre 0 sacaría 89% accuracy con valor clínico nulo. Reportamos accuracy, precision, recall, F1 y ROC-AUC en cada run, pero la **promoción automática usa F1** ([pipeline/train.py](pipeline/train.py) + [pipeline/promote.py](pipeline/promote.py)), que balancea precision y recall sobre la clase positiva — falsos negativos (pacientes en riesgo no detectados) son costosos clínicamente.

### `class_weight=balanced` en ambos modelos

Sin esto, tanto LR como RF colapsan a predecir mayoritariamente 0 dada la imbalance. Con `balanced`, sklearn pesa inversamente a la frecuencia de cada clase durante el entrenamiento.

### Features como JSONB en lugar de columnas tipadas

`pd.get_dummies` produce un número variable de columnas según los valores categóricos presentes en el batch. Tres opciones tenía:

1. **Schema rígido con N columnas** → rompe cada vez que aparece una categoría nueva.
2. **Columna `features TEXT` con CSV serializado** → no consultable.
3. **JSONB con `features` como objeto** ← elegida. Consultable con `jsonb_object_keys`, sin DDL, tolerante a drift.

### Alineación dinámica de features en la API

Implementada en `_align_features()` ([api/main.py:85](api/main.py#L85)). El problema: el modelo en producción puede haber sido entrenado con N features, pero la UI envía siempre 141. Solución:

1. MLflow guarda el `signature` (lista de columnas + tipos) con cada modelo ([pipeline/train.py:111-117](pipeline/train.py#L111-L117)).
2. La API extrae ese signature al cargar el modelo.
3. Al recibir features de la UI, `df.reindex(columns=expected, fill_value=0.0)` rellena con 0 lo faltante y descarta lo desconocido.
4. Fallback: si no hay signature, usa `sklearn.feature_names_in_`.

Esto desacopla el contrato UI ↔ modelo, permitiendo iterar sobre el modelo sin tocar la UI.

### Cursor de batches por offset

El enunciado pide carga incremental en lotes de **máximo 15k**. La implementación ([pipeline/ingest.py:load_batch](pipeline/ingest.py)) cuenta cuántas filas existen en `raw.diabetes_raw` para el `source_file` actual y usa ese número como `skiprows` en `pd.read_csv`. Cada DAG run procesa los siguientes 15k. Tras ~7 runs el CSV se agota y `load_batch` reporta `inserted=0` sin error — el DAG sigue corriendo (preprocess + train) sobre el acumulado.

### Filtro de alta cardinalidad en preprocess

`diag_1/2/3` del dataset 130-US tienen >700 valores únicos (ICD codes), lo que produciría miles de columnas one-hot. El preprocess descarta categóricas con >20 valores únicos ([pipeline/preprocess.py:84-86](pipeline/preprocess.py#L84-L86)) para mantener bounded el espacio de features.

### Imagen Airflow propia (no sidecar / git-sync)

Embebemos `pipeline/`, los DAGs y el CSV en la imagen ([airflow/Dockerfile](airflow/Dockerfile)). Trade-off: cada cambio en `pipeline/` requiere rebuild + helm upgrade. Beneficio: cero dependencias en runtime (no git-sync, no PVC para datos), 100% reproducible.

### `webserverSecretKey` fijo en Airflow

Por defecto Airflow 3.x regenera el JWT secret en cada restart del webserver, lo que invalida los tokens de los task workers y falla los DAGs. Fijamos uno en [airflow/values/values-local.yaml:14](airflow/values/values-local.yaml#L14).

## Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| API en `CrashLoopBackOff` justo después del primer deploy | No hay modelo `champion` en MLflow todavía | Corre el DAG al menos una vez |
| API devuelve `503 no model available` | El champion fue borrado o MLflow está caído | Confirma con `kubectl get pods -n mlops`, luego `POST /reload-model` |
| API devuelve `400 invalid features: X has N features...` | Modelo champion sin signature + sin `feature_names_in_` | Borrar registered model y volver a correr el DAG con la imagen Airflow `>= v1.2.0` |
| DAG falla con `Invalid auth token: Signature verification failed` | `webserverSecretKey` no fijo, JWT se regeneró | Ya está fijo en `airflow/values/values-local.yaml`; haz `helm upgrade` |
| `t_load_batch` reporta `inserted=0, duplicates=15000` | El CSV ya se cargó completamente | Esperado. El DAG sigue corriendo sobre el acumulado |
| MLflow no carga un modelo entrenado en otra versión de sklearn | Diferencia entre `sklearn==1.4.2` (train) vs `1.5.0` (API) | Tolerable (warning); para reproducibilidad estricta, alinea versiones en `pipeline/requirements.txt` y `api/requirements.txt` |
| Grafana dashboard vacío | Prometheus no scrapea la API | Verifica anotaciones `prometheus.io/scrape` en el pod de la API |
| Streamlit sale en blanco / timeout | API tarda en pre-cargar el modelo (cold start) | Aumenta `API_CLIENT_TIMEOUT_SECONDS` en `k8s/ui/configmap.yaml`, ya está en 30s |
| `helm upgrade` queda colgado | PVCs con `Pending` por falta de StorageClass | Verifica `kubectl get sc`, en Docker Desktop suele ser `hostpath` |

---

**Repositorio:** https://github.com/CamiloGarcia06/Mlops_talleres
**Rama del proyecto:** `DanielDiaz/proyecto2`
**Dataset:** [Diabetes 130-US Hospitals (1999-2008) — UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
