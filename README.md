# MLOps Diabetes — Proyecto 2

Arquitectura MLOps end-to-end sobre Kubernetes para el dataset Diabetes 130-US hospitals (1999-2008). Las fases del proyecto se especifican en `openspec/changes/`.

## Fase 1 — Fundaciones (`fundaciones-kubernetes`)

Despliega namespace, PostgreSQL y MinIO en un cluster local de Kubernetes.

### Pre-requisitos
- Cluster local activo (microk8s, kind o minibube) con storage class por defecto.
- `kubectl` apuntando al cluster.
- (Opcional) Cuenta DockerHub si las imágenes propias serán privadas.

### Despliegue

```bash
kubectl apply -k k8s/foundations
kubectl -n mlops get pods,svc,pvc
```

El Job `minio-bootstrap` espera a que MinIO esté listo y crea el bucket `mlflow-artifacts`.

### Validación

```bash
# Postgres listo
kubectl -n mlops exec -it postgres-0 -- pg_isready -U mlops_user

# Schemas y bases creados por init.sql
kubectl -n mlops exec -it postgres-0 -- psql -U mlops_user -d mlops -c "\dn"
kubectl -n mlops exec -it postgres-0 -- psql -U mlops_user -l

# Bucket creado en MinIO
kubectl -n mlops logs job/minio-bootstrap

# Console MinIO (en otra terminal)
kubectl -n mlops port-forward svc/minio-service 9001:9001
# luego abrir http://localhost:9001
```

### Limpieza

```bash
kubectl delete -k k8s/foundations
# si quieres reciclar también los PVCs:
kubectl -n mlops delete pvc --all
```

## Estructura del repositorio

```
k8s/
  namespace.yaml
  foundations/kustomization.yaml   # apply ordenado de la fase 1
  postgres/                        # StatefulSet, PVC, Service, Secret, ConfigMap (init.sql)
  minio/                           # StatefulSet, PVC, Service, Secret, bootstrap Job
  mlflow/                          # (fase 2)
airflow/                           # (fase 4)
openspec/                          # specs y changes por fase
.claude/                           # agentes y skills del proyecto
```

## Fase 2 — MLflow Tracking (`tracking-mlflow`)

Despliega el servidor MLflow con backend PostgreSQL y artifact store MinIO.

### Imagen Docker propia

La imagen se construye desde `docker/mlflow/Dockerfile` (base `python:3.11-slim`, incluye `mlflow==2.13.0`, `psycopg2-binary` y `boto3`).

```bash
# Reemplazar <docker-user> por tu usuario de DockerHub
docker build -t <docker-user>/mlops-mlflow:v0.1.0 -f docker/mlflow/Dockerfile docker/mlflow/
docker push <docker-user>/mlops-mlflow:v0.1.0
```

Antes del push, editar `k8s/mlflow/deployment.yaml` y sustituir `<docker-user>` por el usuario real.

### Despliegue

```bash
# La fase 2 se despliega de forma independiente a las foundations
kubectl apply -k k8s/mlflow
kubectl -n mlops get pods,svc -l app=mlflow
```

### Acceso a la UI (port-forward)

```bash
kubectl -n mlops port-forward svc/mlflow-service 5000:5000
# Abrir http://localhost:5000
```

### Smoke test

```bash
# Terminal 1 — port-forward MLflow
kubectl -n mlops port-forward svc/mlflow-service 5000:5000

# Terminal 2 — port-forward MinIO API
kubectl -n mlops port-forward svc/minio-service 9000:9000

# Terminal 3 — ejecutar smoke test
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin123
pip install mlflow psycopg2-binary boto3   # solo si no están instalados localmente
python scripts/mlflow_smoke.py
```

### Validación manual de backend y artefactos

```bash
# Verificar tablas de MLflow en Postgres
kubectl -n mlops exec -it postgres-0 -- \
  psql -U mlops_user -d mlflow -c "\dt"

# Verificar artefacto en MinIO vía mc (MinIO Client)
kubectl -n mlops port-forward svc/minio-service 9000:9000 &
mc alias set local http://localhost:9000 minioadmin minioadmin123
mc ls local/mlflow-artifacts --recursive
```

### Limpieza

```bash
kubectl delete -k k8s/mlflow
```

## Fase 3 — Pipeline de Entrenamiento (`pipeline-entrenamiento`)

Paquete Python `pipeline/` con la lógica de ingesta, calidad, preprocesamiento, split, entrenamiento y promoción del modelo. Es invocable por CLI y será envuelto por el DAG de Airflow en la fase 4.

### Estructura

```
pipeline/
  config.py            # env vars (DB, MLflow, MinIO, batch size, seed, métrica)
  db/
    connection.py
    migrations.py      # crea raw.diabetes_raw, clean.diabetes_clean, inference.predictions
  ingest.py            # CSV -> raw, chunks <= 15.000, row_hash UNIQUE
  quality.py           # validaciones mínimas
  preprocess.py        # raw -> clean (JSONB features + target binarizado)
  split.py             # train/val/test estratificado, seed configurable
  train.py             # LogisticRegression + RandomForest, log a MLflow
  promote.py           # alias `champion` si supera al actual
  cli.py               # entrypoint unificado
```

### Variables de entorno

| Variable | Default |
|---|---|
| `PG_DSN` | `postgresql://mlops_user:...@postgres-service.mlops:5432/mlops` |
| `MLFLOW_TRACKING_URI` | `http://mlflow-service.mlops:5000` |
| `MLFLOW_S3_ENDPOINT_URL` | `http://minio-service.mlops:9000` |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | credenciales MinIO |
| `SOURCE_CSV` | `/data/Diabetes.csv` |
| `BATCH_SIZE` | `15000` (cap obligatorio del proyecto) |
| `RANDOM_SEED` | `42` |
| `MLFLOW_EXPERIMENT` | `diabetes-classification` |
| `MLFLOW_MODEL_NAME` | `diabetes-classifier` |
| `CHAMPION_ALIAS` | `champion` |
| `PRIMARY_METRIC` | `f1` |

### Métrica principal: F1 (justificación)

Para la clasificación clínica binaria del dataset, **accuracy es engañosa** porque la clase positiva (paciente diabético / readmitido) suele ser minoritaria. Un modelo que prediga siempre la clase mayoritaria obtendría alta accuracy y cero valor clínico. Reportamos accuracy, precision, recall, F1 y ROC-AUC en cada run, pero la **selección automática del champion usa F1**, que balancea precision y recall sobre la clase positiva — el caso clínicamente costoso (falsos negativos = pacientes en riesgo no detectados).

### Ejecutar el pipeline end-to-end

```bash
# desde el cluster (port-forwards activos en otra terminal)
pip install -r pipeline/requirements.txt
export PG_DSN="postgresql://mlops_user:mlops_pass_2026@localhost:5432/mlops"
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin123
export SOURCE_CSV=./data/Diabetes.csv

python -m pipeline.cli all --source ./data/Diabetes.csv
```

O paso a paso:

```bash
python -m pipeline.cli migrate
python -m pipeline.cli ingest --source ./data/Diabetes.csv
python -m pipeline.cli quality
python -m pipeline.cli preprocess
python -m pipeline.cli split
python -m pipeline.cli train
```

### Validación

```bash
# Filas crudas
kubectl -n mlops exec -it postgres-0 -- \
  psql -U mlops_user -d mlops -c "SELECT count(*), count(distinct batch_id) FROM raw.diabetes_raw;"

# Filas limpias con split asignado
kubectl -n mlops exec -it postgres-0 -- \
  psql -U mlops_user -d mlops -c "SELECT split, count(*) FROM clean.diabetes_clean GROUP BY split;"

# Modelo champion en MLflow
curl -s "$MLFLOW_TRACKING_URI/api/2.0/mlflow/registered-models/alias?name=diabetes-classifier&alias=champion" \
  | python3 -m json.tool

# Re-ejecución no duplica filas
python -m pipeline.cli ingest --source ./data/Diabetes.csv  # inserted=0, duplicates=N
```

## Fase 4 — Orquestación con Airflow (`orquestacion-airflow`)

Despliegue de Airflow en Kubernetes (Helm chart oficial, `LocalExecutor`) con metastore en el Postgres del proyecto (DB `airflow`). El DAG `diabetes_mlops_pipeline` orquesta el paquete `pipeline/` de la fase 3.

### Imagen propia (DAG + pipeline embebidos)

```bash
# Build context = repo root para que pueda copiar pipeline/
docker build -f airflow/Dockerfile -t <docker-user>/mlops-airflow:v0.1.0 .
docker push <docker-user>/mlops-airflow:v0.1.0
```

Antes del helm install, sustituir `<docker-user>` en `airflow/values/values-local.yaml` (`images.airflow.repository`).

### Despliegue

```bash
helm repo add apache-airflow https://airflow.apache.org
helm repo update
helm upgrade --install airflow apache-airflow/airflow \
  -n mlops -f airflow/values/values-local.yaml
kubectl -n mlops rollout status deploy/airflow-webserver
```

### UI

```bash
kubectl -n mlops port-forward svc/airflow-webserver 8080:8080
# http://localhost:8080  (admin / admin por defecto del chart)
```

### Tareas del DAG

`migrate -> check_source -> load_batch -> quality -> preprocess -> split -> train -> compare -> promote`

Cada tarea es una llamada delgada a un módulo de `pipeline/`. La idempotencia está garantizada por `row_hash UNIQUE` en raw, upserts en clean y aliases (no stages) en MLflow.

### Validación

```bash
# Listar el DAG
kubectl -n mlops exec deploy/airflow-scheduler -- airflow dags list | grep diabetes

# Trigger manual
kubectl -n mlops exec deploy/airflow-scheduler -- \
  airflow dags trigger diabetes_mlops_pipeline

# Re-ejecución no duplica raw
kubectl -n mlops exec -it postgres-0 -- \
  psql -U mlops_user -d mlops -c "SELECT count(*) FROM raw.diabetes_raw;"
# (debe mantenerse igual entre dos triggers consecutivos sobre la misma fuente)

# Modelo champion en MLflow
curl -s "http://localhost:5000/api/2.0/mlflow/registered-models/alias?name=diabetes-classifier&alias=champion" | python3 -m json.tool
```

### Notas

- El CSV fuente se monta en `/opt/airflow/data/Diabetes.csv`. El `extraVolumes/extraVolumeMounts` por defecto usa `emptyDir`; en un cluster real reemplázalo por un PVC poblado con el archivo o un init container que lo descargue.
- Si una tarea falla, las posteriores no corren (default `trigger_rule=all_success`). El error es visible en la UI y la tarea es reintentable.

## Fase 5 — API de Inferencia (`api-inferencia`)

API FastAPI que carga el modelo `champion` desde MLflow, registra cada inferencia en `inference.predictions` y expone métricas Prometheus.

### Endpoints

| Método | Path | Descripción |
|---|---|---|
| GET | `/health` | Liveness/readiness lógico |
| GET | `/model-info` | Nombre, versión, alias y `loaded_at` del modelo en caché |
| POST | `/reload-model` | Fuerza la recarga del modelo desde MLflow |
| POST | `/predict` | Recibe `{"features": {...}}`, retorna predicción + score + modelo + versión + tiempo |
| GET | `/metrics` | Métricas Prometheus (HTTP + custom de inferencia) |

### Build & push

```bash
docker build -f docker/api/Dockerfile -t <docker-user>/mlops-api:v0.1.0 .
docker push <docker-user>/mlops-api:v0.1.0
# editar k8s/api/deployment.yaml -> sustituir <docker-user>
```

### Despliegue

```bash
kubectl apply -k k8s/api
kubectl -n mlops rollout status deploy/api
```

### Validación

```bash
# Salud
kubectl -n mlops port-forward svc/api 8000:8000
curl -s http://localhost:8000/health

# Modelo cargado
curl -s http://localhost:8000/model-info | python3 -m json.tool

# Predicción (ajustar features al esquema del clean.diabetes_clean)
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": {"glucose": 130, "bmi": 31.4, "age": 45}}' | python3 -m json.tool

# Inferencia registrada en BD
kubectl -n mlops exec -it postgres-0 -- \
  psql -U mlops_user -d mlops -c "SELECT request_id, prediction, score, model_version, latency_ms FROM inference.predictions ORDER BY created_at DESC LIMIT 5;"

# Métricas Prometheus
curl -s http://localhost:8000/metrics | head -40

# Promoción de un nuevo champion sin redeploy: tras correr el DAG y ver una versión nueva,
# se autocarga al expirar el cache (5 min) o forzando:
curl -s -X POST http://localhost:8000/reload-model | python3 -m json.tool
```

## Próximas fases

Ver `openspec/changes/`:
- `api-inferencia` (fase 5)
- `ui-streamlit` (fase 6)
- `observabilidad` (fase 7)

---

# Hallazgos y desvíos por fase

Registro vivo de decisiones, desvíos del enunciado y cosas pendientes que el estudiante debe resolver antes de la sustentación. Se actualiza al cierre de cada fase.

---

## Fase 1 — Fundaciones Kubernetes (`fundaciones-kubernetes`)

Sin desvíos. Manifiestos alineados con la spec.

---

## Fase 2 — Tracking MLflow (`tracking-mlflow`)

**Imagen propia con placeholder de DockerHub.**
- En `k8s/mlflow/deployment.yaml` la imagen aparece como `<docker-user>/mlops-mlflow:v0.1.0`.
- **Acción requerida:** sustituir `<docker-user>` por el usuario real antes de `kubectl apply`.

---

## Fase 3 — Pipeline de Entrenamiento (`pipeline-entrenamiento`)

**Pipeline dataset-agnóstico.**
- `pipeline/preprocess.py` detecta el target automáticamente entre `Outcome` (Pima) y `readmitted` (Diabetes 130-US) y persiste features como JSONB.
- **Razón:** el `data_load.py` previo del repo descarga un dataset distinto al del PDF (Pima 8 features vs. 130-US >50 features). Mantener la pipeline genérica permite alternar sin reescribir DDL.
- **Acción requerida:** confirmar con el profesor cuál dataset es el válido para la entrega. Si es 130-US, reemplazar `data/Diabetes.csv` por el archivo correcto y volver a correr `python -m pipeline.cli all`.

**DAG previo desalineado (resuelto en fase 4).**
- El DAG anterior (`airflow/dags/diabetes_training_dag.py` antes de fase 4) usaba DB `mlflow_db` y schemas con sufijo `_layer`, inconsistentes con fase 1 (DB `mlops`, schemas `raw`/`clean`/`inference`).
- **Resuelto:** la fase 4 lo reescribió para invocar `pipeline/`.

---

## Fase 4 — Orquestación Airflow (`orquestacion-airflow`)

**Volumen del CSV fuente vacío.**
- En `airflow/values/values-local.yaml` el montaje `/opt/airflow/data` es `emptyDir`, así que los pods de Airflow arrancan sin el archivo `Diabetes.csv`.
- **Razón:** la estrategia correcta depende del cluster del estudiante (microk8s vs. kind vs. minikube) y de si quieres versionar el CSV o descargarlo en runtime.
- **Acción requerida — elegir UNA opción:**
  1. **PVC pre-poblado:** crear `data-pvc`, copiar el archivo con `kubectl cp data/Diabetes.csv <pod>:/data/`, y reemplazar `extraVolumes` por:
     ```yaml
     extraVolumes:
       - name: data
         persistentVolumeClaim: { claimName: data-pvc }
     ```
  2. **Init container que descarga:** añadir a `extraInitContainers` un container que haga `curl/wget` al Google Drive del PDF o a MinIO (donde podrías subir el CSV una vez).
  3. **MinIO como fuente:** subir `Diabetes.csv` a un bucket en MinIO y modificar `pipeline/ingest.check_source` para leerlo vía `boto3` en vez de filesystem. Esto se acerca al patrón productivo y aprovecha la infra ya desplegada.
- **Recomendación:** opción 3 (MinIO). Reusa la infra y simula mejor un escenario real de producción.

**Credenciales admin/admin del Helm chart.**
- El chart `apache-airflow` crea un usuario `admin/admin` por defecto.
- **Razón:** suficiente para desarrollo local; la sustentación es local.
- **Acción opcional:** si quieres credenciales propias, añadir `webserver.defaultUser` en `values-local.yaml` o configurar auth externo.

**Imagen propia con placeholder de DockerHub.**
- `airflow/values/values-local.yaml` referencia `<docker-user>/mlops-airflow:v0.1.0`.
- **Acción requerida:** sustituir antes de `helm install`.

**Build context obligatorio = repo root.**
- El Dockerfile debe construirse desde la raíz para poder copiar `pipeline/`. Ver primer comentario del Dockerfile.

---

## Fase 5 — API de Inferencia (`api-inferencia`)

**Esquema de entrada flexible (dict de features).**
- `PredictRequest.features` es un `Dict[str, Any]` en vez de un BaseModel con campos tipados explícitos.
- **Razón:** la pipeline persiste features como JSONB precisamente para tolerar Pima vs. 130-US. Un esquema rígido obligaría a regenerar el código cada vez que cambie el dataset.
- **Trade-off:** la validación 422 por campo faltante recae en sklearn al hacer `predict`, no en Pydantic. La API responde 400 con detalle, no 422.
- **Acción opcional:** una vez fijado el dataset definitivo, derivar un schema tipado de las columnas del modelo registrado y reemplazar el dict.

**Score solo si el modelo expone `predict_proba`.**
- El loader extrae el sklearn nativo del wrapper `pyfunc` para llamar `predict_proba`. Si el champion fuera un modelo sin probabilidades (regresión, custom flavor), `score` queda `null`.
- **Acción:** documentar en la sustentación qué modelos del Registry exponen score.

**Placeholder `<docker-user>` en `k8s/api/deployment.yaml`.**
- **Acción requerida:** sustituir antes de `kubectl apply`.

**Cache TTL = 300s.**
- Una promoción de champion tarda hasta 5 minutos en propagarse a la API automáticamente, o instantánea con `POST /reload-model`.
- Para la sustentación, llamar `/reload-model` después de promover deja el efecto inmediato y observable.

---

## Fase 6 — UI Streamlit (`ui-streamlit`)

**Placeholder `<docker-user>` en `k8s/ui/deployment.yaml`.**
- **Accion requerida:** sustituir `<docker-user>` por el usuario real de DockerHub antes de `kubectl apply`.
- Luego ejecutar:
  ```bash
  docker tag mlops-ui:v0.1.0 <docker-user>/mlops-ui:v0.1.0
  docker push <docker-user>/mlops-ui:v0.1.0
  ```

**Imagen construida localmente como `mlops-ui:v0.1.0` (814 MB).**
- Streamlit incluye muchas dependencias transitivas. El tamanio es esperable para un entorno de desarrollo.
- Para produccion, considerar imagen multi-stage o slim sin dependencias de desarrollo.

**Formulario dual (Diabetes 130-US y Pima).**
- La UI detecta en `session_state` si el usuario cargo un ejemplo Pima para mostrar el formulario correcto.
- Esto asegura compatibilidad con ambas variantes del dataset sin reescribir el formulario.

**Probes de Streamlit via `/_stcore/health`.**
- Streamlit >= 1.30 expone `/_stcore/health` como endpoint de salud. Se usa en readiness y liveness probes.
- Si la imagen usa Streamlit < 1.30, cambiar el probe path a `/_stcore/host-config` o usar `tcpSocket` en su lugar.

**Validaciones de Scenarios 3.1-3.3 diferidas (requieren cluster activo).**
- Ver seccion "Comandos para validar" en el reporte final de la fase.

---

## Cross-cutting

**`kubectl` y `helm` no instalados en la máquina de implementación.**
- Todas las validaciones runtime (apply, port-forward, exec) están listadas como "validación diferida" en el README de cada fase, con los comandos exactos.
- **Acción requerida:** correrlas el estudiante en su cluster local antes de la sustentación.

**Imágenes en DockerHub.**
- Tres imágenes propias deben publicarse:
  - `<docker-user>/mlops-mlflow:v0.1.0`
  - `<docker-user>/mlops-airflow:v0.1.0`
  - `<docker-user>/mlops-api:v0.1.0`
  - `<docker-user>/mlops-ui:v0.1.0` (fase 6 - construida localmente, pendiente push)
- El enunciado lo exige explícitamente para la entrega final.
