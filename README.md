# MLOps Proyecto Final — Prediccion de Precios de Propiedades

Plataforma MLOps end-to-end sobre Kubernetes para predecir precios de propiedades inmobiliarias. El sistema consume datos incrementalmente desde una API externa, valida calidad y drift, decide si reentrenar, entrena y compara modelos de regresion, y promueve automaticamente al mejor.

**Curso:** Operaciones de Machine Learning — Pontificia Universidad Javeriana, 2026-1
**Rama:** `CamiloGarcia/proyecto-final`
**Repositorio:** https://github.com/CamiloGarcia06/Mlops_talleres

## Tabla de contenidos

- [Arquitectura](#arquitectura)
- [Componentes](#componentes)
- [Quickstart](#quickstart)
- [Flujo del DAG](#flujo-del-dag)
- [Criterios de entrenamiento](#criterios-de-entrenamiento)
- [Reglas de promocion](#reglas-de-promocion)
- [API de inferencia](#api-de-inferencia)
- [Modelo de datos](#modelo-de-datos)
- [CI/CD y GitOps](#cicd-y-gitops)
- [Observabilidad](#observabilidad)
- [Decisiones de disenio](#decisiones-de-disenio)
- [Credenciales](#credenciales)

## Arquitectura

```
Usuario → Push → GitHub → GitHub Actions → DockerHub
                                              ↓
                                          Argo CD → Kubernetes (namespace mlops)
                                                    ┌─────────────────────────┐
                                                    │  Fase de entrenamiento  │
        API Datos (externa) ──HTTP──→ Airflow DAG ──→ PostgreSQL (raw/clean)  │
                                         │          │  MLflow ←→ MinIO        │
                                         │          └─────────────────────────┘
                                         │          ┌─────────────────────────┐
                                         │          │  Fase de inferencia     │
                                         │          │  FastAPI ←── MLflow     │
                                         │          │  Streamlit → FastAPI    │
                                         │          │  Locust → FastAPI       │
                                         │          └─────────────────────────┘
                                         │          ┌─────────────────────────┐
                                         │          │  Observabilidad         │
                                         └──────────│  Prometheus → Grafana   │
                                                    └─────────────────────────┘
```

**Dos planos separados:**
- **Entrenamiento:** Airflow consume la API del profesor, almacena datos crudos, valida, decide si entrenar, entrena 3 modelos (LinearRegression, RandomForest, GradientBoosting), registra en MLflow y promueve al mejor por MAE.
- **Inferencia:** FastAPI carga el modelo `champion` desde MLflow (cache 5 min), responde predicciones de precios, y registra cada llamada en `inference.predictions`.

## Componentes

| Componente | Tecnologia | Imagen DockerHub | Proposito |
|---|---|---|---|
| Orquestador | Airflow 3.x | `camilogarcia06/mlops-airflow` | DAG de 19 tareas con bifurcaciones |
| BD relacional | PostgreSQL 16 | `postgres:16` (oficial) | Schemas `raw`, `clean`, `audit`, `inference` + DBs `mlflow`, `airflow` |
| Artifact store | MinIO | `minio/minio` (oficial) | Bucket `mlflow-artifacts` |
| Registro ML | MLflow 2.13 | `camilogarcia06/mlops-mlflow` | Backend PG + artifacts S3 |
| API inferencia | FastAPI | `camilogarcia06/mlops-api` | `/health /predict /model-info /reload-model /metrics` |
| UI | Streamlit | `camilogarcia06/mlops-ui` | Inferencia + historial de entrenamiento |
| Pruebas de carga | Locust | `locustio/locust` (oficial) | Escenario contra `/predict` |
| Metricas | Prometheus | `prom/prometheus` (oficial) | Scrape `/metrics` de la API |
| Dashboards | Grafana | `grafana/grafana` (oficial) | Latencias, RPS, errores, modelo activo |
| GitOps | Argo CD | Instalacion oficial | Sincroniza manifiestos Git → cluster |

## Quickstart

### Prerrequisitos

- microk8s con addons `dns` y `hostpath-storage` habilitados
- `kubectl`, `helm`, `docker`, `make`
- API del profesor: `docker run --rm -d -p 8000:80 --name data-api cristiandiaz13/mlops-puj:data-api-pf-v1`

### Levantar todo

```bash
make data-api          # API del profesor (fuera del cluster)
make up                # Foundations + MLflow + Airflow + API + UI + observabilidad + Argo CD
make forward           # Port-forwards a todos los servicios
```

### Ejecutar el pipeline

```bash
make data-restart      # Reiniciar datos del grupo 1
make trigger-dag       # Disparar DAG desde Airflow
```

### Otros comandos

| Comando | Que hace |
|---|---|
| `make up` | Levanta todo el cluster |
| `make down` | Tumba todo |
| `make forward` | Abre port-forwards con tabla de URLs |
| `make stop-forward` | Cierra port-forwards |
| `make status` | Pods, services, Argo CD |
| `make data-api` | Levanta API del profesor |
| `make data-restart` | Reinicia datos del grupo 1 |
| `make data-stop` | Detiene API del profesor |
| `make trigger-dag` | Dispara el DAG |
| `make build-images` | Construye las 4 imagenes localmente |
| `make import-images` | Importa imagenes a microk8s |
| `make migrate` | Aplica migraciones SQL |

## Flujo del DAG

19 tareas con dos bifurcaciones explicitas:

```
start → fetch_batch_from_api → store_raw_batch
    → validate_schema
    → validate_data_quality      → decide_training ─┬→ train_candidate_model
    → detect_new_categories                          │   → evaluate_candidate_model
    → detect_data_drift                              │   → register_candidate_in_mlflow
    → preprocess_data                                │   → compare_with_production
                                                     │   → decide_promotion ─┬→ promote_model
                                                     │                       └→ reject_model
                                                     └→ skip_training
    → notify_or_log_result → end
```

- **`decide_training`**: bifurca entre entrenar o no, basado en criterios tecnicos.
- **`decide_promotion`**: bifurca entre promover o rechazar el modelo candidato.
- Tareas despues de bifurcaciones usan `trigger_rule="none_failed_min_one_success"`.

## Criterios de entrenamiento

Se evaluan en `decide_training`. Cualquiera dispara entrenamiento:

1. **No hay champion** — primer lote, se entrena siempre.
2. **Data drift significativo** — test KS sobre features numericas (`price`, `house_size`, `acre_lot`). Umbral: p-value < 0.05 en al menos 2 features.
3. **Nuevas categorias frecuentes** — categorias no vistas en `city` o `state` que representen > 5% del lote nuevo.
4. **Volumen acumulado** — datos en clean aumentaron >= 10% respecto al volumen del champion.

**Cuando NO se entrena:** sin drift, sin categorias nuevas significativas, volumen insuficiente. Se registra la razon en `audit.batch_log`.

## Reglas de promocion

**Regla explicita:** *"Promover si el MAE del candidato es al menos 3% menor que el del champion actual Y el RMSE no empeora mas de 1%"*.

- **Metrica primaria: MAE** — interpretable en dolares ("el modelo se equivoca en promedio $X").
- **Guarda RMSE** — evita sacrificar estabilidad ante outliers.
- **Sin champion previo** — primer modelo se promueve automaticamente.
- Toda decision queda registrada en `audit.batch_log` con metricas antes/despues.

**Resultados observados (batch 0, 15k filas):**

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| GradientBoostingRegressor | $44,759 | $56,230 | 0.364 |
| LinearRegression | $44,797 | $57,439 | 0.336 |
| RandomForestRegressor | $46,887 | $59,194 | 0.295 |

## API de inferencia

| Metodo | Endpoint | Descripcion |
|---|---|---|
| GET | `/health` | Liveness/readiness |
| GET | `/model-info` | Modelo activo, version, alias, TTL |
| POST | `/predict` | `{"features": {...}}` → `{prediction, model_name, model_version, processing_time_ms}` |
| POST | `/reload-model` | Recarga inmediata del champion desde MLflow |
| GET | `/metrics` | Metricas Prometheus |

**Ejemplo de prediccion:**
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"bed":3.0,"bath":2.0,"acre_lot":0.5,"house_size":1800.0,"city":"Hartford","state":"Connecticut","status":"for_sale","zip_code":"6105","prev_sold_year":2015.0}}'
```

## Modelo de datos

Cuatro schemas en PostgreSQL:

### `raw.properties_raw`
| Columna | Tipo | Descripcion |
|---|---|---|
| `row_hash` | TEXT PK | SHA-256, garantiza idempotencia |
| `batch_id` | TEXT | Identificador del lote |
| `load_timestamp` | TIMESTAMPTZ | Fecha de ingesta |
| `source` | TEXT | Origen (default `'api'`) |
| `payload` | JSONB | Fila original de la API |

### `clean.properties_clean`
| Columna | Tipo | Descripcion |
|---|---|---|
| `id` | BIGSERIAL PK | — |
| `row_hash` | TEXT FK | Trazabilidad al dato crudo |
| `features` | JSONB | Features transformadas |
| `target` | DOUBLE PRECISION | Precio (`price`) |
| `split` | TEXT | `train` / `test` |

### `audit.batch_log`
| Columna | Tipo | Descripcion |
|---|---|---|
| `batch_id` | TEXT | Lote procesado |
| `training_decision` | BOOLEAN | Se entreno? |
| `training_reason` | TEXT | Razon |
| `promotion_decision` | BOOLEAN | Se promovio? |
| `champion_metric_before` | DOUBLE PRECISION | MAE anterior |
| `champion_metric_after` | DOUBLE PRECISION | MAE nuevo |

### `inference.predictions`
| Columna | Tipo | Descripcion |
|---|---|---|
| `request_id` | UUID PK | ID de la peticion |
| `prediction` | DOUBLE PRECISION | Precio predicho |
| `model_version` | TEXT | Version del modelo |
| `latency_ms` | DOUBLE PRECISION | Tiempo de respuesta |

## CI/CD y GitOps

### GitHub Actions

4 workflows en `.github/workflows/`, uno por componente:
- `build-api.yml` — triggers en `api/`, `docker/api/`
- `build-ui.yml` — triggers en `ui/`, `docker/ui/`
- `build-mlflow.yml` — triggers en `docker/mlflow/`
- `build-airflow.yml` — triggers en `airflow/`, `pipeline/`

Cada push genera tags `sha-<commit>` + `latest`. En PRs solo build sin push.

### Argo CD

4 Applications con sincronizacion automatica (`selfHeal: true`, `prune: true`):
- `mlops-foundations` → `k8s/foundations/`
- `mlops-mlflow` → `k8s/mlflow/`
- `mlops-api` → `k8s/api/`
- `mlops-ui` → `k8s/ui/`

**Flujo completo:** Push → GitHub Actions build imagen → DockerHub → Argo CD detecta cambio en manifiesto → sincroniza al cluster.

## Observabilidad

- **Prometheus** scrappea `/metrics` de la API cada 10s.
- **Grafana** dashboard con: RPS, latencia p50/p95/p99, tasa de errores, modelo activo.
- **Locust** genera carga contra `/predict` con payload de propiedades (features randomizadas).

## Decisiones de disenio

| Decision | Justificacion |
|---|---|
| Un solo PostgreSQL con esquemas separados | Reduce recursos, simplifica credenciales; esquemas dan aislamiento logico suficiente |
| MinIO como object store | Compatibilidad S3, desplegable en cluster sin dependencias externas |
| Drop `brokered_by` y `street` | IDs numericos sin valor predictivo; reducirian ruido y dimensionalidad |
| `prev_sold_date` → `prev_sold_year` | Captura antiguedad sin granularidad temporal inutil para regresion tabular |
| `handle_unknown='ignore'` en encoder | Categorias nuevas producen vector de ceros; seguro para arboles y regresion lineal |
| MAE como metrica primaria | Interpretable en dolares; RMSE como guarda contra outliers |
| 3 modelos (LR, RF, GB) sin fijar a priori | Se promueve el mejor por MAE; permite que el sistema descubra cual funciona mejor |
| API del profesor fuera del cluster | Simula fuente externa como pide el enunciado (seccion 6.1) |
| `mlflow==2.13.0` fijado | Compatibilidad cliente-servidor; version 3.x incompatible con server 2.13 |

## Credenciales

| Servicio | Usuario | Contrasena |
|---|---|---|
| Airflow | `admin` | `admin` |
| Grafana | `admin` | `admin` |
| MinIO | `minioadmin` | `minioadmin123` |
| Argo CD | `admin` | `admin` |
| PostgreSQL | `mlops_user` | `mlops_pass_2026` |
