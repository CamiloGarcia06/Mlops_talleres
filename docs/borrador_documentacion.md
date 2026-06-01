# Borrador de Documentacion — Proyecto Final MLOps 2026-1

> Este archivo es un borrador de trabajo. Registra gotchas, decisiones de disenio y
> tecnicas relevantes a medida que se implementa el proyecto. Al final se organiza
> y se integra en el README o documento de entrega.

---

## Decisiones de disenio

### Arquitectura general

- [x] **Un solo PostgreSQL con esquemas separados** (`raw`, `clean`, `audit`, `inference` + los propios de MLflow y Airflow). Justificacion: reduce la huella de recursos en un cluster local (un solo StatefulSet con PVC), simplifica la gestion de credenciales (un Secret), y los esquemas proporcionan aislamiento logico suficiente. El enunciado permite esta configuracion siempre que se justifique y configure correctamente (seccion 6.5 del PDF).
- [x] **MinIO como object store**. Justificacion: compatibilidad nativa con la API S3 que MLflow espera para artefactos, se despliega como un pod dentro del cluster sin dependencias externas, y permite documentar la creacion del bucket, credenciales y politicas de acceso como pide el enunciado (seccion 6.4).

### Dataset y preprocesamiento

- [x] **Variables categoricas de alta cardinalidad (`brokered_by`, `street`)**: se eliminan del feature set. Ambas son IDs numericos codificados como float (no representan magnitudes); incluirlas introduciria ruido y dimensionalidad excesiva sin valor predictivo. `city` se mantiene como categorica con encoding.
- [x] **`prev_sold_date`**: se extrae el anio como feature numerica (`prev_sold_year`) y se descarta la fecha original. El anio de ultima venta captura la antiguedad de la transaccion previa sin introducir granularidad temporal que el modelo no puede explotar con regresion tabular.
- [x] **Categorias desconocidas en inferencia**: se usa `handle_unknown='ignore'` en el encoder (OrdinalEncoder o OneHotEncoder). Esto produce un vector de ceros para categorias no vistas, lo cual es seguro para modelos de arboles y regresion lineal. Alternativa descartada: agrupar en `other` requiere reentrenar el encoder ante cada categoria nueva, lo cual rompe la idempotencia del preprocesamiento.
- [x] **Features numericas (`bed`, `bath`, `acre_lot`, `house_size`, `zip_code`)**: se aplica StandardScaler. `zip_code` se trata como categorica (OrdinalEncoder) dado que es un codigo geografico, no una magnitud continua. Outliers extremos en `price` (target) se filtran en la etapa de calidad (ej. `price < 1` o `price > 10M`).

### Metricas y reglas de promocion

- [x] **Metrica primaria: MAE (Mean Absolute Error)**. Se elige por interpretabilidad directa: el error se expresa en dolares, lo que facilita comunicar al negocio "el modelo se equivoca en promedio $X". RMSE penaliza errores grandes pero es menos intuitivo; R² es adimensional y no dice cuanto se equivoca el modelo en terminos absolutos.
- [x] **Regla explicita de promocion**: *"Promover si el MAE del candidato es al menos 3% menor que el del champion actual Y el RMSE no empeora mas de 1%"*. Se registran ambas metricas en MLflow y en `audit.batch_log`.
- [x] **Justificacion de la regla**: el umbral de 3% en MAE evita promover modelos con mejoras marginales que podrian ser ruido estadistico. La guarda en RMSE (no empeorar >1%) asegura que el candidato no sacrifique estabilidad ante outliers a cambio de una mejora promedio. Si no hay champion previo (primer modelo), se promueve automaticamente.

### Decision de entrenamiento

- [x] **Criterios tecnicos** (se evaluan en `decide_training`, cualquiera dispara entrenamiento):
  1. **Data drift significativo**: test KS (Kolmogorov-Smirnov) sobre features numericas clave (`price`, `house_size`, `acre_lot`). Umbral: p-value < 0.05 en al menos 2 features.
  2. **Nuevas categorias frecuentes**: si aparecen categorias no vistas en `city` o `state` que representen > 5% del lote nuevo.
  3. **Volumen acumulado**: el lote nuevo aumenta el volumen total de datos en clean en al menos 10% respecto al volumen que se uso para entrenar el champion actual.
  4. **Degradacion del modelo actual**: si se dispone de datos etiquetados recientes, evaluar el champion y verificar que su MAE no se haya degradado > 5% respecto a sus metricas registradas.
  5. **No hay champion**: primer lote, se entrena siempre.
- [x] **Justificacion de umbrales**: KS p-value 0.05 es estandar en pruebas de hipotesis; el 5% de categorias nuevas evita reentrenar por ruido de una sola observacion nueva; el 10% de volumen asegura que el reentrenamiento incorpora datos sustancialmente nuevos y no es un desperdicio de computo.
- [x] **Cuando NO se entrena**: lote demasiado pequenio, sin drift, sin categorias nuevas significativas, y volumen acumulado insuficiente. Se registra la razon en `audit.batch_log.training_reason`.

### Modelos

- [x] **Modelos evaluados**: LinearRegression (baseline), RandomForestRegressor, GradientBoostingRegressor. Los tres son regressors validos para prediccion de precios. Se entrenan y registran en MLflow como runs del mismo experimento; el pipeline selecciona automaticamente el de menor MAE.
- [x] **Resultados observados (batch 0, 15k filas)**:
  - GradientBoostingRegressor: MAE $44,759, RMSE $56,230, R² 0.364 (ganador)
  - LinearRegression: MAE $44,797, RMSE $57,439, R² 0.336
  - RandomForestRegressor: MAE $46,887, RMSE $59,194, R² 0.295
- [x] **Justificacion de incluir los tres**: LinearRegression sirve como baseline interpretable; RandomForest captura no-linealidades y provee feature importance; GradientBoosting optimiza secuencialmente los residuos y suele lograr el menor error. Se entrenan los tres y se promueve el mejor por MAE, sin fijar un modelo a priori.
- [x] **Hiperparametros**: RF (`n_estimators=30`, `max_depth=8`, `min_samples_split=10`), GB (`n_estimators=30`, `max_depth=3`, `learning_rate=0.1`). Todos con `random_state=42`. Se loguean en MLflow.
- [x] **Trade-off de tiempo**: la evaluacion del proyecto recae en las decisiones arquitectonicas, no en la metrica del modelo. Por eso se priorizo tiempo de entrenamiento sobre precision: se redujeron los hiperparametros (de `n_estimators=200`/`max_depth=15-8` a `30`/`8-3`) y se acota el conjunto de entrenamiento con `TRAIN_MAX_ROWS` (default 5000, `0` = sin limite). Medicion previa con la config pesada: GB ~13 min vs RF ~1.5 min sobre 15k filas.

---

## Gotchas y problemas encontrados

> Registrar aqui cualquier problema inesperado, workaround o cosa que costo tiempo resolver.

### Fase 0 — Preparacion

- API del profesor: `cristiandiaz13/mlops-puj:data-api-pf-v1` expuesta en puerto 80
- Endpoints disponibles:
  - `GET /data?group_number=N` — devuelve un lote JSON con `{group_number, batch_number, data: [...]}`
  - `GET /restart_data_generation?group_number=N` — reinicia el contador de lotes para el grupo
  - `GET /health` — healthcheck (`{"status":"OK"}`)
- Cada llamada a `/data` incrementa `batch_number` automaticamente (0, 1, 2, ...)
- El tamanio del lote varia entre llamadas (observados: 73784, 94551, 230366 registros)
- Los datos NO tienen nulos en los lotes observados
- Columnas (12): `brokered_by` (float), `status` (str: "for_sale"/"sold"), `price` (float, target), `bed` (float), `bath` (float), `acre_lot` (float), `street` (float), `city` (str), `state` (str), `zip_code` (float), `house_size` (float), `prev_sold_date` (str, formato YYYY-MM-DD)
- `brokered_by` y `street` son IDs numericos codificados como float, no features interpretables
- `price` rango observado: 1.0 — 300000.0, media ~219k
- `status` valores observados: "for_sale", "sold"
- 50 estados distintos (incluye Puerto Rico)

### Fase 1 — Foundations

- El cluster microk8s necesita los addons `hostpath-storage` y `dns` habilitados antes de aplicar manifiestos. Sin `hostpath-storage` los PVCs quedan en `Pending`; sin `dns` ningun pod resuelve nombres de servicio.
- Los PVCs requieren `storageClassName: microk8s-hostpath` explicitamente — sin StorageClass default, el provisioner no los satisface.
- Si la IP del nodo cambia (ej. por reinicio de red o cambio de equipo), los certificados TLS del kubelet se invalidan. Solucion: `sudo microk8s refresh-certs --cert ca.crt` + restart.
- MLflow entra en crashloop si DNS no esta disponible al arrancar — sus reintentos de conexion a `postgres-service` se agotan. Solucion: habilitar DNS primero y luego `kubectl rollout restart deployment/mlflow`.
- El init SQL de Postgres (`02-schemas.sql` en ConfigMap `postgres-init`) solo se ejecuta en la primera inicializacion del PVC. Si se agregan esquemas nuevos despues, hay que crearlos manualmente con `kubectl exec` o via las migraciones Python.
- La API de datos del profesor NO se despliega dentro del cluster — se corre externamente con `docker run` para simular una fuente de datos externa, como pide el enunciado (seccion 6.1).

### Fase 2 — Pipeline

- MLflow client 3.x es incompatible con MLflow server 2.13 (endpoint `/api/2.0/mlflow/logged-models` no existe en el servidor). Solucion: fijar `mlflow==2.13.0` en `requirements.txt` para que cliente y servidor coincidan.
- El lote de la API puede traer hasta 230k registros pero `MAX_BATCH_SIZE=15000` los recorta. Esto es intencional (requisito del enunciado), pero significa que un lote de la API puede necesitar varias ejecuciones del DAG para consumirse completamente.
- GradientBoostingRegressor tarda ~13 min en entrenar con 12k filas (vs ~1.5 min RandomForest, ~1s LinearRegression). En lotes mas grandes puede ser un cuello de botella.
- Los 3 modelos entrenados son regressors (LinearRegression, RandomForestRegressor, GradientBoostingRegressor). En la prueba, GradientBoosting gano con MAE $44.8k y R² 0.36; LinearRegression fue segundo con MAE $44.8k y R² 0.34; RandomForest tercero con MAE $46.9k y R² 0.30.
- El pipeline `all` del CLI integra auditoria automaticamente: crea una entrada en `audit.batch_log` al inicio y la va actualizando en cada etapa (quality, decide, train, promote). Si no se entrena, se marca como `completed` sin pasar por train/promote.
- `prev_sold_date` se convierte a `prev_sold_year` (int) sin problemas en los lotes observados. El formato es consistente `YYYY-MM-DD` — el placeholder del borrador anterior sobre "formato inconsistente" no se confirmo.

### Fase 3 — Airflow

- El DAG usa `@task.branch()` (TaskFlow API de Airflow 3.x) para las bifurcaciones `decide_training` y `decide_promotion`. Retorna el `task_id` como string para indicar que rama seguir.
- Las tareas posteriores a las bifurcaciones (`notify_or_log_result`, `end`) usan `trigger_rule="none_failed_min_one_success"` para ejecutarse independientemente de que rama se tomo.
- Los datos entre tareas se pasan via XCom (dicts serializados). Los lotes crudos NO pasan por XCom — se almacenan en Postgres y las tareas downstream leen por `batch_id`.
- `DATA_API_URL` apunta a la IP del host (`192.168.2.6:8000`) porque la API del profesor corre fuera del cluster. `host.docker.internal` no funciona en microk8s.
- La imagen de Airflow se construye localmente (`docker build -f airflow/Dockerfile .`) y se importa a microk8s con `docker save | sudo microk8s ctr image import -`. Se usa `pullPolicy: Never` en el chart.
- El Dockerfile ya no copia `data/Diabetes.csv` — la ingesta es via HTTP.
- `airflow/requirements.txt` fija `mlflow==2.13.0` para coincidir con el servidor.
- La tarea `train_candidate_model` incluye el split internamente (llama `split.run()` + `train.run()`) porque el split solo tiene sentido si se va a entrenar.
- `setuptools>=82` elimina `pkg_resources` del paquete. MLflow 2.13 lo requiere. Solucion: instalar `setuptools<81` en el Dockerfile de Airflow antes de las demas dependencias.
- La API del profesor devuelve **400 Bad Request** cuando se agotan los lotes del grupo. Solucion: reiniciar con `GET /restart_data_generation?group_number=N` antes de trigger el DAG.
- DAG verificado end-to-end en Airflow UI: 19 tareas, bifurcaciones visibles (verde = ejecutado, rosa = skipped). Caso probado: primer lote → decide entrenar → entrena 3 modelos → promueve GradientBoosting como champion.

### Fase 4 — API FastAPI

- MLflow exige que las features numericas sean `float64`. Si el cliente envia `int` (ej. `"bed": 3`), la prediccion falla con `Can not safely convert int64 to float64`. Solucion: se agrego cast automatico `int→float64` en `_predict_with_model()` antes de llamar `model.predict()`.
- El modelo fue entrenado en Python 3.12 (imagen Airflow) pero la API corre en Python 3.11. MLflow muestra un warning pero la prediccion funciona correctamente.
- La API carga el modelo `real-estate-regressor@champion` al arrancar. Si no hay champion (primer despliegue sin haber entrenado), el startup loguea warning pero la API sigue respondiendo `/health`. La primera llamada a `/predict` devuelve 503.
- Se elimino el campo `score` de `PredictResponse` y `db.log_inference()` — no aplica en regresion (no hay probabilidades de clase).
- `prediction` cambio de `int` a `float` en schemas, db y main — el precio predicho es un valor continuo en dolares.
- Endpoints verificados en cluster: `/health` (200), `/model-info` (nombre, version, alias, TTL), `/predict` (precio predicho $253k para casa de ejemplo), `/reload-model` (recarga sin redespliegue).
- Port-forwards zombis son un problema frecuente — `pkill -f "kubectl port-forward"` antes de abrir nuevos.

### Fase 5 — Streamlit

- La UI tiene dos secciones (seleccionables en sidebar): "Inferencia" para predecir precios y "Historial de entrenamiento" para ver la tabla `audit.batch_log`.
- El historial consulta Postgres directamente via `psycopg2` (no pasa por la API FastAPI). Se agrego `PG_DSN` al ConfigMap de la UI para este proposito.
- Se incluyen 3 payloads de ejemplo (casa mediana, grande, pequena) para probar rapidamente sin llenar el formulario manualmente.
- Cada entrada del historial muestra: filas recibidas/unicas, validaciones (esquema, calidad, drift), decision de entrenamiento con razon, MLflow run ID, resultado de promocion, y cambio de MAE entre champion anterior y nuevo.

### Fase 6 — GitHub Actions

- 4 workflows creados, uno por componente: `build-api.yml`, `build-ui.yml`, `build-mlflow.yml`, `build-airflow.yml`.
- Cada workflow se activa solo cuando cambian archivos relevantes (path filters): `api/**`, `ui/**`, `docker/mlflow/**`, `airflow/**` + `pipeline/**`.
- Tagging: cada push genera dos tags — `sha-<commit_short>` (trazabilidad) + `latest` (etiqueta estable). En PRs solo hace build sin push.
- Imagenes se publican en DockerHub bajo `camilogarcia06/mlops-*`.
- Manifiestos K8s actualizados de `pullPolicy: Never` (imagen local) a `pullPolicy: Always` con `camilogarcia06/mlops-*:latest`.
- Secretos `DOCKERHUB_USERNAME` (`camilogarcia06`) y `DOCKERHUB_TOKEN` configurados en GitHub Settings > Secrets > Actions.
- Verificado: 3 workflows ejecutados exitosamente en push (API 1m40s, UI 1m1s, Airflow 2m0s). MLflow no se disparo porque no cambiaron archivos en `docker/mlflow/`.

### Fase 7 — Argo CD

- Instalacion requiere `--server-side --force-conflicts` porque el CRD de ApplicationSets excede el limite de 262KB de anotaciones de `kubectl apply` clasico.
- 4 Applications creadas en `k8s/argocd/applications.yaml`: foundations, mlflow, api, ui. Todas con `selfHeal: true` y `prune: true`.
- La contrasena de admin se cambia parcheando el secret `argocd-secret` con un hash bcrypt generado con Python (`bcrypt.hashpw`). Requiere restart del deployment `argocd-server`.
- MLflow quedo en Degraded hasta que se publico la imagen en DockerHub (el workflow no se habia disparado). Se forzo el build con un cambio minimo en el Dockerfile y luego `kubectl rollout restart` para salir de ImagePullBackOff.
- Argo CD UI accesible en `https://localhost:8443` via port-forward al svc `argocd-server` en namespace `argocd`. Login: admin/admin.

### Fase 8 — Observabilidad

- Locustfile actualizado con payload de propiedades (bed, bath, acre_lot, house_size, city, state, status, zip_code, prev_sold_year). Features numericas se randomzan para variar cada request.
- ConfigMap de Locust (`k8s/locust/configmap.yaml`) sincronizado con el locustfile local.
- PVC de Prometheus necesita `storageClassName: microk8s-hostpath` (misma correccion que Fase 1).
- Prometheus scrappea la API correctamente (target `api` → `up`, intervalo 10s).
- Puerto 8080 conflictuaba con CVAT (otro proyecto). Se cambio el port-forward de Airflow a 8082 y el de la API a 8001 para evitar conflictos.
- Observabilidad verificada: Locust genera carga, Prometheus recolecta metricas, Grafana las visualiza.

---

## Tecnicas y patrones utilizados

### Ingesta incremental

- [x] **Patron de consumo**: cada ejecucion del DAG hace una sola llamada a `GET /data?group_number=N`. La API incrementa `batch_number` automaticamente. El pipeline recibe el JSON completo `{group_number, batch_number, data: [...]}`, extrae `batch_number` como `batch_id`.
- [x] **Reintentos**: el cliente HTTP usa `requests` con retry (3 intentos, backoff exponencial) y timeout de 120s. Si la API devuelve error HTTP o respuesta vacia, el DAG falla con estado claro, no oculta el error.
- [x] **Idempotencia**: cada fila se inserta en `raw.properties_raw` con `row_hash` (SHA-256 del payload JSON serializado) como PK. Si el DAG se reejecuta con el mismo lote, los INSERT se ignoran por conflicto de PK (`ON CONFLICT DO NOTHING`). El `batch_id` permite auditar que filas pertenecen a cada lote.
- [x] **Procesamiento incremental (preprocess + split)**: para que el tiempo por corrida sea constante y no crezca con el historico acumulado, cada etapa toca solo el lote nuevo. `preprocess` lee de `raw` unicamente las filas con `status='loaded'` y, en la misma transaccion que el upsert a `clean`, las marca `status='processed'` (asi no se vuelven a leer). `split` solo particiona las filas con `split IS NULL`; las ya asignadas conservan su particion (cada lote mantiene su propio ~80/20, que agregado da ~80/20 global). Antes ambas etapas reprocesaban toda la tabla en cada corrida — costo O(total acumulado), cuadratico en el numero de lotes — y eran la causa de que cada ejecucion del DAG tardara mas que la anterior. Backfill unico al desplegar: `UPDATE raw.properties_raw SET status='processed' WHERE row_hash IN (SELECT row_hash FROM clean.properties_clean)`.

### Validacion de datos

- [x] **Tests de esquema**: se validan las 12 columnas esperadas (nombre y tipo). Se detectan columnas nuevas, faltantes o con tipo de dato incorrecto. Si el esquema falla, el lote se marca como `schema_ok=FALSE` en `audit.batch_log` pero el pipeline continua (el lote se almacena en raw para auditoria).
- [x] **Tests de calidad**: nulos por columna, duplicados exactos (por `row_hash`), rangos invalidos (`price <= 0`, `bed < 0`, `bath < 0`, `house_size <= 0`, `acre_lot < 0`), valores extremos en `price` (> 10M). Se reporta el porcentaje de filas con problemas.
- [x] **Deteccion de drift**: test de Kolmogorov-Smirnov (scipy `ks_2samp`) comparando la distribucion del lote nuevo vs el historico en `clean` para features numericas (`price`, `house_size`, `acre_lot`, `bed`, `bath`). Se considera drift significativo si p-value < 0.05 en al menos 2 features.
- [x] **Deteccion de nuevas categorias**: para `city`, `state`, `status` y `zip_code`, se comparan los valores unicos del lote nuevo contra los valores ya presentes en `clean.properties_clean`. Se reporta la cantidad y proporcion de categorias nuevas.

### Entrenamiento y MLflow

- [x] **Parametros logueados**: `model_type`, `n_estimators`, `max_depth`, `min_samples_split`, `random_state`, `n_features`, `n_training_rows`, `batch_ids_used` (lista de lotes incluidos en el entrenamiento).
- [x] **Metricas logueadas**: `mae`, `rmse`, `r2` (train y test splits). La metrica primaria es `test_mae`.
- [x] **Artefactos**: modelo serializado (MLflow sklearn flavor), grafico de residuales (`residuals.png`), feature importance (`feature_importance.png`), reporte de metricas (`metrics.json`).
- [x] **Tags**: `training_reason` (por que se decidio entrenar), `commit_sha` (version del codigo), `batch_id` (lote que disparo el entrenamiento), `pipeline_version`.
- [x] **Serializacion**: `mlflow.sklearn.log_model()` con firma inferida (`mlflow.models.infer_signature`). La API carga el modelo via `mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")`.
- [x] **Versionamiento de codigo**: se loguea el commit SHA actual como tag `commit_sha` en el run de MLflow. Esto permite reconstruir que version del codigo genero cada modelo.

### Promocion automatica

- [x] **Flujo de comparacion**: despues de entrenar y registrar el candidato en MLflow, se consulta el modelo con alias `champion`. Se comparan `mae_test` y `rmse_test` del candidato contra los del champion. La decision se basa en la regla explicita: MAE baja >= 3% y RMSE no empeora > 1%.
- [x] **Actualizacion del alias**: si el candidato supera la regla, se usa `client.set_registered_model_alias(model_name, "champion", new_version)`. El alias `champion` apunta a la nueva version; la version anterior queda registrada pero sin alias.
- [x] **Sin champion previo**: si no existe un modelo con alias `champion` (primer entrenamiento), el candidato se promueve automaticamente. Esto se detecta con un try/except al intentar cargar el champion actual.

### Inferencia

- [x] **Carga del modelo**: la API FastAPI carga el modelo desde MLflow usando `mlflow.pyfunc.load_model(f"models:/{model_name}@champion")`. Se cachea en memoria con TTL de 300s (5 min). Durante el TTL, las peticiones usan el modelo en cache sin consultar MLflow.
- [x] **Recarga sin redespliegue**: endpoint `POST /reload-model` fuerza la recarga inmediata del modelo desde MLflow, sin reiniciar el pod. Tambien se puede esperar a que el TTL expire naturalmente (max 5 min de latencia de propagacion).
- [x] **Concurrencia durante recarga**: se usa un patron de swap atomico — se carga el modelo nuevo en una variable temporal y solo se reemplaza la referencia en cache cuando la carga es exitosa. Si la carga falla, se mantiene el modelo anterior y se loguea el error. Las peticiones en vuelo no se interrumpen.
- [x] **Registro en `inference.predictions`**: cada llamada a `/predict` inserta una fila con: `request_id` (UUID), `created_at`, `input_payload` (JSONB con las features recibidas), `prediction` (precio estimado como DOUBLE PRECISION), `model_name`, `model_version`, `latency_ms`.

### GitOps

- [x] **Flujo CI/CD completo**: push a Git → GitHub Actions detecta cambios en paths relevantes → build imagen Docker con tag `sha-<commit>` + `latest` → push a DockerHub → actualizar tag de imagen en manifiesto K8s → Argo CD detecta el cambio en Git → sincroniza el manifiesto al cluster → Kubernetes aplica el rolling update.
- [x] **Argo CD Applications**: una Application por componente (api, ui, mlflow, airflow) apuntando al directorio `k8s/<componente>/` del repo. Sincronizacion automatica habilitada con `selfHeal: true` y `prune: true`. No se usa `kubectl apply` como mecanismo principal de despliegue.

### Observabilidad

- [x] **Metricas expuestas por la API** (endpoint `/metrics`, formato Prometheus):
  - `api_requests_total` (counter, labels: endpoint, method, status)
  - `api_request_duration_seconds` (histogram, labels: endpoint)
  - `api_prediction_errors_total` (counter)
  - `api_model_info` (gauge, labels: model_name, model_version)
- [x] **Prometheus scrape**: configurado via `ServiceMonitor` o anotaciones en el Service de la API (`prometheus.io/scrape: "true"`, `prometheus.io/port: "8000"`, `prometheus.io/path: "/metrics"`). Intervalo de scrape: 15s.
- [x] **Dashboard de Grafana**: JSON versionado en `observability/grafana-dashboard.json`, montado via ConfigMap. Paneles: tasa de peticiones (RPS), latencia p50/p95/p99, tasa de errores, version del modelo activo, uso de recursos del pod.
- [ ] Resultados de la prueba de carga con Locust (p50, p95, p99, RPS, error rate) — se completa en Fase 8.

---

## Hallazgos por fase

> Seccion para registrar desvios respecto al enunciado o decisiones no triviales que
> el evaluador necesita entender.

| Fase | Hallazgo / Desvio | Justificacion |
|------|-------------------|---------------|
| 0 | Lotes de tamanio variable (73k-230k), no fijo | La API del profesor no garantiza tamanio constante; el pipeline debe ser robusto ante variaciones |
| 0 | `brokered_by` y `street` son IDs numericos, no features interpretables | Se eliminan del feature set para evitar ruido y dimensionalidad excesiva |
| 0 | 50 estados (incluye Puerto Rico) | El encoder debe manejar todos los estados de EEUU + territorios |
| 0 | Precio minimo observado = $1.0 | Indica posibles datos invalidos; se filtran precios < umbral en calidad |
| 0 | Los datos no tienen nulos en los lotes observados | No se puede asumir que todos los lotes seran libres de nulos; el pipeline debe manejarlos |
| 1 | microk8s requiere addons `hostpath-storage` y `dns` | Sin ellos los PVCs no se provisionan y los pods no resuelven DNS interno |
| 1 | PVCs necesitan `storageClassName: microk8s-hostpath` explicito | El cluster no tiene StorageClass default; sin esto quedan en Pending indefinidamente |
| 1 | Cambio de IP del nodo invalida certs TLS del kubelet | Solucion: `microk8s refresh-certs --cert ca.crt` + restart del cluster |
| 1 | API del profesor se corre fuera del cluster (Docker local) | Simula fuente externa de datos como pide el enunciado (seccion 6.1); no se despliega en K8s |
| 2 | MLflow client 3.x incompatible con server 2.13 | Fijar `mlflow==2.13.0` en requirements para coincidir con la imagen del server en K8s |
| 2 | GradientBoosting tarda ~13 min en 12k filas | Considerar excluirlo o reducir `n_estimators` si los lotes crecen significativamente |
| 2 | Los 3 modelos son regressors validos para prediccion de precios | GradientBoosting gano (MAE $44.8k, R² 0.36), seguido de LinearRegression y RandomForest |
| 2 | `prev_sold_date` formato consistente YYYY-MM-DD | No se confirmo el problema de formato inconsistente anticipado en el borrador inicial |
| 3 | `host.docker.internal` no funciona en microk8s | Usar IP del host directamente en `DATA_API_URL` (ej. `192.168.2.6:8000`) |
| 3 | Imagen Airflow se importa manualmente a microk8s | `docker save \| sudo microk8s ctr image import -` con `pullPolicy: Never` |
| 3 | MLflow client en imagen Airflow debe ser 2.13.0 | Misma restriccion que en Fase 2; se fija en `airflow/requirements.txt` |
| 3 | `setuptools>=82` elimina `pkg_resources` | MLflow 2.13 lo necesita; instalar `setuptools<81` en Dockerfile |
| 3 | API devuelve 400 cuando se agotan los lotes | Reiniciar con `GET /restart_data_generation?group_number=N` antes de trigger |
| 3 | DAG verificado end-to-end con 19 tareas | Bifurcaciones funcionan: decide_training→train, decide_promotion→promote; skip_training y reject_model quedan skipped |
| 4 | MLflow exige float64 en features numericas | Cast automatico int→float64 en `_predict_with_model()` para evitar error del cliente |
| 4 | Modelo entrenado en Python 3.12, API corre en 3.11 | MLflow muestra warning pero funciona correctamente |
| 4 | Campo `score` eliminado de la respuesta | No aplica en regresion; `prediction` es float (precio en dolares) |
| 5 | UI consulta Postgres directamente para historial | No pasa por la API; se agrega `PG_DSN` al ConfigMap de la UI |
| 6 | Workflows usan path filters para no rebuilds innecesarios | Solo se build la imagen del componente cuyos archivos cambiaron |
| 6 | Secretos DOCKERHUB_USERNAME y DOCKERHUB_TOKEN pendientes | Configurar en GitHub Settings > Secrets > Actions antes del primer push |
| 7 | CRD de ApplicationSets excede limite de anotaciones | Usar `--server-side --force-conflicts` al instalar Argo CD |
| 7 | MLflow Degraded hasta publicar imagen en DockerHub | El workflow no se dispara si no cambian archivos en `docker/mlflow/`; forzar con cambio minimo |
| 7 | 4 apps Healthy + Synced verificadas | foundations, mlflow, api, ui — todas sincronizadas desde Git |

---

## Esquemas SQL

### `raw.properties_raw`
| Columna | Tipo | Descripcion |
|---------|------|-------------|
| `row_hash` | TEXT PK | SHA-256 del payload JSON, garantiza idempotencia |
| `batch_id` | TEXT NOT NULL | Identificador del lote (batch_number de la API) |
| `load_timestamp` | TIMESTAMPTZ | Fecha de ingesta |
| `source` | TEXT | Origen del dato (default `'api'`) |
| `status` | TEXT | Estado del registro (default `'loaded'`) |
| `payload` | JSONB | Fila original tal cual llego de la API |

### `clean.properties_clean`
| Columna | Tipo | Descripcion |
|---------|------|-------------|
| `id` | BIGSERIAL PK | ID autoincremental |
| `row_hash` | TEXT UNIQUE FK | Referencia a `raw.properties_raw` (trazabilidad) |
| `batch_id` | TEXT | Lote al que pertenece |
| `processed_at` | TIMESTAMPTZ | Fecha de procesamiento |
| `split` | TEXT | Particion: `train`, `test`, o NULL |
| `features` | JSONB | Features transformadas listas para el modelo |
| `target` | DOUBLE PRECISION | Precio objetivo (`price`) |

### `audit.batch_log`
| Columna | Tipo | Descripcion |
|---------|------|-------------|
| `id` | BIGSERIAL PK | ID autoincremental |
| `batch_id` | TEXT | Lote procesado |
| `run_timestamp` | TIMESTAMPTZ | Fecha de ejecucion |
| `rows_received` | INT | Registros recibidos de la API |
| `rows_after_dedup` | INT | Registros despues de deduplicacion |
| `schema_ok` | BOOLEAN | Validacion de esquema paso? |
| `quality_ok` | BOOLEAN | Validacion de calidad paso? |
| `drift_detected` | BOOLEAN | Se detecto data drift? |
| `training_decision` | BOOLEAN | Se decidio entrenar? |
| `training_reason` | TEXT | Razon de la decision |
| `mlflow_run_id` | TEXT | ID del run en MLflow (si entreno) |
| `model_registered` | BOOLEAN | Se registro modelo en MLflow? |
| `promotion_decision` | BOOLEAN | Se promovio el modelo? |
| `promotion_reason` | TEXT | Razon de promocion/rechazo |
| `champion_metric_before` | DOUBLE PRECISION | MAE del champion antes |
| `champion_metric_after` | DOUBLE PRECISION | MAE del nuevo champion |
| `status` | TEXT | Estado final: `started`, `completed`, `failed` |

### `inference.predictions`
| Columna | Tipo | Descripcion |
|---------|------|-------------|
| `request_id` | UUID PK | ID unico de la peticion |
| `created_at` | TIMESTAMPTZ | Fecha de la inferencia |
| `input_payload` | JSONB | Features enviadas por el usuario |
| `prediction` | DOUBLE PRECISION | Precio predicho |
| `model_name` | TEXT | Nombre del modelo usado |
| `model_version` | TEXT | Version del modelo en MLflow |
| `latency_ms` | DOUBLE PRECISION | Tiempo de respuesta en ms |

---

## Notas sueltas

> Espacio libre para apuntes rapidos que luego se clasifican arriba.

- Grupo asignado: pendiente confirmar con el profesor (se usa `group_number=1` por defecto)
- La API del profesor NO expone endpoint para consultar cuantos lotes hay disponibles; hay que consumir hasta que devuelva respuesta vacia o error
