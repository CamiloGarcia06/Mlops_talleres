NS := mlops

.PHONY: up down forward stop-forward status data-api data-restart data-stop build-images import-images migrate trigger-dag

# ============================================================================
# Levantar todo el cluster
# ============================================================================
up:
	# 1. Foundations (namespace, postgres, minio)
	kubectl apply -k k8s/foundations
	@echo ">>> Esperando Postgres y MinIO..."
	kubectl -n $(NS) wait --for=condition=ready pod -l app=postgres --timeout=180s
	kubectl -n $(NS) wait --for=condition=ready pod -l app=minio --timeout=180s
	# 2. MLflow
	kubectl apply -k k8s/mlflow
	@sleep 5
	kubectl -n $(NS) wait --for=condition=ready pod -l app=mlflow --timeout=120s
	# 3. Airflow (Helm)
	kubectl apply -n $(NS) -f k8s/airflow/secret.yaml
	helm repo add apache-airflow https://airflow.apache.org >/dev/null 2>&1 || true
	helm repo update apache-airflow >/dev/null 2>&1
	helm upgrade --install airflow apache-airflow/airflow -n $(NS) -f airflow/values/values-local.yaml --timeout 20m
	# 4. API + UI
	kubectl apply -k k8s/api
	kubectl apply -k k8s/ui
	# 5. Observabilidad
	kubectl apply -k k8s/prometheus
	kubectl apply -k k8s/grafana
	kubectl apply -k k8s/locust
	# 6. Argo CD Applications
	kubectl apply -f k8s/argocd/applications.yaml 2>/dev/null || true
	@echo ""
	@echo ">>> Cluster listo. Ejecuta: make forward"

# ============================================================================
# Tumbar todo
# ============================================================================
down:
	-helm uninstall airflow -n $(NS) 2>/dev/null
	-kubectl delete -k k8s/locust --ignore-not-found
	-kubectl delete -k k8s/grafana --ignore-not-found
	-kubectl delete -k k8s/prometheus --ignore-not-found
	-kubectl delete -k k8s/ui --ignore-not-found
	-kubectl delete -k k8s/api --ignore-not-found
	-kubectl delete -k k8s/mlflow --ignore-not-found
	-kubectl delete -k k8s/foundations --ignore-not-found
	-kubectl delete -f k8s/argocd/applications.yaml --ignore-not-found 2>/dev/null || true

# ============================================================================
# Port-forwards (mata los anteriores antes de abrir nuevos)
# ============================================================================
forward:
	@rm -f /tmp/pf.pids
	@kubectl -n $(NS) port-forward svc/airflow-api-server 8082:8080 >/dev/null 2>&1 & echo $$! >> /tmp/pf.pids
	@kubectl -n $(NS) port-forward svc/mlflow-service     5000:5000 >/dev/null 2>&1 & echo $$! >> /tmp/pf.pids
	@kubectl -n $(NS) port-forward svc/minio-service      9001:9001 >/dev/null 2>&1 & echo $$! >> /tmp/pf.pids
	@kubectl -n $(NS) port-forward svc/minio-service      9000:9000 >/dev/null 2>&1 & echo $$! >> /tmp/pf.pids
	@kubectl -n $(NS) port-forward svc/api                8001:8000 >/dev/null 2>&1 & echo $$! >> /tmp/pf.pids
	@kubectl -n $(NS) port-forward svc/ui                 8501:8501 >/dev/null 2>&1 & echo $$! >> /tmp/pf.pids
	@kubectl -n $(NS) port-forward svc/prometheus         9090:9090 >/dev/null 2>&1 & echo $$! >> /tmp/pf.pids
	@kubectl -n $(NS) port-forward svc/grafana            3000:3000 >/dev/null 2>&1 & echo $$! >> /tmp/pf.pids
	@kubectl -n $(NS) port-forward svc/locust-master      8089:8089 >/dev/null 2>&1 & echo $$! >> /tmp/pf.pids
	@kubectl -n argocd port-forward svc/argocd-server     8443:443  >/dev/null 2>&1 & echo $$! >> /tmp/pf.pids
	@sleep 2
	@echo ""
	@echo "╔══════════════╦════════════════════════════════╦═════════╦════════════╗"
	@echo "║ Servicio     ║ URL                            ║ UI      ║ Login      ║"
	@echo "╠══════════════╬════════════════════════════════╬═════════╬════════════╣"
	@echo "║ Airflow      ║ http://localhost:8082           ║ Si      ║ admin/admin║"
	@echo "║ MLflow       ║ http://localhost:5000           ║ Si      ║ -          ║"
	@echo "║ MinIO        ║ http://localhost:9001           ║ Si      ║ minioadmin ║"
	@echo "║ API (Swagger)║ http://localhost:8001/docs      ║ Si      ║ -          ║"
	@echo "║ UI Streamlit ║ http://localhost:8501           ║ Si      ║ -          ║"
	@echo "║ Prometheus   ║ http://localhost:9090           ║ Si      ║ -          ║"
	@echo "║ Grafana      ║ http://localhost:3000           ║ Si      ║ admin/admin║"
	@echo "║ Locust       ║ http://localhost:8089           ║ Si      ║ -          ║"
	@echo "║ Argo CD      ║ https://localhost:8443          ║ Si      ║ admin/admin║"
	@echo "║ Data API     ║ http://localhost:8000 (docker)  ║ No      ║ -          ║"
	@echo "╚══════════════╩════════════════════════════════╩═════════╩════════════╝"
	@echo ""
	@echo "Para detener: make stop-forward"

stop-forward:
	@if [ -f /tmp/pf.pids ]; then xargs kill < /tmp/pf.pids 2>/dev/null; fi
	@rm -f /tmp/pf.pids
	@echo "Port-forwards detenidos."

# ============================================================================
# Estado del cluster
# ============================================================================
status:
	@echo "=== Pods ==="
	@kubectl -n $(NS) get pods
	@echo ""
	@echo "=== Services ==="
	@kubectl -n $(NS) get svc
	@echo ""
	@echo "=== Argo CD ==="
	@kubectl get applications -n argocd 2>/dev/null || echo "Argo CD no instalado"

# ============================================================================
# API de datos del profesor (fuera del cluster)
# ============================================================================
data-api:
	docker run --rm -d -p 8000:80 --name data-api cristiandiaz13/mlops-puj:data-api-pf-v1
	@echo ">>> API del profesor corriendo en http://localhost:8000"

data-restart:
	@curl -s "http://localhost:8000/restart_data_generation?group_number=1"
	@echo ""
	@echo ">>> Datos del grupo 1 reiniciados"

data-stop:
	@docker stop data-api 2>/dev/null || true
	@echo ">>> API del profesor detenida"

# ============================================================================
# Trigger del DAG de Airflow
# ============================================================================
trigger-dag:
	@echo ">>> Disparando DAG properties_mlops_pipeline..."
	kubectl -n $(NS) exec deploy/airflow-api-server -- airflow dags unpause properties_mlops_pipeline
	kubectl -n $(NS) exec deploy/airflow-api-server -- airflow dags trigger properties_mlops_pipeline

# ============================================================================
# Build e importar imagenes locales a microk8s
# ============================================================================
build-images:
	docker build -f docker/api/Dockerfile -t mlops-api:local .
	docker build -f docker/ui/Dockerfile -t mlops-ui:local .
	docker build -f docker/mlflow/Dockerfile -t mlops-mlflow:local .
	docker build -f airflow/Dockerfile -t mlops-airflow:local .
	@echo ">>> 4 imagenes construidas"

import-images:
	docker save mlops-api:local | sudo microk8s ctr image import -
	docker save mlops-ui:local | sudo microk8s ctr image import -
	docker save mlops-mlflow:local | sudo microk8s ctr image import -
	docker save mlops-airflow:local | sudo microk8s ctr image import -
	@echo ">>> 4 imagenes importadas a microk8s"

# ============================================================================
# Migraciones del pipeline (requiere port-forward a Postgres)
# ============================================================================
migrate:
	PG_DSN="postgresql://mlops_user:mlops_pass_2026@localhost:5432/mlops" python3 -m pipeline.cli migrate
