# Build context = repo root: docker build -f airflow/Dockerfile -t <user>/mlops-airflow:<tag> .
ARG AIRFLOW_BASE_TAG=3.1.8
FROM apache/airflow:${AIRFLOW_BASE_TAG}

USER root
COPY airflow/requirements.txt /requirements.txt
COPY pipeline/ /opt/airflow/pipeline/
ENV PYTHONPATH=/opt/airflow

USER airflow
RUN pip install --no-cache-dir -r /requirements.txt

COPY airflow/dags/ /opt/airflow/dags/
COPY data/data/Diabetes.csv /opt/airflow/data/Diabetes.csv
