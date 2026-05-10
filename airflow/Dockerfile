# Build context = repo root: docker build -f airflow/Dockerfile -t <user>/mlops-airflow:<tag> .
ARG AIRFLOW_BASE_TAG=3.1.8
FROM apache/airflow:${AIRFLOW_BASE_TAG}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

USER root
COPY airflow/requirements.txt /requirements.txt
RUN uv pip install --no-cache-dir --system -r /requirements.txt

# Project pipeline package made importable by every Airflow worker / scheduler.
COPY pipeline/ /opt/airflow/pipeline/
ENV PYTHONPATH=/opt/airflow

USER airflow

COPY airflow/dags/ /opt/airflow/dags/
