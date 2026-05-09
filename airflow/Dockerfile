ARG AIRFLOW_BASE_TAG=3.1.8
FROM apache/airflow:${AIRFLOW_BASE_TAG}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

USER root
COPY requirements.txt /requirements.txt
RUN uv pip install --no-cache-dir --system -r /requirements.txt
USER airflow

COPY dags/ /opt/airflow/dags/
