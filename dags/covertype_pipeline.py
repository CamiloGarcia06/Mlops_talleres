"""
DAG: covertype_pipeline
========================
Se ejecuta cada 5 minutos. Obtiene datos de la API y los almacena en PostgreSQL.
"""

import os
import logging
from datetime import timedelta

import requests
import pandas as pd
from sqlalchemy import create_engine

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

API_URL = os.environ.get("COVERTYPE_API_URL", "http://localhost:8090")
GROUP_NUMBER = int(os.environ.get("GROUP_NUMBER", "2"))

COLUMN_NAMES = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    "Wilderness_Area", "Soil_Type", "Cover_Type",
]


def _get_engine():
    host = os.environ["DATA_DB_HOST"]
    port = os.environ["DATA_DB_PORT"]
    user = os.environ["DATA_DB_USER"]
    password = os.environ["DATA_DB_PASSWORD"]
    dbname = os.environ["DATA_DB_NAME"]
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")


def fetch_data(**context):
    resp = requests.get(
        f"{API_URL}/data",
        params={"group_number": GROUP_NUMBER},
        timeout=120,
    )
    resp.raise_for_status()
    payload = resp.json()

    batch_number = payload["batch_number"]
    rows = payload["data"]
    logger.info("Batch %d recibido: %d filas", batch_number, len(rows))

    df = pd.DataFrame(rows, columns=COLUMN_NAMES)
    df["batch"] = batch_number

    engine = _get_engine()
    df.to_sql("covertype_raw", engine, if_exists="append", index=False)

    logger.info("Guardado en covertype_raw (batch=%d, filas=%d)", batch_number, len(df))


with DAG(
    dag_id="covertype_pipeline",
    schedule=timedelta(minutes=5),
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "covertype", "proyecto2"],
) as dag:
    t_fetch = PythonOperator(task_id="fetch_data", python_callable=fetch_data)
