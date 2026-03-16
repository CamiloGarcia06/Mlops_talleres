"""
covertype_pipeline.py
---------------------
Pipeline de ingesta incremental del dataset Forest Cover Type.

Arquitectura:
    RAW  →  PROCESSED
    t1 (validate_infra) → t2 (extract_raw) → t3 (transform) → t4 (clean)

Flujo de datos:
    API externa → forest_raw (JSON blob) → forest_processed (columnar)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import timedelta

import pandas as pd
import psycopg2
import requests
from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

_DEFAULT_API_URL  = "http://host.docker.internal:8080"
_DEFAULT_GROUP_ID = 2

_DB_DEFAULTS = {
    "host":     "postgres",
    "port":     "5432",
    "user":     "airflow",
    "password": "airflow",
    "database": "project_data",
}

# Orden posicional exacto que devuelve la API (lista de listas sin headers).

COVERTYPE_COLUMNS: list[str] = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    "Wilderness_Area",
    "Soil_Type",
    "Cover_Type",
]

# ---------------------------------------------------------------------------
# Capa de infraestructura
# ---------------------------------------------------------------------------


class DataExtractor:
    """Cliente HTTP para la API externa de Forest Cover Type."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def fetch_batch(self, group_id: int) -> dict:
        """
        Descarga un batch de datos para el grupo indicado.

        Returns:
            dict con claves ``batch_number`` y ``data`` (lista de filas).

        Raises:
            AirflowException: ante cualquier error HTTP o de red.
        """
        url = f"{self.base_url}/data"
        try:
            response = requests.get(
                url,
                params={"group_number": group_id},
                timeout=20,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise AirflowException(f"Error al consultar la API ({url}): {exc}") from exc


class DatabaseManager:
    """Capa de acceso a PostgreSQL (psycopg2 + SQLAlchemy)."""

    def __init__(self) -> None:
        self.db_name = os.environ.get("DATA_DB_NAME", _DB_DEFAULTS["database"])
        self._cfg = {
            "host":     os.environ.get("DATA_DB_HOST",     _DB_DEFAULTS["host"]),
            "port":     os.environ.get("DATA_DB_PORT",     _DB_DEFAULTS["port"]),
            "user":     os.environ.get("DATA_DB_USER",     _DB_DEFAULTS["user"]),
            "password": os.environ.get("DATA_DB_PASSWORD", _DB_DEFAULTS["password"]),
            "database": self.db_name,
        }

    # ------------------------------------------------------------------
    # Métodos de bajo nivel
    # ------------------------------------------------------------------

    def execute(self, query: str, params: tuple | None = None) -> list | None:
        """
        Ejecuta una sentencia SQL con psycopg2.

        Returns:
            Lista de filas para SELECT, ``None`` para DML.
        """
        try:
            with psycopg2.connect(**self._cfg) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    if cur.description:
                        return cur.fetchall()
                    conn.commit()
        except psycopg2.Error as exc:
            raise AirflowException(f"Error en base de datos: {exc}") from exc
        return None

    def get_engine(self, *, system_db: bool = False):
        """
        Construye un engine de SQLAlchemy.

        Args:
            system_db: Si es ``True`` conecta a ``postgres`` (útil para
                       CREATE DATABASE); en caso contrario usa ``self.db_name``.
        """
        target_db = "postgres" if system_db else self.db_name
        c = self._cfg
        url = (
            f"postgresql+psycopg2://{c['user']}:{c['password']}"
            f"@{c['host']}:{c['port']}/{target_db}"
        )
        return create_engine(url)

    # ------------------------------------------------------------------
    # Operaciones de dominio
    # ------------------------------------------------------------------

    def ensure_database_exists(self) -> None:
        """Crea ``self.db_name`` si todavía no existe."""
        engine = self.get_engine(system_db=True)
        with engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": self.db_name},
            ).fetchone()
            if not exists:
                conn.execute(text(f"CREATE DATABASE {self.db_name}"))
                log.info("Base de datos '%s' creada.", self.db_name)

    def ensure_schema_exists(self) -> None:
        """
        Crea las tablas necesarias si no existen (idempotente).

        Debe ejecutarse una sola vez al inicio del pipeline, antes de
        cualquier operación de lectura o escritura.
        """
        ddl_forest_raw = """
            CREATE TABLE IF NOT EXISTS forest_raw (
                batch_id    INTEGER      PRIMARY KEY,
                data_json   JSONB        NOT NULL,
                ingested_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            )
        """
        ddl_forest_processed = """
            CREATE TABLE IF NOT EXISTS forest_processed (
                id                                  SERIAL       PRIMARY KEY,
                batch_id                            INTEGER      NOT NULL,
                "Elevation"                         INTEGER,
                "Aspect"                            INTEGER,
                "Slope"                             INTEGER,
                "Horizontal_Distance_To_Hydrology"  INTEGER,
                "Vertical_Distance_To_Hydrology"    INTEGER,
                "Horizontal_Distance_To_Roadways"   INTEGER,
                "Hillshade_9am"                     INTEGER,
                "Hillshade_Noon"                    INTEGER,
                "Hillshade_3pm"                     INTEGER,
                "Horizontal_Distance_To_Fire_Points" INTEGER,
                "Wilderness_Area"                   TEXT,
                "Soil_Type"                         TEXT,
                "Cover_Type"                        INTEGER,
                inserted_at                         TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            )
        """
        ddl_forest_training = """
            CREATE TABLE IF NOT EXISTS forest_training (
                id                                  SERIAL       PRIMARY KEY,
                batch_id                            INTEGER      NOT NULL,
                "Elevation"                         INTEGER,
                "Aspect"                            INTEGER,
                "Slope"                             INTEGER,
                "Horizontal_Distance_To_Hydrology"  INTEGER,
                "Vertical_Distance_To_Hydrology"    INTEGER,
                "Horizontal_Distance_To_Roadways"   INTEGER,
                "Hillshade_9am"                     INTEGER,
                "Hillshade_Noon"                    INTEGER,
                "Hillshade_3pm"                     INTEGER,
                "Horizontal_Distance_To_Fire_Points" INTEGER,
                "Wilderness_Area"                   TEXT,
                "Soil_Type"                         TEXT,
                "Cover_Type"                        INTEGER,
                inserted_at                         TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            )
        """
        self.execute(ddl_forest_raw)
        self.execute(ddl_forest_processed)
        self.execute(ddl_forest_training)
        log.info("Schema verificado: forest_raw, forest_processed y forest_training listas.")

    def save_raw_batch(self, batch_id: int, rows: list) -> bool:
        """
        Persiste un batch en la tabla ``forest_raw`` (capa RAW).

        Returns:
            ``True`` si se insertó, ``False`` si ya existía (ON CONFLICT).
        """
        query = """
            INSERT INTO forest_raw (batch_id, data_json)
            VALUES (%s, %s)
            ON CONFLICT (batch_id) DO NOTHING
        """
        self.execute(query, (batch_id, json.dumps(rows)))
        log.info("Batch %s persistido en forest_raw.", batch_id)
        return True

    def load_raw_batch(self, batch_id: int) -> list:
        """
        Recupera las filas JSON de un batch desde ``forest_raw``.

        Raises:
            AirflowException: si el batch no se encuentra.
        """
        result = self.execute(
            "SELECT data_json FROM forest_raw WHERE batch_id = %s",
            (batch_id,),
        )
        if not result:
            raise AirflowException(
                f"Batch {batch_id} no encontrado en forest_raw. "
                "Verifica que t2 completó exitosamente."
            )
        return result[0][0]


# ---------------------------------------------------------------------------
# Tareas del DAG
# ---------------------------------------------------------------------------


def task_validate_infra() -> None:
    """
    t1 · Verifica conectividad, crea la base de datos y provisiona el schema.

    Es el único punto de setup de infraestructura del pipeline. Al ser
    idempotente (IF NOT EXISTS), es seguro ejecutarlo en cada corrida sin
    riesgo de duplicar objetos.

    Orden de operaciones:
        1. Ping a la API externa.
        2. Crear la base de datos si no existe (conexión a ``postgres``).
        3. Crear las tablas RAW y PROCESSED si no existen.
    """
    api_url  = os.environ.get("COVERTYPE_API_URL", _DEFAULT_API_URL)
    group_id = int(os.environ.get("GROUP_NUMBER", _DEFAULT_GROUP_ID))

    # 1 — Verificar API
    extractor = DataExtractor(api_url)
    payload   = extractor.fetch_batch(group_id)
    batch_id  = payload.get("batch_number")
    row_count = len(payload.get("data", []))
    log.info("API OK — batch_number=%s, filas=%d", batch_id, row_count)

    # 2 — Crear base de datos si no existe
    db = DatabaseManager()
    db.ensure_database_exists()
    log.info("Base de datos OK — host=%s, db=%s", db._cfg["host"], db.db_name)

    # 3 — Provisionar schema (tablas)
    db.ensure_schema_exists()


def task_extract_to_raw(ti) -> None:
    """
    t2 · Descarga un batch y lo guarda en la capa RAW (``forest_raw``).

    Publica ``batch_id`` vía XCom para que t3 sepa qué procesar.
    """
    api_url  = os.environ.get("COVERTYPE_API_URL", _DEFAULT_API_URL)
    group_id = int(os.environ.get("GROUP_NUMBER", _DEFAULT_GROUP_ID))

    extractor = DataExtractor(api_url)
    db        = DatabaseManager()

    payload  = extractor.fetch_batch(group_id)
    batch_id = payload.get("batch_number")
    rows     = payload.get("data", [])

    if not rows:
        raise AirflowException(f"Batch {batch_id} llegó vacío desde la API.")

    db.save_raw_batch(batch_id, rows)

    ti.xcom_push(key="batch_id", value=batch_id)
    log.info("Extracción completada — batch_id=%s, filas=%d", batch_id, len(rows))


def task_transform_to_processed(ti) -> None:
    """
    t3 · Lee ``forest_raw``, transforma y escribe en ``forest_processed``.

    Lee el ``batch_id`` publicado por t2 vía XCom, garantizando que
    siempre procesa exactamente el batch que fue ingestado en la misma
    ejecución del DAG.
    """
    batch_id = ti.xcom_pull(task_ids="extract_raw", key="batch_id")
    if batch_id is None:
        raise AirflowException(
            "XCom 'batch_id' no encontrado. "
            "Asegúrate de que la tarea 'extract_raw' finalizó correctamente."
        )

    db = DatabaseManager()

    # Leer desde RAW
    raw_data = db.load_raw_batch(batch_id)
    data     = raw_data if isinstance(raw_data, list) else json.loads(raw_data)

    # Transformar — la API devuelve lista de listas (sin headers).
    # Cada sublista tiene exactamente len(COVERTYPE_COLUMNS) valores posicionales.
    df = pd.DataFrame(data, columns=COVERTYPE_COLUMNS)

    # Validar que el número de columnas coincide con el schema esperado
    if df.shape[1] != len(COVERTYPE_COLUMNS):
        raise AirflowException(
            f"El batch {batch_id} tiene {df.shape[1]} columnas; "
            f"se esperaban {len(COVERTYPE_COLUMNS)}. Revisa COVERTYPE_COLUMNS."
        )

    # Castear columnas numéricas (la API envía todo como strings)
    int_cols = [c for c in COVERTYPE_COLUMNS if c not in ("Wilderness_Area", "Soil_Type")]
    df[int_cols] = df[int_cols].apply(pd.to_numeric, errors="coerce")

    df.insert(0, "batch_id", batch_id)

    # Escribir en PROCESSED
    engine = db.get_engine()
    df.to_sql(
        "forest_processed",
        engine,
        if_exists="append",
        index=False,
        method="multi",   # batch insert — más eficiente que row-by-row
    )

    log.info(
        "Transformación completada — batch_id=%s, filas insertadas=%d",
        batch_id,
        len(df),
    )



def task_clean_for_training(ti) -> None:
    """
    t4 · Limpia ``forest_processed`` y genera ``forest_training``.

    Operaciones (todas idempotentes sobre el batch actual):
        1. Elimina filas con cualquier valor nulo.
        2. Elimina duplicados exactos dentro del batch.
        3. Filtra filas con valores físicamente imposibles
           (Elevation <= 0 o Cover_Type fuera del rango 1-7).
        4. Escribe el resultado en ``forest_training`` (append).
    """
    batch_id = ti.xcom_pull(task_ids="extract_raw", key="batch_id")
    if batch_id is None:
        raise AirflowException(
            "XCom 'batch_id' no encontrado. "
            "Asegúrate de que la tarea 'extract_raw' finalizó correctamente."
        )

    db     = DatabaseManager()
    engine = db.get_engine()

    # Leer solo el batch recién procesado
    df = pd.read_sql(
        "SELECT * FROM forest_processed WHERE batch_id = %(batch_id)s",
        engine,
        params={"batch_id": batch_id},
    )

    initial_rows = len(df)

    # 1 — Eliminar nulos
    df = df.dropna()
    after_nulls = len(df)

    # 2 — Eliminar duplicados (excluye id y metadata)
    feature_cols = COVERTYPE_COLUMNS
    df = df.drop_duplicates(subset=feature_cols)
    after_dupes = len(df)

    # 3 — Filtrar valores físicamente imposibles
    df = df[df["Elevation"] > 0]
    df = df[df["Cover_Type"].between(1, 7)]
    after_filter = len(df)

    if df.empty:
        raise AirflowException(
            f"Batch {batch_id}: ninguna fila sobrevivió la limpieza. "
            "Revisa la calidad del batch en forest_processed."
        )

    # 4 — Escribir en tabla de entrenamiento
    df.drop(columns=["id", "inserted_at"], errors="ignore", inplace=True)
    df.to_sql(
        "forest_training",
        engine,
        if_exists="append",
        index=False,
        method="multi",
    )

    log.info(
        "Limpieza completada — batch_id=%s | "
        "inicial=%d | sin_nulos=%d | sin_dupes=%d | final=%d | descartadas=%d",
        batch_id,
        initial_rows,
        after_nulls,
        after_dupes,
        after_filter,
        initial_rows - after_filter,
    )


# ---------------------------------------------------------------------------
# Definición del DAG
# ---------------------------------------------------------------------------

with DAG(
    dag_id="covertype_pipeline_v3",
    description="Ingesta incremental del dataset Forest Cover Type (RAW → PROCESSED).",
    schedule_interval=timedelta(minutes=5),
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "incremental", "covertype"],
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    },
) as dag:

    t1 = PythonOperator(
        task_id="validate_infra",
        python_callable=task_validate_infra,
    )

    t2 = PythonOperator(
        task_id="extract_raw",
        python_callable=task_extract_to_raw,
    )

    t3 = PythonOperator(
        task_id="transform_processed",
        python_callable=task_transform_to_processed,
    )

    t4 = PythonOperator(
        task_id="clean_for_training",
        python_callable=task_clean_for_training,
    )

    t1 >> t2 >> t3 >> t4