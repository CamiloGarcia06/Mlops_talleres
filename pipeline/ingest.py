"""Ingesta incremental desde la API externa hacia raw.properties_raw."""

from __future__ import annotations

import hashlib
import json
import logging

import requests
from psycopg2.extras import Json, execute_batch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pipeline.config import load
from pipeline.db.connection import connect

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 15_000


def _row_hash(row: dict) -> str:
    canonical = json.dumps(
        {k: ("" if v is None else str(v)) for k, v in sorted(row.items())},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _http_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def fetch_batch(api_url: str | None = None, group_number: int | None = None) -> dict:
    """Llama a GET /data?group_number=N y retorna el JSON completo."""
    settings = load()
    url = api_url or settings.data_api_url
    group = group_number or settings.group_number

    session = _http_session()
    resp = session.get(f"{url}/data", params={"group_number": group}, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("data"):
        logger.warning("la API devolvio un lote vacio (group=%d)", group)
        return {"group_number": group, "batch_number": None, "data": []}

    logger.info(
        "lote recibido: group=%d batch=%s rows=%d",
        data["group_number"], data["batch_number"], len(data["data"]),
    )
    return data


def store_batch(api_response: dict) -> dict:
    """Almacena el lote crudo en raw.properties_raw."""
    records = api_response.get("data", [])
    batch_id = str(api_response.get("batch_number", "unknown"))

    if not records:
        summary = {"batch_id": batch_id, "inserted": 0, "duplicates": 0, "total_rows": 0}
        logger.info("nada que almacenar: %s", summary)
        return summary

    chunk = records[:MAX_BATCH_SIZE]

    insert_sql = (
        "INSERT INTO raw.properties_raw "
        "(row_hash, batch_id, source, status, payload) "
        "VALUES (%s, %s, 'api', 'loaded', %s) "
        "ON CONFLICT (row_hash) DO NOTHING"
    )

    rows = []
    for record in chunk:
        rh = _row_hash(record)
        rows.append((rh, batch_id, Json(record)))

    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM raw.properties_raw")
        before = cur.fetchone()[0]
        execute_batch(cur, insert_sql, rows, page_size=1_000)
        cur.execute("SELECT count(*) FROM raw.properties_raw")
        after = cur.fetchone()[0]

    inserted = after - before
    duplicates = len(rows) - inserted

    summary = {
        "batch_id": batch_id,
        "inserted": inserted,
        "duplicates": duplicates,
        "total_rows": len(rows),
    }
    logger.info("ingesta finalizada: %s", summary)
    return summary


def run(api_url: str | None = None, group_number: int | None = None) -> dict:
    """Fetch + store en un solo paso."""
    api_response = fetch_batch(api_url, group_number)
    return store_batch(api_response)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(json.dumps(run(), indent=2))
