"""Postgres connection helper. Single source of truth for the DSN."""

from __future__ import annotations

import contextlib
from typing import Iterator

import psycopg2
from psycopg2.extensions import connection as PgConnection

from pipeline.config import load


@contextlib.contextmanager
def connect() -> Iterator[PgConnection]:
    """Yield a psycopg2 connection that is committed/rolled back on exit."""
    conn = psycopg2.connect(load().pg_dsn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
