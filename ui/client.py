"""HTTP client wrapper for the inference API.

All network calls from the UI pass through this module so that error
handling, timeouts, and the base URL are defined in one place.
The UI never imports mlflow, psycopg2 or any DB driver — all inference
goes through the API.
"""

from __future__ import annotations

import os
from typing import Any

import requests

_DEFAULT_API_URL = "http://api:8000"


def _base() -> str:
    return os.environ.get("API_URL", _DEFAULT_API_URL).rstrip("/")


def get_model_info(timeout: int = 30) -> dict[str, Any]:
    """Return the /model-info payload or raise requests.RequestException."""
    resp = requests.get(f"{_base()}/model-info", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def predict(features: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    """Call POST /predict and return the response dict.

    Raises:
        requests.HTTPError: on 4xx/5xx responses.
        requests.RequestException: on network failures.
    """
    resp = requests.post(
        f"{_base()}/predict",
        json={"features": features},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()
