"""
locustfile.py — Taller Locust 

Dataset: Wine (sklearn.datasets.load_wine)
Endpoint: POST /predict  con WineFeatures del repo

Stress test:  10 000 usuarios, spawn rate 500/s
Soak test:    carga sostenida 30+ minutos

Criterios de aceptación:
  ✓ Tasa de error < 1 %
  ✓ Latencia p95  < 2 000 ms
  ✓ Sin reinicios del contenedor (docker stats)
"""

from locust import HttpUser, task, between
import random

# ── Rangos reales del Wine dataset (sklearn) ──────────────────────────────────
# Los nombres de modelo son los run_name que MLflow asigna al entrenar.
# Ajusta esta lista con los que aparezcan en GET /models de tu API.
MODELOS_DISPONIBLES = [
    "mlp_small",
    "mlp_medium",
    "mlp_large",
    "mlp_wide",
    "mlp_deep",
]


def payload_aleatorio(model_name: str) -> dict:
    """Genera features dentro del rango real del Wine dataset."""
    return {
        "model_name":                  model_name,
        "alcohol":                     round(random.uniform(11.0, 15.0), 2),
        "malic_acid":                  round(random.uniform(0.7, 5.8), 2),
        "ash":                         round(random.uniform(1.4, 3.2), 2),
        "alcalinity_of_ash":           round(random.uniform(10.6, 30.0), 2),
        "magnesium":                   round(random.uniform(70.0, 162.0), 1),
        "total_phenols":               round(random.uniform(0.98, 3.88), 2),
        "flavanoids":                  round(random.uniform(0.34, 5.08), 2),
        "nonflavanoid_phenols":        round(random.uniform(0.13, 0.66), 2),
        "proanthocyanins":             round(random.uniform(0.41, 3.58), 2),
        "color_intensity":             round(random.uniform(1.28, 13.0), 2),
        "hue":                         round(random.uniform(0.48, 1.71), 2),
        "od280_od315_diluted_wines":   round(random.uniform(1.27, 4.0), 2),
        "proline":                     round(random.uniform(278.0, 1680.0), 1),
    }


class UsuarioDeCarga(HttpUser):
    """
    Simula un usuario realizando inferencias contra la API de vino.
    Estructura igual al ejemplo del enunciado del taller.
    """
    wait_time = between(1, 2.5)

    def on_start(self):
        """Warm-up: verifica que la API responde y obtiene modelos disponibles."""
        response = self.client.get("/health")
        if response.status_code != 200:
            return
        # Obtener modelos reales desde la API
        r = self.client.get("/models")
        if r.status_code == 200:
            disponibles = r.json().get("available", [])
            if disponibles:
                self.modelos = disponibles
            else:
                self.modelos = MODELOS_DISPONIBLES
        else:
            self.modelos = MODELOS_DISPONIBLES

    @task
    def hacer_inferencia(self):
        """Tarea principal — replica el patrón del enunciado del taller."""
        model_name = random.choice(self.modelos)
        payload = payload_aleatorio(model_name)

        # Enviar una petición POST al endpoint /predict
        response = self.client.post("/predict", json=payload)

        # Validación de respuesta (igual que el ejemplo del enunciado)
        if response.status_code != 200:
            print("❌ Error en la inferencia:", response.text)
