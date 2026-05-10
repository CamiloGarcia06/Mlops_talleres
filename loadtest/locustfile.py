"""Locust load-test for the Diabetes Inference API.

Target endpoint: POST /predict
The feature payload matches the numeric columns expected by the
preprocessed clean.diabetes_clean table (130-US Hospitals dataset).

Run from the Locust UI or headless:
    locust -f locustfile.py --headless -u 50 -r 10 --run-time 5m \\
           --host http://api:8000 --html report.html
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

# Representative feature sample for the Diabetes 130-US dataset.
# Numeric columns after median imputation; one-hot columns not present
# default to 0.0 at the sklearn ColumnTransformer level.
_BASE_SAMPLE: dict = {
    "encounter_id": 2278392,
    "patient_nbr": 8222157,
    "age": 7,
    "time_in_hospital": 3,
    "num_lab_procedures": 41,
    "num_procedures": 0,
    "num_medications": 11,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 9,
    "race_Caucasian": 1,
    "race_AfricanAmerican": 0,
    "race_Hispanic": 0,
    "race_Asian": 0,
    "race_Other": 0,
    "gender_Male": 0,
    "gender_Female": 1,
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "insulin_No": 0,
    "insulin_Steady": 1,
    "insulin_Up": 0,
    "insulin_Down": 0,
    "change_Ch": 1,
    "change_No": 0,
    "diabetesMed_Yes": 1,
    "diabetesMed_No": 0,
    "A1Cresult_None": 1,
    "A1Cresult_8": 0,
    "A1Cresult_Norm": 0,
    "A1Cresult_7": 0,
    "metformin_No": 0,
    "metformin_Steady": 1,
    "glipizide_No": 1,
    "glipizide_Steady": 0,
    "glyburide_No": 1,
    "glyburide_Steady": 0,
    "pioglitazone_No": 1,
    "pioglitazone_Steady": 0,
    "rosiglitazone_No": 1,
    "rosiglitazone_Steady": 0,
}


class PredictUser(HttpUser):
    """Simulates a client calling /predict repeatedly."""

    wait_time = between(0.1, 0.5)

    @task
    def predict(self) -> None:
        payload = dict(_BASE_SAMPLE)
        # Perturb key numeric features to avoid any upstream caching
        payload["num_lab_procedures"] = random.randint(1, 80)
        payload["num_medications"] = random.randint(1, 30)
        payload["time_in_hospital"] = random.randint(1, 14)
        payload["number_diagnoses"] = random.randint(1, 16)
        payload["number_inpatient"] = random.randint(0, 5)
        payload["number_emergency"] = random.randint(0, 5)

        with self.client.post(
            "/predict",
            json={"features": payload},
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"status={resp.status_code} body={resp.text[:120]}")

    @task(weight=1)
    def health(self) -> None:
        """Light probe to confirm the API is alive."""
        self.client.get("/health", name="/health")
