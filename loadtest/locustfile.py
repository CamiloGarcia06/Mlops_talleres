"""Locust load-test for the Diabetes Inference API.

Target endpoint: POST /predict
All 141 features in alphabetical order (matching JSONB storage order used
during training). Missing / unknown categoricals default to 0.0.

Run from the Locust UI or headless:
    locust -f locustfile.py --headless -u 50 -r 10 --run-time 5m \
           --host http://api:8000 --html report.html
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

# All 141 features in alphabetical order (matches training column order).
# One-hot groups: exactly one 1 for each group; None/unknown groups = all 0.
_BASE_SAMPLE: dict = {
    "A1Cresult_>7": 0.0, "A1Cresult_>8": 0.0, "A1Cresult_Norm": 0.0,
    "acarbose_Down": 0.0, "acarbose_No": 1.0, "acarbose_Steady": 0.0, "acarbose_Up": 0.0,
    "acetohexamide_No": 1.0, "acetohexamide_Steady": 0.0,
    "admission_source_id": 7.0, "admission_type_id": 1.0,
    "age_[0-10)": 0.0, "age_[10-20)": 0.0, "age_[20-30)": 0.0, "age_[30-40)": 0.0,
    "age_[40-50)": 0.0, "age_[50-60)": 0.0, "age_[60-70)": 1.0, "age_[70-80)": 0.0,
    "age_[80-90)": 0.0, "age_[90-100)": 0.0,
    "change_Ch": 0.0, "change_No": 1.0,
    "chlorpropamide_Down": 0.0, "chlorpropamide_No": 1.0,
    "chlorpropamide_Steady": 0.0, "chlorpropamide_Up": 0.0,
    "citoglipton_No": 1.0,
    "diabetesMed_No": 0.0, "diabetesMed_Yes": 1.0,
    "discharge_disposition_id": 1.0,
    "encounter_id": 2278392.0,
    "examide_No": 1.0,
    "gender_Female": 1.0, "gender_Male": 0.0, "gender_Unknown/Invalid": 0.0,
    "glimepiride_Down": 0.0, "glimepiride_No": 1.0,
    "glimepiride-pioglitazone_No": 1.0, "glimepiride-pioglitazone_Steady": 0.0,
    "glimepiride_Steady": 0.0, "glimepiride_Up": 0.0,
    "glipizide_Down": 0.0, "glipizide-metformin_No": 1.0, "glipizide-metformin_Steady": 0.0,
    "glipizide_No": 1.0, "glipizide_Steady": 0.0, "glipizide_Up": 0.0,
    "glyburide_Down": 0.0, "glyburide-metformin_Down": 0.0, "glyburide-metformin_No": 1.0,
    "glyburide-metformin_Steady": 0.0, "glyburide-metformin_Up": 0.0,
    "glyburide_No": 1.0, "glyburide_Steady": 0.0, "glyburide_Up": 0.0,
    "insulin_Down": 0.0, "insulin_No": 0.0, "insulin_Steady": 1.0, "insulin_Up": 0.0,
    "max_glu_serum_>200": 0.0, "max_glu_serum_>300": 0.0, "max_glu_serum_Norm": 0.0,
    "metformin_Down": 0.0, "metformin_No": 0.0,
    "metformin-pioglitazone_No": 1.0, "metformin-pioglitazone_Steady": 0.0,
    "metformin-rosiglitazone_No": 1.0, "metformin-rosiglitazone_Steady": 0.0,
    "metformin_Steady": 1.0, "metformin_Up": 0.0,
    "miglitol_Down": 0.0, "miglitol_No": 1.0, "miglitol_Steady": 0.0, "miglitol_Up": 0.0,
    "nateglinide_Down": 0.0, "nateglinide_No": 1.0,
    "nateglinide_Steady": 0.0, "nateglinide_Up": 0.0,
    "number_diagnoses": 9.0, "number_emergency": 0.0,
    "number_inpatient": 0.0, "number_outpatient": 0.0,
    "num_lab_procedures": 41.0, "num_medications": 11.0, "num_procedures": 0.0,
    "patient_nbr": 8222157.0,
    "payer_code_?": 0.0, "payer_code_BC": 0.0, "payer_code_CH": 0.0, "payer_code_CM": 0.0,
    "payer_code_CP": 0.0, "payer_code_DM": 0.0, "payer_code_FR": 0.0, "payer_code_HM": 0.0,
    "payer_code_MC": 1.0, "payer_code_MD": 0.0, "payer_code_MP": 0.0, "payer_code_OG": 0.0,
    "payer_code_OT": 0.0, "payer_code_PO": 0.0, "payer_code_SI": 0.0, "payer_code_SP": 0.0,
    "payer_code_UN": 0.0, "payer_code_WC": 0.0,
    "pioglitazone_Down": 0.0, "pioglitazone_No": 1.0,
    "pioglitazone_Steady": 0.0, "pioglitazone_Up": 0.0,
    "race_?": 0.0, "race_AfricanAmerican": 0.0, "race_Asian": 0.0,
    "race_Caucasian": 1.0, "race_Hispanic": 0.0, "race_Other": 0.0,
    "repaglinide_Down": 0.0, "repaglinide_No": 1.0,
    "repaglinide_Steady": 0.0, "repaglinide_Up": 0.0,
    "rosiglitazone_Down": 0.0, "rosiglitazone_No": 1.0,
    "rosiglitazone_Steady": 0.0, "rosiglitazone_Up": 0.0,
    "time_in_hospital": 3.0,
    "tolazamide_No": 1.0, "tolazamide_Steady": 0.0, "tolazamide_Up": 0.0,
    "tolbutamide_No": 1.0, "tolbutamide_Steady": 0.0,
    "troglitazone_No": 1.0, "troglitazone_Steady": 0.0,
    "weight_?": 1.0, "weight_[0-25)": 0.0, "weight_[100-125)": 0.0,
    "weight_[125-150)": 0.0, "weight_[150-175)": 0.0, "weight_[175-200)": 0.0,
    "weight_>200": 0.0, "weight_[25-50)": 0.0, "weight_[50-75)": 0.0, "weight_[75-100)": 0.0,
}


class PredictUser(HttpUser):
    """Simulates a client calling /predict repeatedly."""

    wait_time = between(0.1, 0.5)

    @task
    def predict(self) -> None:
        payload = dict(_BASE_SAMPLE)
        # Perturb key numeric features to avoid any upstream caching
        payload["num_lab_procedures"] = float(random.randint(1, 80))
        payload["num_medications"] = float(random.randint(1, 30))
        payload["time_in_hospital"] = float(random.randint(1, 14))
        payload["number_diagnoses"] = float(random.randint(1, 16))
        payload["number_inpatient"] = float(random.randint(0, 5))
        payload["number_emergency"] = float(random.randint(0, 5))

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
