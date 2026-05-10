"""Predefined example payloads for the inference form.

These match the feature set produced by pipeline/preprocess.py for the
Diabetes 130-US hospitals dataset (numerical / encoded columns).
The payload covers the most discriminant features used by the trained model;
any extra feature the model expects will fall back to the model's own
imputer / default handling.
"""

from __future__ import annotations

# Typical readmitted patient profile
SAMPLE_PAYLOAD: dict = {
    "age": 65,
    "time_in_hospital": 5,
    "num_lab_procedures": 44,
    "num_procedures": 1,
    "num_medications": 17,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 1,
    "number_diagnoses": 9,
    "max_glu_serum": 0,
    "a1c_result": 0,
    "change": 1,
    "diabetes_med": 1,
    "gender": 1,
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
}

# Minimal Pima-style payload (8 features) — used as fallback if the model was
# trained on the Pima dataset instead.
PIMA_SAMPLE_PAYLOAD: dict = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
}
