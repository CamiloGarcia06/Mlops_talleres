"""Streamlit inference UI for the Diabetes MLOps project.

The UI communicates exclusively with the FastAPI inference service.
It does NOT import mlflow, psycopg2 or any database driver.
"""

from __future__ import annotations

import streamlit as st

import client
from examples import PIMA_SAMPLE_PAYLOAD, SAMPLE_PAYLOAD

# ---------------------------------------------------------------------------
# All 141 one-hot features with sensible defaults (alphabetical = training order)
# ---------------------------------------------------------------------------
_MODEL_BASE: dict = {
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

_AGE_BUCKETS = [
    (0,  10, "age_[0-10)"),  (10, 20, "age_[10-20)"), (20, 30, "age_[20-30)"),
    (30, 40, "age_[30-40)"), (40, 50, "age_[40-50)"), (50, 60, "age_[50-60)"),
    (60, 70, "age_[60-70)"), (70, 80, "age_[70-80)"), (80, 90, "age_[80-90)"),
    (90, 200, "age_[90-100)"),
]


def _to_model_features(form: dict) -> dict:
    """Convert simplified form values to the full 141-feature one-hot payload."""
    payload = dict(_MODEL_BASE)

    # Direct numeric features
    for key in (
        "admission_type_id", "discharge_disposition_id", "admission_source_id",
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses",
    ):
        if key in form:
            payload[key] = float(form[key])

    # age → bucket one-hot
    age = int(form.get("age", 60))
    for lo, hi, col in _AGE_BUCKETS:
        payload[col] = 1.0 if lo <= age < hi else 0.0

    # gender one-hot  (0=Female, 1=Male, 2=Unknown)
    g = int(form.get("gender", 1))
    payload["gender_Female"]          = 1.0 if g == 0 else 0.0
    payload["gender_Male"]            = 1.0 if g == 1 else 0.0
    payload["gender_Unknown/Invalid"] = 1.0 if g == 2 else 0.0

    # change one-hot  (0=No, 1=Ch)
    ch = int(form.get("change", 0))
    payload["change_No"] = 1.0 if ch == 0 else 0.0
    payload["change_Ch"] = 1.0 if ch == 1 else 0.0

    # diabetesMed one-hot  (0=No, 1=Yes)
    dm = int(form.get("diabetes_med", 1))
    payload["diabetesMed_No"]  = 1.0 if dm == 0 else 0.0
    payload["diabetesMed_Yes"] = 1.0 if dm == 1 else 0.0

    # max_glu_serum one-hot  (0=None, 1=>200, 2=>300, 3=Norm)
    mg = int(form.get("max_glu_serum", 0))
    payload["max_glu_serum_>200"] = 1.0 if mg == 1 else 0.0
    payload["max_glu_serum_>300"] = 1.0 if mg == 2 else 0.0
    payload["max_glu_serum_Norm"] = 1.0 if mg == 3 else 0.0

    # A1Cresult one-hot  (0=None, 1=>7, 2=>8, 3=Norm)
    a1c = int(form.get("a1c_result", 0))
    payload["A1Cresult_>7"]  = 1.0 if a1c == 1 else 0.0
    payload["A1Cresult_>8"]  = 1.0 if a1c == 2 else 0.0
    payload["A1Cresult_Norm"] = 1.0 if a1c == 3 else 0.0

    return payload


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar: model info
# ---------------------------------------------------------------------------
st.sidebar.title("Diabetes Predictor")
st.sidebar.markdown("---")

try:
    model_info = client.get_model_info()
    st.sidebar.subheader("Modelo activo")
    st.sidebar.markdown(f"**Nombre:** {model_info.get('model_name', 'N/A')}")
    st.sidebar.markdown(f"**Version:** {model_info.get('model_version', 'N/A')}")
    st.sidebar.markdown(f"**Alias:** {model_info.get('model_alias', 'N/A')}")
except Exception as exc:  # noqa: BLE001
    st.sidebar.warning(f"No se pudo obtener info del modelo: {exc}")

st.sidebar.markdown("---")
st.sidebar.caption("UI conectada a la API FastAPI via variable de entorno API_URL.")

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
_DEFAULT_VALUES: dict = {
    "age": 50,
    "time_in_hospital": 3,
    "num_lab_procedures": 40,
    "num_procedures": 1,
    "num_medications": 14,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 7,
    "max_glu_serum": 0,
    "a1c_result": 0,
    "change": 0,
    "diabetes_med": 1,
    "gender": 1,
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
}

for k, v in _DEFAULT_VALUES.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Prediccion de Reingreso Hospitalario")
st.markdown(
    "Ingrese los valores clinicos del paciente o cargue un ejemplo predefinido "
    "y presione **Predecir** para obtener la probabilidad de reingreso."
)

# ---------------------------------------------------------------------------
# Example loader buttons
# ---------------------------------------------------------------------------
col_ex1, col_ex2, _ = st.columns([1, 1, 4])

with col_ex1:
    if st.button("Cargar valores de ejemplo (130-US)"):
        st.session_state.update(SAMPLE_PAYLOAD)
        st.rerun()

with col_ex2:
    if st.button("Cargar valores de ejemplo (Pima)"):
        st.session_state.update(PIMA_SAMPLE_PAYLOAD)
        st.rerun()

st.markdown("---")

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("Datos del paciente")

_pima_keys = {"Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
              "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"}
_using_pima = any(k in st.session_state for k in _pima_keys)

with st.form("predict_form"):
    if _using_pima:
        # ---- Pima layout ----
        c1, c2, c3, c4 = st.columns(4)
        pregnancies = c1.number_input(
            "Pregnancies", min_value=0, max_value=20,
            value=int(st.session_state.get("Pregnancies", 0)), step=1,
        )
        glucose = c2.number_input(
            "Glucose", min_value=0, max_value=300,
            value=int(st.session_state.get("Glucose", 120)), step=1,
        )
        blood_pressure = c3.number_input(
            "BloodPressure", min_value=0, max_value=200,
            value=int(st.session_state.get("BloodPressure", 70)), step=1,
        )
        skin_thickness = c4.number_input(
            "SkinThickness", min_value=0, max_value=100,
            value=int(st.session_state.get("SkinThickness", 20)), step=1,
        )
        c5, c6, c7, c8 = st.columns(4)
        insulin = c5.number_input(
            "Insulin", min_value=0, max_value=900,
            value=int(st.session_state.get("Insulin", 0)), step=1,
        )
        bmi = c6.number_input(
            "BMI", min_value=0.0, max_value=70.0,
            value=float(st.session_state.get("BMI", 25.0)), step=0.1,
        )
        dpf = c7.number_input(
            "DiabetesPedigreeFunction", min_value=0.0, max_value=3.0,
            value=float(st.session_state.get("DiabetesPedigreeFunction", 0.5)),
            step=0.001, format="%.3f",
        )
        age_pima = c8.number_input(
            "Age", min_value=1, max_value=120,
            value=int(st.session_state.get("Age", 30)), step=1,
        )

        features_payload: dict = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age_pima,
        }
        use_conversion = False

    else:
        # ---- Diabetes 130-US layout ----
        c1, c2, c3 = st.columns(3)

        age = c1.number_input(
            "Edad (anos)", min_value=1, max_value=120,
            value=int(st.session_state.get("age", 50)), step=1,
        )
        time_in_hospital = c2.number_input(
            "Dias en hospital", min_value=1, max_value=30,
            value=int(st.session_state.get("time_in_hospital", 3)), step=1,
        )
        num_lab_procedures = c3.number_input(
            "Num procedimientos lab", min_value=0, max_value=200,
            value=int(st.session_state.get("num_lab_procedures", 40)), step=1,
        )

        c4, c5, c6 = st.columns(3)
        num_procedures = c4.number_input(
            "Num procedimientos", min_value=0, max_value=20,
            value=int(st.session_state.get("num_procedures", 1)), step=1,
        )
        num_medications = c5.number_input(
            "Num medicamentos", min_value=0, max_value=100,
            value=int(st.session_state.get("num_medications", 14)), step=1,
        )
        number_diagnoses = c6.number_input(
            "Num diagnosticos", min_value=1, max_value=20,
            value=int(st.session_state.get("number_diagnoses", 7)), step=1,
        )

        c7, c8, c9 = st.columns(3)
        number_outpatient = c7.number_input(
            "Visitas ambulatorias", min_value=0, max_value=50,
            value=int(st.session_state.get("number_outpatient", 0)), step=1,
        )
        number_emergency = c8.number_input(
            "Visitas emergencia", min_value=0, max_value=50,
            value=int(st.session_state.get("number_emergency", 0)), step=1,
        )
        number_inpatient = c9.number_input(
            "Hospitalizaciones previas", min_value=0, max_value=20,
            value=int(st.session_state.get("number_inpatient", 0)), step=1,
        )

        c10, c11, c12 = st.columns(3)
        max_glu_serum = c10.selectbox(
            "Max glucosa serica (0=No medido, 1=>200, 2=>300, 3=Normal)",
            options=[0, 1, 2, 3],
            index=int(st.session_state.get("max_glu_serum", 0)),
        )
        a1c_result = c11.selectbox(
            "HbA1c (0=No medido, 1=>7, 2=>8, 3=Normal)",
            options=[0, 1, 2, 3],
            index=int(st.session_state.get("a1c_result", 0)),
        )
        gender = c12.selectbox(
            "Genero (0=Femenino, 1=Masculino)",
            options=[0, 1],
            index=int(st.session_state.get("gender", 1)),
        )

        c13, c14, c15 = st.columns(3)
        change = c13.selectbox(
            "Cambio medicacion (0=No, 1=Si)",
            options=[0, 1],
            index=int(st.session_state.get("change", 0)),
        )
        diabetes_med = c14.selectbox(
            "Medicacion diabetes (0=No, 1=Si)",
            options=[0, 1],
            index=int(st.session_state.get("diabetes_med", 1)),
        )
        admission_type_id = c15.number_input(
            "Tipo admision (1-8)", min_value=1, max_value=8,
            value=int(st.session_state.get("admission_type_id", 1)), step=1,
        )

        c16, c17 = st.columns(2)
        discharge_disposition_id = c16.number_input(
            "Tipo alta (1-30)", min_value=1, max_value=30,
            value=int(st.session_state.get("discharge_disposition_id", 1)), step=1,
        )
        admission_source_id = c17.number_input(
            "Fuente admision (1-25)", min_value=1, max_value=25,
            value=int(st.session_state.get("admission_source_id", 7)), step=1,
        )

        features_payload = {
            "age": age,
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "number_diagnoses": number_diagnoses,
            "max_glu_serum": max_glu_serum,
            "a1c_result": a1c_result,
            "change": change,
            "diabetes_med": diabetes_med,
            "gender": gender,
            "admission_type_id": admission_type_id,
            "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id": admission_source_id,
        }
        use_conversion = True

    submitted = st.form_submit_button("Predecir", type="primary")

# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------
if submitted:
    api_payload = _to_model_features(features_payload) if use_conversion else features_payload

    with st.spinner("Consultando la API..."):
        try:
            result = client.predict(api_payload)
        except __import__("requests").HTTPError as exc:
            status = exc.response.status_code
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except Exception:  # noqa: BLE001
                detail = exc.response.text
            st.error(f"La API respondio {status}: {detail}")
            result = None
        except __import__("requests").RequestException as exc:
            st.error(f"No se pudo contactar la API: {exc}")
            result = None

    if result is not None:
        pred = result.get("prediction")
        score = result.get("score")
        model_name = result.get("model_name", "N/A")
        model_version = result.get("model_version", "N/A")
        request_id = result.get("request_id", "N/A")
        latency = result.get("processing_time_ms")

        st.markdown("---")
        st.subheader("Resultado de la prediccion")

        res_col1, res_col2, res_col3 = st.columns(3)

        label = "REINGRESO" if pred == 1 else "NO reingreso"
        color = "red" if pred == 1 else "green"
        res_col1.markdown(
            f"<h2 style='color:{color}'>{label}</h2>", unsafe_allow_html=True
        )

        if score is not None:
            res_col2.metric("Probabilidad de reingreso", f"{score:.1%}")
        else:
            res_col2.info("Score no disponible para este modelo.")

        if latency is not None:
            res_col3.metric("Latencia de inferencia", f"{latency:.1f} ms")

        with st.expander("Detalle de la respuesta"):
            st.json(result)

        st.caption(
            f"Modelo: **{model_name}** v{model_version} | "
            f"Request ID: `{request_id}`"
        )
