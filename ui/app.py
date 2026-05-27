"""UI Streamlit para inferencia de precios y historial de entrenamiento."""

from __future__ import annotations

import streamlit as st

import client
from examples import EXAMPLE_HOUSE_1, EXAMPLE_HOUSE_2, EXAMPLE_HOUSE_3

US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
    "Washington", "West Virginia", "Wisconsin", "Wyoming",
]

st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# --- Sidebar ---
st.sidebar.title("Real Estate Predictor")
st.sidebar.markdown("---")

try:
    model_info = client.get_model_info()
    st.sidebar.subheader("Modelo activo")
    st.sidebar.markdown(f"**Nombre:** {model_info.get('model_name', 'N/A')}")
    st.sidebar.markdown(f"**Version:** {model_info.get('model_version', 'N/A')}")
    st.sidebar.markdown(f"**Alias:** {model_info.get('model_alias', 'N/A')}")
except Exception as exc:
    st.sidebar.warning(f"No se pudo obtener info del modelo: {exc}")

st.sidebar.markdown("---")
page = st.sidebar.radio("Seccion", ["Inferencia", "Historial de entrenamiento"])

# =====================================================================
# SECCION 1: INFERENCIA
# =====================================================================
if page == "Inferencia":
    st.title("Prediccion de Precio de Propiedad")
    st.markdown(
        "Ingrese las caracteristicas de la propiedad y presione **Predecir** "
        "para obtener el precio estimado."
    )

    # --- Ejemplos ---
    col_ex1, col_ex2, col_ex3, _ = st.columns([1, 1, 1, 3])
    with col_ex1:
        if st.button("Casa mediana"):
            st.session_state.update(EXAMPLE_HOUSE_1)
            st.rerun()
    with col_ex2:
        if st.button("Casa grande"):
            st.session_state.update(EXAMPLE_HOUSE_2)
            st.rerun()
    with col_ex3:
        if st.button("Casa pequena"):
            st.session_state.update(EXAMPLE_HOUSE_3)
            st.rerun()

    st.markdown("---")

    # --- Formulario ---
    with st.form("predict_form"):
        st.subheader("Caracteristicas de la propiedad")

        c1, c2, c3 = st.columns(3)
        bed = c1.number_input(
            "Habitaciones", min_value=0, max_value=20,
            value=int(st.session_state.get("bed", 3)), step=1,
        )
        bath = c2.number_input(
            "Banos", min_value=0, max_value=10,
            value=int(st.session_state.get("bath", 2)), step=1,
        )
        house_size = c3.number_input(
            "Tamano (sq ft)", min_value=100, max_value=50000,
            value=int(st.session_state.get("house_size", 1800)), step=50,
        )

        c4, c5, c6 = st.columns(3)
        acre_lot = c4.number_input(
            "Lote (acres)", min_value=0.0, max_value=1000.0,
            value=float(st.session_state.get("acre_lot", 0.5)), step=0.1,
        )
        status = c5.selectbox(
            "Estado de venta",
            options=["for_sale", "sold"],
            index=0 if st.session_state.get("status", "for_sale") == "for_sale" else 1,
        )
        prev_sold_year = c6.number_input(
            "Ano ultima venta", min_value=1900, max_value=2026,
            value=int(st.session_state.get("prev_sold_year", 2015)), step=1,
        )

        c7, c8, c9 = st.columns(3)
        state = c7.selectbox(
            "Estado",
            options=US_STATES,
            index=US_STATES.index(st.session_state.get("state", "Connecticut"))
            if st.session_state.get("state", "Connecticut") in US_STATES else 0,
        )
        city = c8.text_input(
            "Ciudad",
            value=st.session_state.get("city", "Hartford"),
        )
        zip_code = c9.text_input(
            "Codigo postal",
            value=str(st.session_state.get("zip_code", "6105")),
        )

        submitted = st.form_submit_button("Predecir", type="primary")

    # --- Resultado ---
    if submitted:
        features = {
            "bed": float(bed),
            "bath": float(bath),
            "acre_lot": float(acre_lot),
            "house_size": float(house_size),
            "city": city,
            "state": state,
            "status": status,
            "zip_code": zip_code,
            "prev_sold_year": float(prev_sold_year),
        }

        with st.spinner("Consultando la API..."):
            try:
                result = client.predict(features)
            except Exception as exc:
                st.error(f"Error: {exc}")
                result = None

        if result is not None:
            prediction = result.get("prediction", 0)
            model_name = result.get("model_name", "N/A")
            model_version = result.get("model_version", "N/A")
            request_id = result.get("request_id", "N/A")
            latency = result.get("processing_time_ms")

            st.markdown("---")
            st.subheader("Resultado")

            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Precio estimado", f"${prediction:,.0f}")
            if latency is not None:
                res_col2.metric("Latencia", f"{latency:.1f} ms")
            res_col3.metric("Modelo", f"v{model_version}")

            with st.expander("Detalle de la respuesta"):
                st.json(result)

            st.caption(
                f"Modelo: **{model_name}** v{model_version} | "
                f"Request ID: `{request_id}`"
            )

# =====================================================================
# SECCION 2: HISTORIAL DE ENTRENAMIENTO
# =====================================================================
elif page == "Historial de entrenamiento":
    st.title("Historial de Entrenamiento y Despliegue")
    st.markdown(
        "Registro de cada lote procesado: si se entreno o no, la razon, "
        "si el modelo fue promovido o rechazado, y los cambios de desempeno."
    )

    history = client.get_audit_history(limit=50)

    if not history:
        st.info("No hay registros de auditoria disponibles.")
    else:
        for entry in history:
            batch_id = entry.get("batch_id", "?")
            status_val = entry.get("status", "unknown")
            trained = entry.get("training_decision")
            promoted = entry.get("promotion_decision")
            timestamp = entry.get("run_timestamp", "")

            if promoted:
                icon = "🟢"
                label = "Promovido"
            elif trained and not promoted:
                icon = "🟡"
                label = "Entrenado, no promovido"
            elif trained is False:
                icon = "⚪"
                label = "No entreno"
            else:
                icon = "🔵"
                label = status_val

            with st.expander(f"{icon} Batch {batch_id} — {label} ({timestamp})"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Filas recibidas", entry.get("rows_received", "?"))
                c2.metric("Filas unicas", entry.get("rows_after_dedup", "?"))
                c3.metric("Estado", status_val)

                c4, c5, c6 = st.columns(3)
                c4.metric("Esquema OK", "Si" if entry.get("schema_ok") else "No")
                c5.metric("Calidad OK", "Si" if entry.get("quality_ok") else "No")
                c6.metric("Drift detectado", "Si" if entry.get("drift_detected") else "No")

                st.markdown(f"**Decision de entrenamiento:** {'Si' if trained else 'No'}")
                st.markdown(f"**Razon:** {entry.get('training_reason', 'N/A')}")

                if trained:
                    st.markdown(f"**MLflow Run ID:** `{entry.get('mlflow_run_id', 'N/A')}`")
                    st.markdown(f"**Modelo registrado:** {'Si' if entry.get('model_registered') else 'No'}")
                    st.markdown(f"**Promocion:** {'Si' if promoted else 'No'}")
                    st.markdown(f"**Razon promocion:** {entry.get('promotion_reason', 'N/A')}")

                    mae_before = entry.get("champion_metric_before")
                    mae_after = entry.get("champion_metric_after")
                    if mae_before is not None and mae_after is not None:
                        st.markdown(f"**MAE champion anterior:** ${mae_before:,.0f}")
                        st.markdown(f"**MAE nuevo modelo:** ${mae_after:,.0f}")
                        delta = ((mae_before - mae_after) / mae_before * 100) if mae_before > 0 else 0
                        st.markdown(f"**Cambio:** {delta:+.1f}%")
                    elif mae_after is not None:
                        st.markdown(f"**MAE del modelo:** ${mae_after:,.0f}")
