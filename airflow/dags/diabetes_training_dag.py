"""
Diabetes MLOps Pipeline
Tareas:
  1. setup_schema          - Crea esquemas y tablas (idempotente)
  2. validate_source       - Valida disponibilidad del archivo fuente
  3. load_raw_data         - Carga incremental por lotes a raw_layer
  4. validate_data_quality - Validación básica de calidad
  5. preprocess_data       - Preprocesamiento y transformación
  6. store_clean_data      - Almacena datos limpios en clean_layer
  7. split_data            - Asigna splits train/val/test
  8. train_and_log         - Entrena modelos y registra en MLflow
  9. compare_models        - Compara nuevo modelo vs producción
 10. promote_best_model    - Promueve el mejor modelo al alias productivo
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime

from airflow.sdk import dag, task

DATA_BUCKET = "data"
DATA_KEY = "data/Diabetes.csv"
DATA_SOURCE = "minio_diabetes_csv"
PG_CONN_STR = os.getenv(
    "PG_CONN_STR",
    "postgresql://mlops_user:mlops_pass_2026@postgres-service.mlops.svc.cluster.local:5432/mlflow_db",
)
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-service.mlops.svc.cluster.local:5000",
)
EXPERIMENT_NAME = "diabetes-classification"
MODEL_NAME = "diabetes-classifier"
PRODUCTION_ALIAS = "production"
TARGET_COL = "Outcome"
BATCH_SIZE = 100
FEATURE_COLS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree_function", "age",
]


def _conn():
    import psycopg2
    return psycopg2.connect(PG_CONN_STR)


def _row_hash(row: dict) -> str:
    payload = json.dumps({k: str(v) for k, v in sorted(row.items())}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


@dag(
    dag_id="diabetes_mlops_pipeline",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "diabetes"],
)
def diabetes_mlops_pipeline():

    # ── 1. Esquemas y tablas ────────────────────────────────────────────────
    @task()
    def setup_schema() -> None:
        statements = [
            "CREATE SCHEMA IF NOT EXISTS raw_layer",
            """CREATE TABLE IF NOT EXISTS raw_layer.diabetes (
                id                         SERIAL PRIMARY KEY,
                record_id                  VARCHAR(64) UNIQUE,
                batch_id                   VARCHAR(64),
                load_datetime              TIMESTAMP DEFAULT NOW(),
                source                     VARCHAR(255),
                status                     VARCHAR(20) DEFAULT 'loaded',
                pregnancies                INT,
                glucose                    INT,
                blood_pressure             INT,
                skin_thickness             INT,
                insulin                    INT,
                bmi                        FLOAT,
                diabetes_pedigree_function FLOAT,
                age                        INT,
                outcome                    INT
            )""",
            "CREATE SCHEMA IF NOT EXISTS clean_layer",
            """CREATE TABLE IF NOT EXISTS clean_layer.diabetes (
                id                         SERIAL PRIMARY KEY,
                record_id                  VARCHAR(64) UNIQUE,
                batch_id                   VARCHAR(64),
                processed_datetime         TIMESTAMP DEFAULT NOW(),
                split                      VARCHAR(10),
                pregnancies                FLOAT,
                glucose                    FLOAT,
                blood_pressure             FLOAT,
                skin_thickness             FLOAT,
                insulin                    FLOAT,
                bmi                        FLOAT,
                diabetes_pedigree_function FLOAT,
                age                        FLOAT,
                outcome                    INT
            )""",
            "CREATE SCHEMA IF NOT EXISTS inference_layer",
            """CREATE TABLE IF NOT EXISTS inference_layer.logs (
                id                 SERIAL PRIMARY KEY,
                request_id         VARCHAR(64),
                inference_datetime TIMESTAMP DEFAULT NOW(),
                input_data         JSONB,
                prediction         INT,
                probability        FLOAT,
                model_name         VARCHAR(255),
                model_version      VARCHAR(50),
                response_time_ms   FLOAT
            )""",
        ]
        con = _conn()
        con.autocommit = True
        with con.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
        con.close()

    # ── 2. Validar fuente ───────────────────────────────────────────────────
    @task()
    def validate_source() -> str:
        import boto3
        from botocore.exceptions import ClientError
        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio-service.mlops.svc.cluster.local:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
        )
        try:
            s3.head_object(Bucket=DATA_BUCKET, Key=DATA_KEY)
        except ClientError as e:
            raise ValueError(f"Archivo no encontrado en MinIO: {DATA_BUCKET}/{DATA_KEY} — {e}")
        return f"s3://{DATA_BUCKET}/{DATA_KEY}"

    # ── 3. Carga incremental por lotes ──────────────────────────────────────
    @task()
    def load_raw_data(source_url: str, **kwargs) -> dict:
        import io
        import boto3
        import pandas as pd

        batch_id = kwargs.get("ds_nodash", datetime.now().strftime("%Y%m%d"))

        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio-service.mlops.svc.cluster.local:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
        )
        obj = s3.get_object(Bucket=DATA_BUCKET, Key=DATA_KEY)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))

        expected = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
        ]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columnas faltantes: {missing}. "
                f"Columnas encontradas: {df.columns.tolist()}"
            )
        inserted = skipped = 0

        con = _conn()
        with con, con.cursor() as cur:
            for start in range(0, len(df), BATCH_SIZE):
                for _, row in df.iloc[start:start + BATCH_SIZE].iterrows():
                    rd = row.to_dict()
                    record_id = _row_hash(rd)
                    cur.execute(
                        """
                        INSERT INTO raw_layer.diabetes
                            (record_id, batch_id, source, status,
                             pregnancies, glucose, blood_pressure, skin_thickness,
                             insulin, bmi, diabetes_pedigree_function, age, outcome)
                        VALUES (%s,%s,%s,'loaded',%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (record_id) DO NOTHING
                        """,
                        (record_id, batch_id, DATA_SOURCE,
                         int(rd["Pregnancies"]), int(rd["Glucose"]),
                         int(rd["BloodPressure"]), int(rd["SkinThickness"]),
                         int(rd["Insulin"]), float(rd["BMI"]),
                         float(rd["DiabetesPedigreeFunction"]), int(rd["Age"]),
                         int(rd["Outcome"])),
                    )
                    if cur.rowcount:
                        inserted += 1
                    else:
                        skipped += 1
        con.close()
        return {"batch_id": batch_id, "inserted": inserted, "skipped": skipped}

    # ── 4. Validación de calidad ────────────────────────────────────────────
    @task()
    def validate_data_quality(load_result: dict) -> dict:
        import pandas as pd
        batch_id = load_result["batch_id"]

        con = _conn()
        df = pd.read_sql(
            "SELECT * FROM raw_layer.diabetes WHERE batch_id = %s AND status = 'loaded'",
            con, params=(batch_id,),
        )
        con.close()

        issues = []
        for col in df.columns:
            nulls = df[col].isnull().sum()
            if nulls:
                issues.append(f"Nulos en '{col}': {nulls}")

        if (df["glucose"] <= 0).any():
            issues.append(f"Glucosa <= 0: {(df['glucose'] <= 0).sum()} registros")
        if (df["bmi"] <= 0).any():
            issues.append(f"BMI <= 0: {(df['bmi'] <= 0).sum()} registros")
        if not df["outcome"].isin([0, 1]).all():
            issues.append("Outcome contiene valores fuera de {0,1}")

        status = "validated" if not issues else "quality_issues"
        con = _conn()
        with con, con.cursor() as cur:
            cur.execute(
                "UPDATE raw_layer.diabetes SET status = %s WHERE batch_id = %s",
                (status, batch_id),
            )
        con.close()

        if issues:
            raise ValueError("Problemas de calidad: " + "; ".join(issues))

        return {"batch_id": batch_id, "records": len(df), "issues": issues}

    # ── 5. Preprocesamiento ─────────────────────────────────────────────────
    @task()
    def preprocess_data(quality_result: dict) -> dict:
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        batch_id = quality_result["batch_id"]

        con = _conn()
        df = pd.read_sql(
            "SELECT * FROM raw_layer.diabetes WHERE batch_id = %s AND status = 'validated'",
            con, params=(batch_id,),
        )
        con.close()

        X = df[["pregnancies", "glucose", "blood_pressure", "skin_thickness",
                 "insulin", "bmi", "diabetes_pedigree_function", "age"]].copy()

        # Reemplaza ceros inválidos con la mediana de cada columna
        zero_invalid = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
        for col in zero_invalid:
            median = X[col][X[col] > 0].median()
            X[col] = X[col].replace(0, median)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        processed = []
        for i, (x_row, rec_id, outcome) in enumerate(
            zip(X_scaled, df["record_id"].values, df["outcome"].values)
        ):
            processed.append({
                "record_id": rec_id,
                "batch_id": batch_id,
                "pregnancies": float(x_row[0]),
                "glucose": float(x_row[1]),
                "blood_pressure": float(x_row[2]),
                "skin_thickness": float(x_row[3]),
                "insulin": float(x_row[4]),
                "bmi": float(x_row[5]),
                "diabetes_pedigree_function": float(x_row[6]),
                "age": float(x_row[7]),
                "outcome": int(outcome),
            })

        return {"batch_id": batch_id, "records": processed}

    # ── 6. Almacenamiento de datos limpios ──────────────────────────────────
    @task()
    def store_clean_data(preprocessed: dict) -> dict:
        batch_id = preprocessed["batch_id"]
        records = preprocessed["records"]
        inserted = 0

        con = _conn()
        with con, con.cursor() as cur:
            for r in records:
                cur.execute(
                    """
                    INSERT INTO clean_layer.diabetes
                        (record_id, batch_id, pregnancies, glucose, blood_pressure,
                         skin_thickness, insulin, bmi, diabetes_pedigree_function,
                         age, outcome)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (record_id) DO NOTHING
                    """,
                    (r["record_id"], batch_id, r["pregnancies"], r["glucose"],
                     r["blood_pressure"], r["skin_thickness"], r["insulin"],
                     r["bmi"], r["diabetes_pedigree_function"], r["age"],
                     r["outcome"]),
                )
                inserted += cur.rowcount
        con.close()
        return {"batch_id": batch_id, "inserted": inserted}

    # ── 7. Separación train / val / test ────────────────────────────────────
    @task()
    def split_data(store_result: dict) -> dict:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        batch_id = store_result["batch_id"]

        con = _conn()
        df = pd.read_sql(
            "SELECT id, outcome FROM clean_layer.diabetes WHERE batch_id = %s",
            con, params=(batch_id,),
        )
        con.close()

        idx = df["id"].tolist()
        y = df["outcome"].tolist()

        idx_train, idx_tmp, y_train, y_tmp = train_test_split(
            idx, y, test_size=0.3, random_state=42, stratify=y,
        )
        idx_val, idx_test, _, _ = train_test_split(
            idx_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp,
        )

        split_map = (
            [(i, "train") for i in idx_train]
            + [(i, "val") for i in idx_val]
            + [(i, "test") for i in idx_test]
        )

        con = _conn()
        with con, con.cursor() as cur:
            for row_id, split in split_map:
                cur.execute(
                    "UPDATE clean_layer.diabetes SET split = %s WHERE id = %s",
                    (split, row_id),
                )
        con.close()

        return {
            "batch_id": batch_id,
            "train": len(idx_train),
            "val": len(idx_val),
            "test": len(idx_test),
        }

    # ── 8. Entrenamiento y registro en MLflow ───────────────────────────────
    @task()
    def train_and_log(split_result: dict) -> dict:
        import mlflow
        import mlflow.sklearn
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        batch_id = split_result["batch_id"]
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        con = _conn()
        df = pd.read_sql(
            "SELECT * FROM clean_layer.diabetes WHERE batch_id = %s",
            con, params=(batch_id,),
        )
        con.close()

        def _split(s):
            sub = df[df["split"] == s]
            return sub[FEATURE_COLS].values, sub["outcome"].values

        X_train, y_train = _split("train")
        X_val,   y_val   = _split("val")
        X_test,  y_test  = _split("test")

        candidates = {
            "random_forest": RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42
            ),
            "logistic_regression": LogisticRegression(
                max_iter=1000, random_state=42
            ),
        }

        best_run_id, best_f1 = None, -1.0

        for name, model in candidates.items():
            with mlflow.start_run(run_name=f"{name}_{batch_id}") as run:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val)[:, 1]

                val_f1  = f1_score(y_val, y_pred)
                val_acc = accuracy_score(y_val, y_pred)
                val_auc = roc_auc_score(y_val, y_prob)
                test_f1 = f1_score(y_test, model.predict(X_test))

                mlflow.log_params({"model_type": name, "batch_id": batch_id})
                mlflow.log_metrics({
                    "val_f1": val_f1, "val_accuracy": val_acc,
                    "val_auc": val_auc, "test_f1": test_f1,
                })
                mlflow.sklearn.log_model(
                    model, artifact_path="model",
                    registered_model_name=MODEL_NAME,
                )

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_run_id = run.info.run_id

        return {"best_run_id": best_run_id, "best_f1": best_f1, "batch_id": batch_id}

    # ── 9 + 10. Comparación y promoción ────────────────────────────────────
    @task()
    def promote_best_model(train_result: dict) -> str:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        new_run_id = train_result["best_run_id"]
        new_f1 = train_result["best_f1"]

        # Obtener métricas del modelo en producción actual
        prod_f1 = -1.0
        try:
            prod_version = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
            prod_run = client.get_run(prod_version.run_id)
            prod_f1 = prod_run.data.metrics.get("val_f1", -1.0)
        except Exception:
            pass  # No existe modelo en producción aún

        # Obtener la versión registrada del nuevo modelo
        versions = client.search_model_versions(f"run_id='{new_run_id}'")
        if not versions:
            raise ValueError(f"No se encontró versión para run_id={new_run_id}")
        new_version = versions[0].version

        if new_f1 > prod_f1:
            client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, new_version)
            return (
                f"Promovido v{new_version} a '{PRODUCTION_ALIAS}' "
                f"(val_f1={new_f1:.4f} > prod_f1={prod_f1:.4f})"
            )
        return (
            f"Modelo actual conservado "
            f"(prod_f1={prod_f1:.4f} >= new_f1={new_f1:.4f})"
        )

    # ── Pipeline ────────────────────────────────────────────────────────────
    schema   = setup_schema()
    source   = validate_source()
    loaded   = load_raw_data(source)
    quality  = validate_data_quality(loaded)
    prep     = preprocess_data(quality)
    stored   = store_clean_data(prep)
    splits   = split_data(stored)
    trained  = train_and_log(splits)
    promote_best_model(trained)

    schema >> source


diabetes_mlops_pipeline()
