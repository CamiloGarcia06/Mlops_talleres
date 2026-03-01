import os
from pathlib import Path

import joblib
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine, text

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["island", "sex"]
MODELS_DIR = Path("/models")


def _get_engine():
    host = os.environ["DATA_DB_HOST"]
    port = os.environ["DATA_DB_PORT"]
    user = os.environ["DATA_DB_USER"]
    password = os.environ["DATA_DB_PASSWORD"]
    dbname = os.environ["DATA_DB_NAME"]
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")


def clear_database():
    engine = _get_engine()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS penguins_raw"))
        conn.execute(text("DROP TABLE IF EXISTS penguins_clean"))


def load_raw_data():
    df = load_penguins()
    engine = _get_engine()
    df.to_sql("penguins_raw", engine, if_exists="replace", index=False)


def preprocess_data():
    engine = _get_engine()
    df = pd.read_sql_table("penguins_raw", engine)
    df = df.drop(columns=["year"])
    df = df.dropna(subset=["species"])
    df.to_sql("penguins_clean", engine, if_exists="replace", index=False)


def _make_preprocessor():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    num_idx = list(range(len(NUMERIC_FEATURES)))
    cat_idx = list(range(len(NUMERIC_FEATURES), len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)))
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_idx),
            ("cat", categorical_transformer, cat_idx),
        ],
        remainder="drop",
    )


def train_model():
    engine = _get_engine()
    df = pd.read_sql_table("penguins_clean", engine)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, clf in classifiers.items():
        pipeline = Pipeline([
            ("preprocessor", _make_preprocessor()),
            ("classifier", clf),
        ])
        pipeline.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, pipeline.predict(X_test))
        print(f"{name}: accuracy={accuracy:.4f}")

        joblib.dump(pipeline, MODELS_DIR / f"{name}.joblib")


with DAG(
    dag_id="penguins_pipeline",
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["ml", "penguins"],
) as dag:
    t1 = PythonOperator(task_id="clear_database", python_callable=clear_database)
    t2 = PythonOperator(task_id="load_raw_data", python_callable=load_raw_data)
    t3 = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    t4 = PythonOperator(task_id="train_model", python_callable=train_model)

    t1 >> t2 >> t3 >> t4
