"""Train penguin species classifiers (neural networks) and save pipelines to app/models/."""

from pathlib import Path

import joblib
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODELS_DIR = Path("app/models")
NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["island", "sex"]


def make_preprocessor() -> ColumnTransformer:
    """Create a fresh (unfitted) preprocessor."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def main() -> None:
    # --- Data processing ---
    print("Loading data...")
    df = load_penguins()
    df = df.dropna(subset=["species"])
    df = df.drop(columns=["year"])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # --- Model training: 5 MLP architectures ---
    classifiers = {
        "mlp_small": MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42),
        "mlp_medium": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        "mlp_large": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42),
        "mlp_wide": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42),
        "mlp_deep": MLPClassifier(hidden_layer_sizes=(64, 64, 64, 32), max_iter=500, random_state=42),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, clf in classifiers.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print("=" * 60)

        pipeline = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("classifier", clf),
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        output_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(pipeline, output_path)
        print(f"Saved: {output_path}")

    print(f"\nDone. {len(classifiers)} models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
