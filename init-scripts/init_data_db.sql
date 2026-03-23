-- =============================================================
-- Inicialización de la base de datos de DATOS (ml_datasets)
-- Esta DB es exclusiva para almacenar el dataset y features
-- procesados. Es DIFERENTE a la DB de MLflow.
-- =============================================================

-- Tabla: datos crudos del dataset Wine (sklearn)
CREATE TABLE IF NOT EXISTS wine_raw (
    id              SERIAL PRIMARY KEY,
    alcohol         FLOAT,
    malic_acid      FLOAT,
    ash             FLOAT,
    alcalinity_of_ash FLOAT,
    magnesium       FLOAT,
    total_phenols   FLOAT,
    flavanoids      FLOAT,
    nonflavanoid_phenols FLOAT,
    proanthocyanins FLOAT,
    color_intensity FLOAT,
    hue             FLOAT,
    od280_od315_diluted_wines FLOAT,
    proline         FLOAT,
    target          INTEGER,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Tabla: features procesados (normalizados/escalados)
CREATE TABLE IF NOT EXISTS wine_processed (
    id              SERIAL PRIMARY KEY,
    raw_id          INTEGER REFERENCES wine_raw(id),
    alcohol         FLOAT,
    malic_acid      FLOAT,
    ash             FLOAT,
    alcalinity_of_ash FLOAT,
    magnesium       FLOAT,
    total_phenols   FLOAT,
    flavanoids      FLOAT,
    nonflavanoid_phenols FLOAT,
    proanthocyanins FLOAT,
    color_intensity FLOAT,
    hue             FLOAT,
    od280_od315_diluted_wines FLOAT,
    proline         FLOAT,
    target          INTEGER,
    split           VARCHAR(10) DEFAULT 'train',  -- 'train' o 'test'
    processed_at    TIMESTAMP DEFAULT NOW()
);

-- Tabla: registro de experimentos por corrida
CREATE TABLE IF NOT EXISTS experiment_log (
    id              SERIAL PRIMARY KEY,
    mlflow_run_id   VARCHAR(64),
    experiment_name VARCHAR(128),
    model_type      VARCHAR(64),
    hyperparams     JSONB,
    metrics         JSONB,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Índices para mejorar consultas
CREATE INDEX IF NOT EXISTS idx_wine_raw_target ON wine_raw(target);
CREATE INDEX IF NOT EXISTS idx_wine_processed_split ON wine_processed(split);
CREATE INDEX IF NOT EXISTS idx_experiment_log_run ON experiment_log(mlflow_run_id);