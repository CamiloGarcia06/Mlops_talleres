-- Crear la base de datos para el proyecto
CREATE DATABASE project_data;

-- Conectarse a la base de datos del proyecto para crear las tablas de Forest
\c project_data;

-- Crear esquema de base de datos para el proyecto Forest Cover
-- Esta tabla recibe el JSON íntegro de la API (Capa Bronze)
CREATE TABLE IF NOT EXISTS forest_raw (
    id SERIAL PRIMARY KEY,
    batch_id INT,               -- Para identificar el lote de la API
    data_json JSONB,            -- Almacenamiento binario del JSON
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Esta tabla contiene los datos tipificados según la Tabla 1 del proyecto (Capa Silver)
CREATE TABLE IF NOT EXISTS forest_processed (
    id SERIAL PRIMARY KEY,
    elevation INT,
    aspect INT,
    slope INT,
    hz_dist_hydrology INT,
    vt_dist_hydrology INT,
    hz_dist_roadways INT,
    hillshade_9am INT,
    hillshade_noon INT,
    hillshade_3pm INT,
    hz_dist_fire_points INT,
    wilderness_area INT,        -- Representación numérica inicial
    soil_type INT,              -- Representación numérica inicial
    cover_type INT,             -- Nuestro TARGET (1-7)
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para el dataset final que usará el entrenamiento (Capa Gold)
CREATE TABLE IF NOT EXISTS forest_training_ready (
    id SERIAL PRIMARY KEY,
    features FLOAT8[],          -- Array de características normalizadas
    target INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);