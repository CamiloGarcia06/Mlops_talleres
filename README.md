# Penguins ML - Desarrollo en Contenedores

Ambiente de desarrollo para Machine Learning usando **Docker Compose**, **UV** y **JupyterLab**. Entrena redes neuronales sobre el dataset [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) y sirve los modelos mediante una API en FastAPI. Ambos servicios comparten los modelos a traves de un volumen Docker.

## Arquitectura

```
                    docker-compose.yml
                           |
            +--------------+--------------+
            |                             |
     jupyter (8888)                  api (8989)
     JupyterLab + UV             FastAPI + UV
            |                             |
            +--------> /models <----------+
                   (volumen compartido)
```

El notebook entrena modelos y los guarda en `/models/`. La API los carga desde el mismo volumen para hacer inferencia. Un endpoint `/reload` permite recargar modelos sin reiniciar el servicio.

---

## Requisitos

- [Docker](https://docs.docker.com/get-docker/) con Docker Compose
- [Task](https://taskfile.dev/) (opcional, simplifica los comandos)

No necesitas Python ni UV instalados localmente; todo corre dentro de los contenedores.

---

## Inicio rapido

```bash
task up          # Construir y levantar ambos servicios
```

O sin Task:

```bash
docker compose up -d --build
```

Luego:

1. Abrir JupyterLab en `http://localhost:8888`
2. Ejecutar el notebook `train_models.ipynb` para entrenar los modelos
3. Recargar modelos en la API: `curl -X POST http://localhost:8989/reload`
4. Hacer predicciones: `POST http://localhost:8989/predict`

---

## Servicios

### JupyterLab (puerto 8888)

Entorno de desarrollo con todas las dependencias de ciencia de datos instaladas via UV. El notebook `train_models.ipynb` entrena 5 arquitecturas de redes neuronales y guarda los modelos en el volumen compartido.

### API FastAPI (puerto 8989)

Sirve los modelos entrenados para inferencia. Documentacion interactiva disponible en `http://localhost:8989/docs`.

---

## Comandos Task

| Comando | Descripcion |
|---------|-------------|
| `task up` | Construir y levantar todos los servicios |
| `task down` | Detener y eliminar servicios |
| `task build` | Solo construir imagenes |
| `task restart` | Reiniciar servicios |
| `task logs` | Ver logs de todos los servicios |
| `task logs:api` | Ver logs de la API |
| `task logs:jupyter` | Ver logs de JupyterLab |
| `task shell:api` | Shell en el contenedor de la API |
| `task shell:jupyter` | Shell en el contenedor de JupyterLab |
| `task reload` | Recargar modelos en la API |
| `task health` | Verificar estado de la API |
| `task models` | Listar modelos disponibles |

---

## Modelos entrenados

El notebook entrena 5 redes neuronales (`MLPClassifier`) con diferentes arquitecturas, cada una como un pipeline completo de scikit-learn (preprocesamiento + modelo):

| Modelo | Capas ocultas | Descripcion |
|--------|--------------|-------------|
| `mlp_small` | (32,) | 1 capa, 32 neuronas |
| `mlp_medium` | (64, 32) | 2 capas |
| `mlp_large` | (128, 64, 32) | 3 capas |
| `mlp_wide` | (256, 128) | 2 capas anchas |
| `mlp_deep` | (64, 64, 64, 32) | 4 capas profundas |

El preprocesamiento incluye: imputacion de valores faltantes, escalado estandar para features numericos y one-hot encoding para categoricos.

---

## API - Endpoints

### `GET /health`

```bash
curl http://localhost:8989/health
```

```json
{"status": "ok", "models_loaded": 5}
```

### `GET /models`

```bash
curl http://localhost:8989/models
```

```json
{"available": ["mlp_deep", "mlp_large", "mlp_medium", "mlp_small", "mlp_wide"]}
```

### `POST /predict`

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| `bill_length_mm` | float | Largo del pico (mm) |
| `bill_depth_mm` | float | Profundidad del pico (mm) |
| `flipper_length_mm` | float | Largo de la aleta (mm) |
| `body_mass_g` | float | Masa corporal (g) |
| `island` | string | `"Torgersen"`, `"Biscoe"` o `"Dream"` |
| `sex` | string | `"male"` o `"female"` |
| `model_name` | string | Modelo a usar (ver `GET /models`) |

```bash
curl -X POST http://localhost:8989/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0,
    "island": "Torgersen",
    "sex": "male",
    "model_name": "mlp_small"
  }'
```

```json
{"species": "Adelie", "model_used": "mlp_small"}
```

### `POST /reload`

Recarga los modelos desde el volumen compartido sin reiniciar el servicio.

```bash
curl -X POST http://localhost:8989/reload
```

```json
{"status": "reloaded", "models_loaded": 5}
```

### Errores

| Codigo | Causa |
|--------|-------|
| 404 | `model_name` no existe |
| 422 | Campos faltantes o tipos invalidos |
| 503 | No hay modelos cargados |

---

## Estructura del proyecto

```
.
├── docker-compose.yml           # Orquestacion de servicios
├── api/
│   ├── Dockerfile               # Imagen API con UV
│   ├── pyproject.toml           # Dependencias de la API
│   └── app/
│       ├── __init__.py
│       ├── main.py              # FastAPI (inferencia + reload)
│       └── schemas.py           # Modelos Pydantic
├── jupyter/
│   ├── Dockerfile               # Imagen JupyterLab con UV
│   ├── pyproject.toml           # Dependencias de ciencia de datos
│   └── notebooks/
│       └── train_models.ipynb   # Notebook de entrenamiento
├── models/                      # Volumen compartido (modelos .joblib)
├── Taskfile.yml                 # Task runner
└── README.md
```

---

## Tecnologias

- **Docker Compose** - Orquestacion de servicios
- **UV** - Gestor de dependencias Python (reemplaza pip)
- **JupyterLab** - Entorno de desarrollo interactivo
- **FastAPI** - Framework para la API de inferencia
- **scikit-learn** - Entrenamiento de modelos (MLPClassifier)
- **Palmer Penguins** - Dataset de clasificacion
