# Penguins ML API

API de inferencia en FastAPI que predice la especie de pingüino usando redes neuronales (`MLPClassifier` de scikit-learn) entrenadas sobre el dataset [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/). Todo el flujo (entrenamiento, build y servicio) corre dentro de Docker.

## Requisitos

- [Docker](https://docs.docker.com/get-docker/)
- [Task](https://taskfile.dev/) (go-task)

No necesitas Python instalado localmente; todo se ejecuta dentro del contenedor.

---

## Docker y Task

El proyecto usa **Docker** para encapsular el entorno (dependencias, entrenamiento y servicio) y **Task** como task runner que simplifica los comandos Docker.

### Imagen Docker

El `Dockerfile` parte de `python:3.12.3-slim`, instala las dependencias de `requirements.txt` y copia el codigo fuente. El contenedor expone el puerto `8989` y ejecuta Uvicorn por defecto.

### Comandos Task

| Comando | Que hace |
|---------|----------|
| `task build` | Construye la imagen Docker `penguins-api:latest` |
| `task train` | Construye la imagen y ejecuta `train.py` dentro de un contenedor temporal. Los modelos generados se montan en `app/models/` local via volumen (`-v`) |
| `task up` | Construye la imagen y levanta el contenedor `penguins-api` en background, mapeando el puerto `8989` |
| `task down` | Detiene y elimina el contenedor `penguins-api` |
| `task logs` | Muestra los logs del contenedor en tiempo real |
| `task shell` | Abre una shell bash dentro del contenedor en ejecucion |

### Flujos de uso

**Primer uso (entrenar + levantar):**

```bash
task train   # Entrena los 5 modelos dentro de Docker → genera app/models/*.joblib
task up      # Construye imagen con los modelos y levanta la API
```

**Re-entrenar modelos:**

```bash
task down    # Detener API actual
task train   # Re-entrenar (sobreescribe los .joblib)
task up      # Levantar con los nuevos modelos
```

**Desarrollo / debug:**

```bash
task up      # Levantar API
task logs    # Ver logs en tiempo real
task shell   # Entrar al contenedor para inspeccionar
task down    # Detener cuando termines
```

**Solo construir la imagen (sin levantar):**

```bash
task build
```

### Como funciona el volumen en `task train`

El entrenamiento corre dentro de Docker, pero los modelos deben quedar en tu maquina local para que `task up` los incluya en la imagen final. Esto se logra montando `./app/models` como volumen:

```
docker run --rm -v ./app/models:/app/app/models penguins-api:latest python3 train.py
```

- `--rm`: elimina el contenedor temporal al terminar
- `-v ./app/models:/app/app/models`: sincroniza la carpeta local con la del contenedor, asi los `.joblib` persisten

---

## API - Documentacion de endpoints

La API corre en `http://localhost:8989`. Tambien puedes acceder a la documentacion interactiva de Swagger en `http://localhost:8989/docs`.

### `GET /health`

Verifica que el servicio esta activo y cuantos modelos tiene cargados.

**Request:**

```bash
curl http://localhost:8989/health
```

**Response (200):**

```json
{
  "status": "ok",
  "models_loaded": 5
}
```

---

### `GET /models`

Lista los nombres de todos los modelos disponibles para prediccion.

**Request:**

```bash
curl http://localhost:8989/models
```

**Response (200):**

```json
{
  "available": ["mlp_deep", "mlp_large", "mlp_medium", "mlp_small", "mlp_wide"]
}
```

---

### `POST /predict`

Predice la especie de un pinguino a partir de sus caracteristicas fisicas y el modelo elegido.

**Request body:**

| Campo | Tipo | Requerido | Descripcion |
|-------|------|-----------|-------------|
| `bill_length_mm` | float | Si | Largo del pico (mm) |
| `bill_depth_mm` | float | Si | Profundidad del pico (mm) |
| `flipper_length_mm` | float | Si | Largo de la aleta (mm) |
| `body_mass_g` | float | Si | Masa corporal (g) |
| `island` | string | Si | Isla de origen: `"Torgersen"`, `"Biscoe"` o `"Dream"` |
| `sex` | string | Si | Sexo: `"male"` o `"female"` |
| `model_name` | string | Si | Modelo a usar (ver `GET /models`) |

**Request:**

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

**Response (200):**

```json
{
  "species": "Adelie",
  "model_used": "mlp_small"
}
```

**Errores posibles:**

| Codigo | Causa |
|--------|-------|
| 404 | `model_name` no existe. La respuesta incluye la lista de modelos disponibles |
| 422 | Faltan campos requeridos o tipos invalidos |
| 503 | No hay modelos cargados (no se ejecuto `task train`) |

**Ejemplo de error 404:**

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
    "model_name": "modelo_inexistente"
  }'
```

```json
{
  "detail": "Model 'modelo_inexistente' not found. Available: ['mlp_deep', 'mlp_large', 'mlp_medium', 'mlp_small', 'mlp_wide']"
}
```

---

## Modelos entrenados

El script `train.py` entrena 5 redes neuronales (`MLPClassifier`), cada una como un pipeline completo de scikit-learn (preprocesamiento + modelo):

| Modelo | Arquitectura | Descripcion |
|--------|-------------|-------------|
| `mlp_small` | (32,) | 1 capa oculta, 32 neuronas |
| `mlp_medium` | (64, 32) | 2 capas ocultas |
| `mlp_large` | (128, 64, 32) | 3 capas ocultas |
| `mlp_wide` | (256, 128) | 2 capas anchas |
| `mlp_deep` | (64, 64, 64, 32) | 4 capas ocultas |

El preprocesamiento incluye: imputacion de valores faltantes, escalado estandar para features numericos y one-hot encoding para features categoricos.

---

## Estructura del proyecto

```
.
├── app/
│   ├── __init__.py       # Paquete Python
│   ├── main.py           # API FastAPI (inferencia)
│   ├── schemas.py        # Modelos Pydantic
│   └── models/           # Artefactos .joblib (generados por train.py)
├── train.py              # Script de entrenamiento
├── requirements.txt      # Dependencias Python
├── Dockerfile            # Imagen Docker
└── Taskfile.yml          # Task runner (comandos Docker)
```
