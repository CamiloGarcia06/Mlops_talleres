# Endpoint de predicción — Clasificador de especies de pingüinos

## Descripción

El endpoint `/predict` recibe las características físicas y geográficas de un pingüino y retorna la **especie predicha** junto con las **probabilidades** para cada clase.

Modelo: red neuronal (`MLPClassifier`) entrenada con el dataset Palmer Penguins.

---

## Request

```
POST http://localhost:8989/predict
Content-Type: application/json
```

### Body

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `bill_length_mm` | `float` | Sí | Largo del pico en milímetros |
| `bill_depth_mm` | `float` | Sí | Profundidad del pico en milímetros |
| `flipper_length_mm` | `float` | Sí | Largo de la aleta en milímetros |
| `body_mass_g` | `float` | Sí | Masa corporal en gramos |
| `island` | `string` | Sí | Isla de origen: `torgersen`, `biscoe` o `dream` |
| `sex` | `string` | Sí | Sexo: `male`, `female` o `unknown` |

> Todos los valores de texto son case-insensitive.

---

## Response

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `species` | `string` | Especie predicha: `adelie`, `chinstrap` o `gentoo` |
| `probabilities` | `object` | Probabilidad asignada a cada especie (suma = 1.0) |

---

## Ejemplos

### Adelie

```bash
curl -X POST http://localhost:8989/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0,
    "island": "torgersen",
    "sex": "male"
  }'
```

```json
{
  "species": "adelie",
  "probabilities": {
    "adelie": 1.0,
    "chinstrap": 0.0,
    "gentoo": 0.0
  }
}
```

### Gentoo

```bash
curl -X POST http://localhost:8989/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bill_length_mm": 46.1,
    "bill_depth_mm": 13.2,
    "flipper_length_mm": 211.0,
    "body_mass_g": 4500.0,
    "island": "biscoe",
    "sex": "female"
  }'
```

```json
{
  "species": "gentoo",
  "probabilities": {
    "adelie": 0.0,
    "chinstrap": 0.0,
    "gentoo": 0.9999
  }
}
```

### Chinstrap

```bash
curl -X POST http://localhost:8989/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bill_length_mm": 49.0,
    "bill_depth_mm": 19.5,
    "flipper_length_mm": 210.0,
    "body_mass_g": 3950.0,
    "island": "dream",
    "sex": "male"
  }'
```

```json
{
  "species": "chinstrap",
  "probabilities": {
    "adelie": 0.0004,
    "chinstrap": 0.9993,
    "gentoo": 0.0004
  }
}
```

---

## Errores

### 422 — Validación fallida

Se retorna cuando faltan campos o los tipos son incorrectos.

```bash
curl -X POST http://localhost:8989/predict \
  -H "Content-Type: application/json" \
  -d '{"bill_length_mm": 39.1}'
```

```json
{
  "detail": [
    {
      "loc": ["body", "bill_depth_mm"],
      "msg": "Field required",
      "type": "missing"
    }
  ]
}
```

---

## Documentación interactiva

FastAPI genera documentación automática disponible en:

- **Swagger UI**: [http://localhost:8989/docs](http://localhost:8989/docs)
- **ReDoc**: [http://localhost:8989/redoc](http://localhost:8989/redoc)
