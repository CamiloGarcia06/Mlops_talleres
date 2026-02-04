# Penguins API - Tasks

Este proyecto incluye un `Taskfile.yml` para simplificar el ciclo de vida del contenedor.

## Descripción

API en FastAPI que expone el dataset `palmerpenguins` para exploración básica. Incluye endpoints de salud, listado con filtros y consulta por índice.

## Endpoints

- `GET /health`: estado del servicio.
- `GET /penguins`: listado con filtros `species`, `island`, `sex` y `limit` (1-500).
- `GET /penguins/{row_id}`: detalle por índice (0-based).

## Comandos disponibles

1. `task build`
Construye la imagen Docker `penguins-api:latest` usando el `Dockerfile`.

1. `task up`
Levanta el servicio con `docker compose` y reconstruye si hay cambios.

1. `task down`
Detiene y elimina los contenedores levantados por `docker compose`.

1. `task logs`
Muestra y sigue los logs del servicio `api`.

1. `task shell`
Abre una shell dentro del contenedor en ejecución `penguins-api`.

## Requisitos

- Docker y Docker Compose
- Task (go-task)

## Uso rápido

```bash
task build
task up
```

Luego el API estará disponible en `http://localhost:8989`.

## Convenciones de commits

Recomendación: usar formato imperativo y prefijos claros. Ejemplos:

- `feat(api): add species filter`
- `fix(api): handle null sex values`
- `docs(readme): add usage section`
- `chore(deps): bump pandas to 2.2.3`

Lista de prefijos sugeridos: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`.

## Nombres de ramas

Formato sugerido: `<tipo>/<tema-corto>` en kebab-case.

- `feat/add-species-filter`
- `fix/null-serialization`
- `docs/expand-readme`
- `chore/update-deps`

Para tareas grandes, puedes usar `feat/<tema>` + subramas si hace falta.

## Cómo agregar nuevo contenido al repositorio

1. Código de API:
   - Agrega nuevos endpoints en `app/main.py`.
   - Mantén los filtros y validaciones simples y coherentes con los existentes.
1. Modelos o artefactos:
   - Guarda artefactos en `app/models/` (p. ej. `.joblib`).
   - Versiona artefactos solo si son pequeños; si crecen, considera un almacenamiento externo.
1. Esquemas:
   - Añade Pydantic en `app/schemas/` si necesitas contratos explícitos.
1. Datos y training (si aplica):
   - Implementa pipelines en `src/data/`, `src/features/` y `src/training/`.
1. Dependencias:
   - Agrega nuevas librerías en `requirements.txt`.
1. Documentación:
   - Actualiza este README con endpoints, ejemplos y cambios relevantes.

## Estructura esperada de carpetas

```
.
├─ app/
│  ├─ main.py          # FastAPI (app = FastAPI())
│  ├─ models/          # Artefactos del modelo (p. ej., .joblib)
│  └─ schemas/         # Esquemas Pydantic
├─ src/
│  ├─ data/            # Descarga y carga de datos (palmerpenguins)
│  ├─ features/        # Transformaciones / procesamiento
│  └─ training/        # Entrenamiento del modelo
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
└─ Taskfile.yml
```
