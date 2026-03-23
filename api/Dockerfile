FROM python:3.11-slim

WORKDIR /app

# Dependencias
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# Instalamos con uv
# --system   → instala en el Python del sistema, no en un virtualenv
# --no-cache → no guarda cache para mantener la imagen liviana
RUN uv pip install --system --no-cache -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]