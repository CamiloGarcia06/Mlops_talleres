# Imagen base oficial de JupyterLab con soporte para Data Science
FROM jupyter/scipy-notebook:python-3.11

# Instalar librerías necesarias
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt
# Directorio de trabajo
WORKDIR /home/jovyan/work

# Puerto por defecto de JupyterLab
EXPOSE 8888