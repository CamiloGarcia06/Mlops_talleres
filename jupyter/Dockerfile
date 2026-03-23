# Imagen base oficial de JupyterLab con soporte para Data Science
FROM jupyter/scipy-notebook:python-3.11

# Instalar librerías necesarias para el taller (TensorFlow/Keras)
RUN pip install --no-cache-dir -r requirements.txt

# Directorio de trabajo
WORKDIR /home/jovyan/work

# Puerto por defecto de JupyterLab
EXPOSE 8888