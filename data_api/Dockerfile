FROM python:3.9-slim

# Definir directorio de trabajo
WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY main.py .

# Crear el directorio donde la API buscará el CSV y los timestamps
RUN mkdir /data

# Exponer el puerto solicitado por el taller
EXPOSE 8080

# Ejecutar con Uvicorn en el puerto 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]