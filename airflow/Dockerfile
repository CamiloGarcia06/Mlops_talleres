# Usamos la imagen oficial de Airflow
FROM apache/airflow:2.7.1-python3.9

# Usamos el user root para tener permisos de instalación de sistema
USER root

# Instalamos uv desde el binario oficial para acelerar la instalación de Python
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Instalamos dependencias del sistema necesarias para psycopg2 y compilación
# 1. apt-get update: Actualiza el catálogo de programas Linux
# 2. -y (Assume Yes), --no-install-recommends (No instalar paquetes sugeridos)
# 3. gcc: Compilador para que el driver de Postgres (psycopg2) funcione
# 4. python3-dev: Headers para que GCC convierta funciones C en funciones Python (Interprete)
# 5. libpq-dev: Librería nativa para comunicación con Postgres
# 6. clean y rm: Borra instaladores (.deb) y el catálogo para optimizar la imagen
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Usamos uv para instalar las dependencias de Python
# --system instala en el entorno global del contenedor
RUN uv pip install --system --no-cache \
    psycopg2-binary \
    pandas \
    requests \
    boto3

# Por seguridad, volvemos al usuario airflow para la ejecución del orquestador
USER airflow