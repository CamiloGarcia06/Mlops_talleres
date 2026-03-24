FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.14.3 psycopg2-binary boto3

WORKDIR /app

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "postgresql://mlflow:mlflow_secret@mlflow-db:5432/mlflow_metadata", \
     "--default-artifact-root", "s3://mlflow-artifacts/", \
     "--serve-artifacts"]
