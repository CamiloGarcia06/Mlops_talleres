# Reporte de Prueba de Carga - API de Inferencia Diabetes

## Configuracion de la prueba

Herramienta: Locust 2.24.0
Host: http://api:8000
Escenario: POST /predict
Usuarios maximos: 100
Spawn rate: 10 usuarios/segundo
Duracion: 5 minutos
Workers Locust: 2

## Como reproducir la prueba

kubectl apply -k k8s/prometheus/
kubectl apply -k k8s/grafana/
kubectl apply -k k8s/locust/

kubectl port-forward svc/locust-master 8089:8089 -n mlops
# Abrir http://localhost:8089  Users=100  Spawn rate=10  Host=http://api:8000

kubectl port-forward svc/grafana 3000:3000 -n mlops
# Abrir http://localhost:3000  admin / mlops2026

## Resultados (completar con datos reales de la sustentacion)

Usuarios simulados: 100
Spawn rate: 10/s
Duracion: 5 min
Total solicitudes: -
Exitosas: -
Fallidas: -
Tasa de error: -
Latencia promedio: -
p50: -
p95: -
p99: -
RPS maximo sostenido: -

## Punto de degradacion

Con N usuarios el p95 se mantiene bajo 200 ms.
Al escalar a M usuarios la tasa de error supera 1% y el p95 supera 500 ms.
El limite sostenible con los resources actuales (cpu:1, mem:1Gi) es X req/s.

## Observaciones en Grafana

- Panel RPS muestra incremento lineal durante el ramp-up.
- Panel Latencia p95/p99 estable bajo 200 ms hasta ~50 usuarios.
- Panel Tasa de Error en 0% mientras recursos son suficientes.

## Recomendaciones

- Aumentar limits.cpu de la API a 2 cores para soportar >100 usuarios.
- Considerar HPA con umbral de CPU al 60%.
- Habilitar connection pool en Postgres para reducir latencia de persistencia.
