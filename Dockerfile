# Dockerfile (Railway-friendly)
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV API_KEY=devkey123 \
    METRICS_DB=/tmp/metrics.db \
    PORT=8081

EXPOSE 8081

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8081"]
