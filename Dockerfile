# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py . 

# Expose & defaults
EXPOSE 8081
ENV PORT=8081
# We will store the SQLite file on a mounted volume at /data
ENV METRICS_DB=/data/metrics.db

# Create /data and give permissions to an unprivileged user
RUN adduser --disabled-password --gecos "" appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /data /app

USER appuser
CMD ["python", "-u", "app.py"]
