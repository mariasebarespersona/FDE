# ---- Build a tiny, fast image for the API only ----
    FROM python:3.12-slim

    # system deps
    RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl && \
        rm -rf /var/lib/apt/lists/*
    
    WORKDIR /app
    
    # copy requirements & install
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # copy app
    COPY app.py .
    
    # env defaults (override in Railway)
    ENV API_KEY=devkey123 \
        METRICS_DB=/data/metrics.db \
        PORT=8081
    
    # make a writable volume for sqlite
    VOLUME ["/data"]
    
    # expose port for Railway/Fly.io (they inject PORT)
    EXPOSE 8081
    
    # start
    CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8081"]
    