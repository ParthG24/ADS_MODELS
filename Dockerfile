# ── Dockerfile for Shopify Returns Prediction API ─────────────────────────────
# Experiment 6: Containerization & API Deployment
#
# Build : docker build -t shopify-returns-api .
# Run   : docker run -p 8000:8000 shopify-returns-api
# Test  : curl http://localhost:8000/health

FROM python:3.11-slim

# Metadata
LABEL maintainer="Experiment Team"
LABEL description="Shopify Returns Prediction API — Experiment 6"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy API code and model artifacts
COPY api.py .
COPY saved_models/ ./saved_models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
