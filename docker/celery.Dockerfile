# ============================================================
# FinRisk AI — Celery Worker Dockerfile
# Shares backend image base; uses separate CMD for Celery
# ============================================================

FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r finrisk && useradd -r -g finrisk -s /bin/false finrisk

COPY --from=builder /install /usr/local

COPY backend/ backend/
COPY scripts/ scripts/
COPY data/ data/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN chown -R finrisk:finrisk /app

USER finrisk

# Default: start one Celery worker with concurrency 4
# Override CMD in docker-compose for beat scheduler
CMD ["celery", "-A", "backend.app.workers.celery_app", "worker", \
     "--loglevel=info", "--concurrency=4", "-Q", "finrisk"]
