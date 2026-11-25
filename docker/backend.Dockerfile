# ============================================================
# FinRisk AI — Production Backend Dockerfile (Multi-Stage)
# Base: python:3.12-slim | Target: FastAPI + Uvicorn + Celery
# ============================================================

# ── Stage 1: Build wheel dependencies ───────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools only in builder stage — not in final image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies into /install prefix
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Minimal runtime image ──────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Only install runtime system libs (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r finrisk && useradd -r -g finrisk -s /bin/false finrisk

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY backend/ backend/
COPY scripts/ scripts/
COPY data/ data/

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Give ownership to non-root user
RUN chown -R finrisk:finrisk /app

USER finrisk

# Expose FastAPI port
EXPOSE 8000

# Health check for orchestrators (ECS, Kubernetes, docker-compose)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production entrypoint: 4 Uvicorn workers, access log off for perf
CMD ["uvicorn", "backend.app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--no-access-log"]
