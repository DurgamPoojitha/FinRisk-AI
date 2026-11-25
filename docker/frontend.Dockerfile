# ============================================================
# FinRisk AI — Production Frontend Dockerfile (Multi-Stage)
# Base: python:3.12-slim | Target: Streamlit on port 8501
# ============================================================

# ── Stage 1: Dependency installer ───────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Minimal runtime image ──────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -r finrisk && useradd -r -g finrisk -s /bin/false finrisk

COPY --from=builder /install /usr/local

# Copy frontend code and shared utilities
COPY frontend/ frontend/
COPY backend/ backend/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BACKEND_API_URL=http://backend:8000/api

RUN chown -R finrisk:finrisk /app

USER finrisk

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true", \
     "--browser.gatherUsageStats", "false"]
