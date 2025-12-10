# Multi-stage build for Trash Optimizer
# Combines inference backend (FastAPI) and webapp (Streamlit) in one container

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV using the official installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy pyproject.toml files from parent context (pinned dependencies)
COPY inference/pyproject.toml /app/inference/pyproject.toml
COPY webapp/pyproject.toml /app/webapp/pyproject.toml

# Install Python dependencies with UV (faster than pip, pinned versions)
RUN uv pip install --system --no-cache /app/inference && \
    uv pip install --system --no-cache /app/webapp

# Copy application code from parent context
COPY inference/ /app/inference/
COPY webapp/ /app/webapp/

# Create directory for GCP credentials (mounted at runtime)
RUN mkdir -p /app/secrets

# Copy supervisor configuration from deployment subdirectory
COPY deployment/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose Streamlit port (8501)
# FastAPI runs internally on 8000
EXPOSE 8501

# Health check on Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run supervisor to manage both services
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
