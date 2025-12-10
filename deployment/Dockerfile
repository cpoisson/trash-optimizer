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

# Copy requirements files from parent context
COPY inference/requirements.txt /app/inference_requirements.txt
COPY webapp/requirements.txt /app/webapp_requirements.txt

# Install Python dependencies with UV (faster than pip)
RUN uv pip install --system --no-cache -r /app/inference_requirements.txt && \
    uv pip install --system --no-cache -r /app/webapp_requirements.txt

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
