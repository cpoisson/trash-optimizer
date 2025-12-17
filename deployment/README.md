# Deployment Guide - Trash Optimizer

This guide explains how to containerize and deploy the Trash Optimizer application (Inference + Webapp) as a single Docker container.

## Architecture

The container runs two services managed by **supervisord**:
- **FastAPI (Inference Backend)**: Port 8000 (internal only)
- **Streamlit (Webapp)**: Port 8501 (exposed to host)

The webapp calls the inference API via `localhost:8000` within the container.

## Prerequisites

1. Docker and Docker Compose installed
2. Valid credentials for:
   - Hugging Face Token (for model download)
   - OpenRouteService API Key
   - Google Cloud Platform service account JSON file

## Setup Instructions

### 1. Prepare Secrets

Create the secrets directory:
```bash
mkdir -p deployment/secrets
```

Copy your GCP credentials JSON file:
```bash
cp /path/to/your/gcp-service-account.json deployment/secrets/gcp-credentials.json
```

**IMPORTANT**: Never commit `deployment/secrets/` or `deployment/.env` to git!

### 2. Configure GCP Deployment (for Cloud Run deployment)

If deploying to Google Cloud Platform, configure your GCP settings:
```bash
cp deployment/gcp/config.sh.template deployment/gcp/config.sh
```

Edit `deployment/gcp/config.sh` with your values:
```bash
PROJECT_ID="your-gcp-project-id"
REGION="europe-west9"  # or your preferred region
DATASET_ID="your-bigquery-dataset-name"
```

**IMPORTANT**: `deployment/gcp/config.sh` is gitignored and will not be committed.

### 3. Configure Environment Variables

Copy the template and fill in your secrets:
```bash
cp deployment/.env.template deployment/.env
```

Edit `deployment/.env` with your actual values:
```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
HF_MODEL_REPO_ID=your-username/your-model-repo
GEO_SERVICE_API_KEY=your_ors_api_key_here
INFERENCE_SERVICE_URL=http://localhost:8000
GCP_PROJECT=your-gcp-project-id
GCP_DATASET=your-dataset-name
GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/gcp-credentials.json
```

### 4. Build and Run

Using Docker Compose (recommended):
```bash
cd deployment
docker-compose up --build
```

Or using Docker directly:
```bash
cd deployment

# Build the image
docker build -t trash-optimizer:latest -f Dockerfile ..

# Run the container
docker run -d \
  --name trash-optimizer \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/secrets/gcp-credentials.json:/app/secrets/gcp-credentials.json:ro \
  trash-optimizer:latest
```

### 5. Access the Application

Open your browser and navigate to:
```
http://localhost:8501
```

**Note**: All Docker commands should be run from the `deployment/` directory.


## Monitoring and Logs

View logs from both services:
```bash
cd deployment

# All logs
docker-compose logs -f

# Inference service only
docker-compose exec trash-optimizer tail -f /var/log/supervisor/inference.out.log

# Webapp service only
docker-compose exec trash-optimizer tail -f /var/log/supervisor/webapp.out.log
```

## Updating the Application

```bash
# Pull latest code
git pull

# Rebuild and restart
cd deployment
docker-compose up --build -d
```

