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

### 2. Configure Environment Variables

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

### 3. Build and Run

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

### 4. Access the Application

Open your browser and navigate to:
```
http://localhost:8501
```

**Note**: All Docker commands should be run from the `deployment/` directory.

## Secret Management Best Practices

### Development Environment

**Option 1: Environment Variables + Volume Mount** (Current approach)
- Store sensitive values in `deployment/.env` (gitignored)
- Mount GCP JSON file as read-only volume
- Simple and suitable for single-server deployments

### Production Environment

**Option 2: Docker Secrets** (For Docker Swarm)
```bash
# Create secrets
echo "your_hf_token" | docker secret create hf_token -
echo "your_api_key" | docker secret create geo_api_key -
docker secret create gcp_credentials deployment/secrets/gcp-credentials.json

# Reference in docker-compose.yml
secrets:
  - hf_token
  - geo_api_key
  - gcp_credentials
```

**Option 3: External Secret Management**
- Use AWS Secrets Manager, HashiCorp Vault, or Google Secret Manager
- Inject secrets at runtime via init containers or sidecar patterns
- Best for Kubernetes deployments

### Cloud Platform Deployments

**Google Cloud Run:**
```bash
# Store secrets in Secret Manager
gcloud secrets create hf-token --data-file=-
gcloud secrets create geo-api-key --data-file=-

# Deploy with secrets mounted
gcloud run deploy trash-optimizer \
  --image gcr.io/your-project/trash-optimizer \
  --set-secrets="HF_TOKEN=hf-token:latest,GEO_SERVICE_API_KEY=geo-api-key:latest" \
  --service-account=your-service-account@project.iam.gserviceaccount.com
```

**AWS ECS/Fargate:**
- Use AWS Systems Manager Parameter Store or Secrets Manager
- Reference secrets in task definition with `secrets` parameter

**Azure Container Instances:**
- Use Azure Key Vault
- Mount secrets as secure environment variables

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

## Troubleshooting

### Model fails to download
- Check `HF_TOKEN` is valid and has read access
- Verify `HF_MODEL_REPO_ID` exists and is accessible
- Check internet connectivity from container

### BigQuery connection errors
- Verify GCP credentials JSON file is mounted correctly
- Ensure service account has BigQuery permissions
- Check `GCP_PROJECT` and `GCP_DATASET` values

### Port conflicts
- Ensure port 8501 is not already in use: `lsof -i :8501`
- Change port mapping in docker-compose.yml if needed

## Security Considerations

1. **Never commit secrets**: Add to `.gitignore`:
   ```
   deployment/.env
   deployment/secrets/
   ```

2. **Use read-only mounts**: Always mount credential files with `:ro` flag

3. **Limit service account permissions**: Use principle of least privilege for GCP service account

4. **Scan images**: Regularly scan for vulnerabilities:
   ```bash
   docker scan trash-optimizer:latest
   ```

5. **Update dependencies**: Keep base image and Python packages up to date
