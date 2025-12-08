#!/bin/bash
set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "${SCRIPT_DIR}/config.sh" ]; then
    echo "❌ Error: config.sh not found"
    echo "   Please copy config.sh.template to config.sh and update with your values"
    exit 1
fi
source "${SCRIPT_DIR}/config.sh"

IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"

echo "🚀 Deploying Trash Optimizer to Cloud Run..."
echo ""
echo "Configuration:"
echo "  Service: ${SERVICE_NAME}"
echo "  Region: ${REGION} (Paris)"
echo "  Image: ${IMAGE_URL}"
echo "  Memory: 2GB"
echo "  CPU: 1 vCPU"
echo "  Port: 8501 (Streamlit webapp)"
echo "  Min instances: 0 (scales to zero when idle)"
echo "  Max instances: 10"
echo ""

gcloud run deploy ${SERVICE_NAME} \
    --image=${IMAGE_URL} \
    --platform=managed \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --port=8501 \
    --memory=2Gi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=10 \
    --timeout=300 \
    --concurrency=80 \
    --allow-unauthenticated \
    --set-env-vars="GCP_PROJECT=${PROJECT_ID},GCP_DATASET=nantes,INFERENCE_SERVICE_URL=http://localhost:8000,HF_MODEL_REPO_ID=cpoisson/trash-optimizer-models" \
    --set-secrets="HF_TOKEN=HF_TOKEN:latest,GEO_SERVICE_API_KEY=GEO_SERVICE_API_KEY:latest"

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Service URL:"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --format="value(status.url)")

echo "  ${SERVICE_URL}"
echo ""
echo "🎯 Quick Actions:"
echo "  • Open webapp: ${SERVICE_URL}"
echo "  • View logs: gcloud run services logs read ${SERVICE_NAME} --region=${REGION} --limit=50"
echo "  • Service info: ./deployment/gcp/service-control.sh status"
echo ""
echo "💡 Service is configured to scale to zero when idle (no cost when not in use)"
