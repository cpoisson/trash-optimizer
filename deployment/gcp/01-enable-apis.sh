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

echo "🔧 Setting active project to ${PROJECT_ID}..."
gcloud config set project ${PROJECT_ID}

echo "🔧 Enabling required GCP APIs..."
gcloud services enable \
  run.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com \
  bigquery.googleapis.com \
  cloudbuild.googleapis.com

echo "✅ APIs enabled successfully"
echo ""
echo "Enabled APIs:"
echo "  - Cloud Run (container hosting)"
echo "  - Secret Manager (secure credential storage)"
echo "  - Artifact Registry (Docker image storage)"
echo "  - BigQuery (trash collection points data)"
echo "  - Cloud Build (Docker image building)"
echo ""
echo "⏳ Waiting 30 seconds for APIs to propagate..."
sleep 30
echo "✅ Ready to proceed with deployment"
