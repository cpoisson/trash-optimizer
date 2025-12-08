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

echo "🔑 Configuring BigQuery permissions for Cloud Run service account..."

# Get Cloud Run default service account
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "Service Account: ${SERVICE_ACCOUNT}"
echo ""

# Grant BigQuery Data Viewer (read access to all datasets)
echo "Granting BigQuery Data Viewer role..."
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/bigquery.dataViewer" \
    --condition=None

# Grant BigQuery Job User (run queries)
echo "Granting BigQuery Job User role..."
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/bigquery.jobUser" \
    --condition=None

echo "✅ BigQuery permissions configured successfully"
echo ""
echo "Service account can now:"
echo "  • Read data from dataset: ${DATASET_ID}.trash_collection_points_complete"
echo "  • Execute BigQuery jobs (queries)"
echo ""
echo "📊 BigQuery Tables Accessible:"
echo "  • trash_collection_points_complete (main table used by webapp)"
