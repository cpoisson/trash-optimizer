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

# ⚠️  SECURITY WARNING: This script uses the default Compute Engine service account
# This implementation is for DEMO/DEVELOPMENT purposes only.
# This is NOT recommended for production environments because:
# 1. The default service account is shared across all Cloud Run services in the project
# 2. It grants broad permissions (project-level access to ALL datasets)
# 3. Violates the principle of least privilege
#
# FOR PRODUCTION:
# - Create a dedicated service account for this specific application
# - Grant dataset-level permissions (not project-level)
# - Use IAM conditions to restrict access further
# - Example:
#   gcloud iam service-accounts create trash-optimizer-sa \
#       --display-name="Trash Optimizer Service Account"
#   gcloud projects add-iam-policy-binding ${PROJECT_ID} \
#       --member="serviceAccount:trash-optimizer-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
#       --role="roles/bigquery.dataViewer" \
#       --condition='expression=resource.name.startsWith("projects/${PROJECT_ID}/datasets/${DATASET_ID}"),title=dataset-access'

# Get Cloud Run default service account
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "Service Account: ${SERVICE_ACCOUNT}"
echo "⚠️  WARNING: Using default compute service account (not recommended for production)"
echo ""

# Grant BigQuery Data Viewer (read access to all datasets)
# ⚠️  WARNING: This grants access to ALL datasets in the project
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
