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

echo "🔐 Creating secrets in Secret Manager..."

# Navigate to deployment directory where .env is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found in $(pwd)"
    echo "   Please ensure deployment/.env exists with HF_TOKEN and GEO_SERVICE_API_KEY"
    exit 1
fi

# Load secrets from .env
source .env

if [ -z "${HF_TOKEN}" ]; then
    echo "❌ Error: HF_TOKEN not found in .env"
    exit 1
fi

if [ -z "${GEO_SERVICE_API_KEY}" ]; then
    echo "❌ Error: GEO_SERVICE_API_KEY not found in .env"
    exit 1
fi

# Create HF_TOKEN secret
echo "Creating HF_TOKEN secret..."
if gcloud secrets describe HF_TOKEN --project=${PROJECT_ID} &>/dev/null; then
    echo "  Secret HF_TOKEN already exists, updating version..."
    echo -n "${HF_TOKEN}" | gcloud secrets versions add HF_TOKEN \
        --data-file=- \
        --project=${PROJECT_ID}
else
    echo "  Creating new secret HF_TOKEN..."
    echo -n "${HF_TOKEN}" | gcloud secrets create HF_TOKEN \
        --data-file=- \
        --replication-policy="automatic" \
        --project=${PROJECT_ID}
fi

# Create GEO_SERVICE_API_KEY secret
echo "Creating GEO_SERVICE_API_KEY secret..."
if gcloud secrets describe GEO_SERVICE_API_KEY --project=${PROJECT_ID} &>/dev/null; then
    echo "  Secret GEO_SERVICE_API_KEY already exists, updating version..."
    echo -n "${GEO_SERVICE_API_KEY}" | gcloud secrets versions add GEO_SERVICE_API_KEY \
        --data-file=- \
        --project=${PROJECT_ID}
else
    echo "  Creating new secret GEO_SERVICE_API_KEY..."
    echo -n "${GEO_SERVICE_API_KEY}" | gcloud secrets create GEO_SERVICE_API_KEY \
        --data-file=- \
        --replication-policy="automatic" \
        --project=${PROJECT_ID}
fi

echo "✅ Secrets created/updated successfully"

# Grant Cloud Run service account access to secrets
echo ""
echo "🔑 Granting Cloud Run service account access to secrets..."

PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# Try to grant permissions, but continue if it fails (may need Owner role)
if gcloud secrets add-iam-policy-binding HF_TOKEN \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor" \
    --project=${PROJECT_ID} 2>/dev/null; then
    echo "  ✓ Granted access to HF_TOKEN"
else
    echo "  ⚠️  Could not grant access to HF_TOKEN (need Owner/Secret Manager Admin role)"
    echo "     Cloud Run deployment will handle this automatically"
fi

if gcloud secrets add-iam-policy-binding GEO_SERVICE_API_KEY \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor" \
    --project=${PROJECT_ID} 2>/dev/null; then
    echo "  ✓ Granted access to GEO_SERVICE_API_KEY"
else
    echo "  ⚠️  Could not grant access to GEO_SERVICE_API_KEY (need Owner/Secret Manager Admin role)"
    echo "     Cloud Run deployment will handle this automatically"
fi

echo ""
echo "✅ Secrets setup complete"
echo ""
echo "Service Account: ${SERVICE_ACCOUNT}"
echo ""
echo "📝 Secrets stored in Secret Manager:"
echo "  • HF_TOKEN (Hugging Face model download)"
echo "  • GEO_SERVICE_API_KEY (OpenRouteService routing)"
echo ""
echo "💡 Note: Cloud Run deployment (step 5) will automatically grant the service account"
echo "   access to these secrets when you use --set-secrets flag."
