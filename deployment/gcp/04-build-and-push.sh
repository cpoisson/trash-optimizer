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

TAG="latest"

# Default to local build (faster)
BUILD_METHOD="${1:-local}"

echo "🐳 Building and pushing Docker image to Artifact Registry..."

# Create Artifact Registry repository if it doesn't exist
echo "Creating Artifact Registry repository in ${REGION}..."
gcloud artifacts repositories create ${REPO_NAME} \
    --repository-format=docker \
    --location=${REGION} \
    --description="Trash Optimizer container images (inference + webapp)" \
    --project=${PROJECT_ID} 2>/dev/null || echo "  Repository already exists"

# Configure Docker authentication
echo "Configuring Docker authentication for Artifact Registry..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Build image URL
IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}"

# Navigate to project root (parent of deployment/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

if [ "${BUILD_METHOD}" = "cloud" ]; then
    echo ""
    echo "Building image with Cloud Build (remote, slower but no local Docker needed)..."
    echo "  Image will be tagged as: ${IMAGE_URL}"
    echo ""

    # Build using Cloud Build
    gcloud builds submit \
        --project=${PROJECT_ID} \
        --substitutions=_IMAGE_URL=${IMAGE_URL} \
        --config=deployment/gcp/cloudbuild.yaml \
        .
else
    echo ""
    echo "Building image locally (faster, multi-platform for Cloud Run linux/amd64)..."
    echo "  Image will be tagged as: ${IMAGE_URL}"
    echo ""

    # Build locally with correct architecture for Cloud Run (linux/amd64)
    docker buildx build \
        --platform linux/amd64 \
        -f deployment/Dockerfile \
        -t ${IMAGE_URL} \
        --load \
        .

    echo ""
    echo "Pushing image to Artifact Registry..."
    docker push ${IMAGE_URL}
fi

echo ""
echo "✅ Image built and pushed successfully"
echo ""
echo "📦 Image Details:"
echo "  Build method: ${BUILD_METHOD}"
echo "  Registry: Artifact Registry (${REGION})"
echo "  Repository: ${REPO_NAME}"
echo "  Image URL: ${IMAGE_URL}"
echo "  Platform: linux/amd64 (Cloud Run compatible)"
echo "  Contents: FastAPI inference (port 8000) + Streamlit webapp (port 8501)"
echo ""
echo "💡 Usage: $0 [local|cloud]"
echo "   local - Build locally with Docker (faster, default)"
echo "   cloud - Build with Cloud Build (slower, no local Docker needed)"
