#!/bin/bash
# Quick setup script for Trash Optimizer deployment

set -e

echo "=== Trash Optimizer Deployment Setup ==="
echo ""

# Ensure we're in the deployment directory
cd "$(dirname "$0")"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"
echo ""

# Create secrets directory
echo "📁 Creating secrets directory..."
mkdir -p secrets

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env from template..."
    cp .env.template .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env with your actual credentials:"
    echo "   - HF_TOKEN"
    echo "   - HF_MODEL_REPO_ID"
    echo "   - GEO_SERVICE_API_KEY"
    echo "   - GCP_PROJECT"
    echo "   - GCP_DATASET"
    echo ""
    echo "Press Enter when you've updated .env..."
    read -r
else
    echo "✅ .env already exists"
fi

# Check if GCP credentials exist
if [ ! -f "secrets/gcp-credentials.json" ]; then
    echo ""
    echo "⚠️  GCP credentials file not found!"
    echo "   Please copy your GCP service account JSON file to:"
    echo "   deployment/secrets/gcp-credentials.json"
    echo ""
    echo "   Example:"
    echo "   cp ~/Downloads/your-service-account.json secrets/gcp-credentials.json"
    echo ""
    echo "Press Enter when you've added the GCP credentials file..."
    read -r
else
    echo "✅ GCP credentials file found"
fi

echo ""
echo "🚀 Starting Docker Compose build and run..."
docker-compose up --build -d

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 View logs:"
echo "   docker-compose logs -f"
echo ""
echo "🌐 Access the application:"
echo "   http://localhost:8501"
echo ""
echo "🛑 Stop the application:"
echo "   docker-compose down"
