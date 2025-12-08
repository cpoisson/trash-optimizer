#!/bin/bash

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "${SCRIPT_DIR}/config.sh" ]; then
    echo "❌ Error: config.sh not found"
    echo "   Please copy config.sh.template to config.sh and update with your values"
    exit 1
fi
source "${SCRIPT_DIR}/config.sh"

case "$1" in
    on|start)
        echo "🟢 Starting service (min instances = 1)..."
        gcloud run services update ${SERVICE_NAME} \
            --min-instances=1 \
            --region=${REGION} \
            --project=${PROJECT_ID}
        echo ""
        echo "✅ Service is now always running (no cold starts)"
        echo "💰 Cost: ~€15-20/month for 1 instance always running (2GB RAM)"
        ;;

    off|stop)
        echo "🔴 Stopping service (min instances = 0)..."
        gcloud run services update ${SERVICE_NAME} \
            --min-instances=0 \
            --region=${REGION} \
            --project=${PROJECT_ID}
        echo ""
        echo "✅ Service will scale to zero when idle (cold starts enabled)"
        echo "💰 Cost: Pay only for actual usage (~5-10s cold start penalty)"
        ;;

    status)
        echo "📊 Service Status:"
        echo ""
        gcloud run services describe ${SERVICE_NAME} \
            --region=${REGION} \
            --project=${PROJECT_ID} \
            --format="table(status.url,spec.template.spec.containers[0].resources.limits.memory,spec.template.metadata.annotations.'autoscaling.knative.dev/minScale',spec.template.metadata.annotations.'autoscaling.knative.dev/maxScale')"
        echo ""
        echo "🔗 Direct Links:"
        SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
            --region=${REGION} \
            --project=${PROJECT_ID} \
            --format="value(status.url)")
        echo "  Webapp: ${SERVICE_URL}"
        echo "  Logs: https://console.cloud.google.com/run/detail/${REGION}/${SERVICE_NAME}/logs?project=${PROJECT_ID}"
        ;;

    logs)
        echo "📜 Recent logs (last 50 lines):"
        echo ""
        gcloud run services logs read ${SERVICE_NAME} \
            --region=${REGION} \
            --project=${PROJECT_ID} \
            --limit=50
        ;;

    url)
        gcloud run services describe ${SERVICE_NAME} \
            --region=${REGION} \
            --project=${PROJECT_ID} \
            --format="value(status.url)"
        ;;

    *)
        echo "Trash Optimizer - Cloud Run Service Control"
        echo ""
        echo "Usage: $0 {on|off|status|logs|url}"
        echo ""
        echo "Commands:"
        echo "  on/start  - Keep 1 instance always running (no cold starts, ~€15/month)"
        echo "  off/stop  - Scale to zero when idle (cold starts, pay per use only)"
        echo "  status    - Show current service configuration"
        echo "  logs      - View recent application logs"
        echo "  url       - Print service URL only"
        echo ""
        echo "Examples:"
        echo "  $0 on      # Enable always-on mode"
        echo "  $0 status  # Check current state"
        echo "  $0 logs    # View logs"
        exit 1
        ;;
esac
