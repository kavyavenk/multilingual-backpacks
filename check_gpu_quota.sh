#!/bin/bash
# Script to check GPU quota and provide instructions for requesting increase

PROJECT="nlp-project-kav-jen"

echo "Checking GPU quota for project: $PROJECT"
echo ""

# Check GPU quota
QUOTA=$(gcloud compute project-info describe --project="$PROJECT" --format="value(quotas[metric=GPUS_ALL_REGIONS].limit)" 2>&1)
USAGE=$(gcloud compute project-info describe --project="$PROJECT" --format="value(quotas[metric=GPUS_ALL_REGIONS].usage)" 2>&1)

echo "Current GPU Quota: $QUOTA"
echo "Current GPU Usage: $USAGE"
echo ""

if [ "$QUOTA" = "0.0" ] || [ -z "$QUOTA" ]; then
    echo "❌ GPU quota is 0 - you need to request a quota increase!"
    echo ""
    echo "To request GPU quota increase:"
    echo ""
    echo "Option 1: Via GCP Console (Recommended)"
    echo "  1. Open: https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT"
    echo "  2. Filter by 'GPU' or search for 'GPUS_ALL_REGIONS'"
    echo "  3. Select the quota and click 'EDIT QUOTAS'"
    echo "  4. Request 1-2 GPUs (T4 GPUs)"
    echo "  5. Submit the request (usually approved within hours)"
    echo ""
    echo "Option 2: Via gcloud command"
    echo "  gcloud alpha service-quota update --service=compute.googleapis.com \\"
    echo "    --consumer=projects/$PROJECT \\"
    echo "    --metric=compute.googleapis.com/gpus_all_regions \\"
    echo "    --value=2"
    echo ""
    echo "Note: Quota increases may take 24-48 hours to be approved."
    echo ""
else
    echo "✅ GPU quota available: $QUOTA GPUs"
    echo "   Usage: $USAGE GPUs"
    echo "   Available: $(echo "$QUOTA - $USAGE" | bc) GPUs"
    echo ""
    echo "You can create GPU instances! Use:"
    echo "  ./create_gpu_instance.sh"
fi

