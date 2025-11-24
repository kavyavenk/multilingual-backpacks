#!/bin/bash
# Script to create a GPU instance for the multilingual-backpacks project

INSTANCE_NAME="${1:-backpack-gpu-instance}"
ZONE="${2:-us-central1-a}"
MACHINE_TYPE="${3:-n1-standard-4}"  # 4 vCPUs, 15GB RAM
GPU_TYPE="${4:-nvidia-tesla-t4}"
GPU_COUNT="${5:-1}"
PROJECT="nlp-project-kav-jen"

echo "Creating GPU instance for multilingual-backpacks project..."
echo "Instance name: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "Machine type: $MACHINE_TYPE"
echo "GPU: $GPU_COUNT x $GPU_TYPE"
echo "Project: $PROJECT"
echo ""

# Check if instance already exists
EXISTING=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT" 2>&1)
if [ $? -eq 0 ]; then
    echo "Instance '$INSTANCE_NAME' already exists in zone '$ZONE'"
    echo "Use: ./connect_gcp_gpu.sh $INSTANCE_NAME $ZONE"
    exit 1
fi

# Create the instance with GPU
echo "Creating instance (this may take a few minutes)..."
gcloud compute instances create "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
    --image-family="pytorch-2-7-cu128-ubuntu-2204-nvidia-570" \
    --image-project="deeplearning-platform-release" \
    --maintenance-policy="TERMINATE" \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-standard \
    --project="$PROJECT" \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata="install-nvidia-driver=True" \
    --no-address || {
    echo "Error creating instance. Common issues:"
    echo "1. GPU quota may be exceeded - check quotas"
    echo "2. Zone may not have GPU availability"
    echo "3. Billing may not be enabled"
    exit 1
}

echo ""
echo "Instance created successfully!"
echo ""
echo "Next steps:"
echo "1. Connect: ./connect_gcp_gpu.sh $INSTANCE_NAME $ZONE"
echo "2. Or manually: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "Note: GPU instances are expensive. Remember to stop/delete when not in use:"
echo "  Stop: gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo "  Delete: gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"

