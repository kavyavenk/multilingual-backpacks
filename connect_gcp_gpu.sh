#!/bin/bash
# Helper script to connect to GCP GPU instance

# Default values (update these or pass as arguments)
INSTANCE_NAME="${1:-gpu-instance}"
ZONE="${2:-us-east1-b}"
PROJECT="${3:-nlp-project-kav-jen}"

echo "Connecting to GCP GPU instance..."
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "Project: $PROJECT"
echo ""

# Check if instance exists and is running
echo "Checking instance status..."
gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT" 2>&1 | grep -E "(status|machineType|name)" || {
    echo "Error: Could not find instance '$INSTANCE_NAME' in zone '$ZONE'"
    echo ""
    echo "Available options:"
    echo "1. List all instances: gcloud compute instances list"
    echo "2. Start a stopped instance: gcloud compute instances start $INSTANCE_NAME --zone=$ZONE"
    echo "3. Create a new GPU instance (if needed)"
    exit 1
}

# Start instance if stopped
STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT" --format="value(status)" 2>/dev/null)
if [ "$STATUS" = "TERMINATED" ] || [ "$STATUS" = "STOPPED" ]; then
    echo "Instance is $STATUS. Starting it..."
    gcloud compute instances start "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT"
    echo "Waiting for instance to be ready..."
    sleep 10
fi

# Connect via SSH
echo "Connecting via SSH..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT" -- -L 8888:localhost:8888

