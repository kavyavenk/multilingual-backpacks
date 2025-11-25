#!/bin/bash
# Quick setup script to transfer code to GCP GPU and run training + evaluation

set -e

# Configuration
INSTANCE_NAME="gpu-backpack-train"
ZONE="us-east1-b"
PROJECT="nlp-project-kav-jen"
REMOTE_DIR="multilingual-backpacks"

echo "=================================================="
echo "GCP GPU TRAINING SETUP"
echo "=================================================="
echo ""

# Step 1: Check if instance exists
echo "Checking for GPU instance..."
if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT" &>/dev/null; then
    echo "✓ Instance '$INSTANCE_NAME' exists"
    
    # Check if running
    STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT" --format="value(status)")
    if [ "$STATUS" != "RUNNING" ]; then
        echo "Starting instance..."
        gcloud compute instances start "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT"
        sleep 10
    fi
else
    echo "Creating GPU instance..."
    gcloud compute instances create "$INSTANCE_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --machine-type=n1-standard-4 \
        --accelerator=type=nvidia-tesla-t4,count=1 \
        --image-family=pytorch-latest-gpu \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --boot-disk-size=50GB \
        --metadata="install-nvidia-driver=True"
    
    echo "Waiting for instance to be ready..."
    sleep 30
fi

echo ""
echo "Step 2: Transferring code to GPU instance..."
# Create tar of relevant files (exclude out/, __pycache__, etc.)
tar czf /tmp/multilingual-backpacks.tar.gz \
    --exclude='out/*' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='*.pdf' \
    -C /Users/itskavya/Documents \
    multilingual-backpacks

# Transfer to instance
gcloud compute scp /tmp/multilingual-backpacks.tar.gz \
    "$INSTANCE_NAME:~/" \
    --zone="$ZONE" \
    --project="$PROJECT"

# Extract and setup
gcloud compute ssh "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --command="tar xzf multilingual-backpacks.tar.gz && \
               cd multilingual-backpacks && \
               pip install -q -r requirements.txt && \
               echo 'Setup complete!'"

echo ""
echo "Step 3: Running training and evaluation..."
gcloud compute ssh "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --command="cd multilingual-backpacks && bash train_and_eval.sh"

echo ""
echo "Step 4: Downloading results..."
gcloud compute scp --recurse \
    "$INSTANCE_NAME:~/multilingual-backpacks/out" \
    /Users/itskavya/Documents/multilingual-backpacks/ \
    --zone="$ZONE" \
    --project="$PROJECT"

echo ""
echo "=================================================="
echo "COMPLETE!"
echo "=================================================="
echo ""
echo "Results saved to: out/tiny/"
echo ""
echo "To stop the instance (to save costs):"
echo "  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "To delete the instance:"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
echo ""
