#!/bin/bash
# Monitor training progress

echo "Training Monitor"
echo "================"
echo ""

# Check if process is running
PID=$(pgrep -f "python train_tiny.py")
if [ -n "$PID" ]; then
    echo "✓ Training process is running (PID: $PID)"
else
    echo "✗ Training process not found"
    echo ""
    echo "Last 20 lines of log:"
    tail -20 training_tiny.log 2>/dev/null || echo "No log file found"
    exit 1
fi

echo ""
echo "Latest log output:"
echo "-------------------"
tail -15 training_tiny.log 2>/dev/null || echo "No log file yet"

echo ""
echo "-------------------"
echo ""
echo "To monitor live: tail -f training_tiny.log"
echo "To stop training: kill $PID"
echo ""
