#!/usr/bin/env python3
"""
Monitor training progress for tiny Backpack model
"""

import json
import os
import time
import subprocess

def get_process_info():
    """Check if training process is running"""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'python train_tiny.py'],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            pid = result.stdout.strip()
            # Get runtime
            ps_result = subprocess.run(
                ['ps', '-p', pid, '-o', 'etime=,cputime=,%cpu=,%mem='],
                capture_output=True,
                text=True
            )
            return pid, ps_result.stdout.strip()
        return None, None
    except:
        return None, None

def check_training_progress():
    """Monitor training progress"""
    print("="*70)
    print("TRAINING PROGRESS MONITOR")
    print("="*70)
    print()
    
    # Check process
    pid, stats = get_process_info()
    if pid:
        print(f"✓ Training process running (PID: {pid})")
        if stats:
            parts = stats.split()
            if len(parts) >= 4:
                print(f"  Runtime: {parts[0]}")
                print(f"  CPU time: {parts[1]}")
                print(f"  CPU usage: {parts[2]}%")
                print(f"  Memory: {parts[3]}%")
    else:
        print("✗ Training process not found")
        print("\nCheck if training completed or failed.")
        return
    
    print()
    
    # Check training log
    log_file = "out/tiny/training_log.json"
    if os.path.exists(log_file):
        with open(log_file) as f:
            data = json.load(f)
        
        iterations = data.get('iterations', [])
        train_losses = data.get('train_loss', [])
        val_losses = data.get('val_loss', [])
        
        if iterations:
            current_iter = iterations[-1]
            target_iter = 1000
            progress = (current_iter / target_iter) * 100
            
            print(f"Progress: {current_iter}/{target_iter} iterations ({progress:.1f}%)")
            print()
            
            if len(train_losses) > 0:
                print(f"Latest metrics (iter {current_iter}):")
                print(f"  Train loss: {train_losses[-1]:.4f}")
                print(f"  Val loss:   {val_losses[-1]:.4f}")
                
                if len(train_losses) > 1:
                    print()
                    print(f"Initial train loss: {train_losses[0]:.4f}")
                    print(f"Loss reduction: {train_losses[0] - train_losses[-1]:.4f}")
            
            print()
            print("Training curve data points:", len(iterations))
        else:
            print("No training iterations logged yet (model initializing...)")
    else:
        print(f"Log file not found: {log_file}")
    
    print()
    
    # Check for checkpoint
    ckpt_file = "out/tiny/ckpt.pt"
    if os.path.exists(ckpt_file):
        ckpt_size = os.path.getsize(ckpt_file) / (1024*1024)  # MB
        print(f"✓ Checkpoint saved: {ckpt_size:.1f} MB")
    else:
        print("○ No checkpoint saved yet (will save after training)")
    
    print()
    print("="*70)
    print()
    print("Commands:")
    print(f"  Monitor live: watch -n 5 'python {__file__}'")
    print(f"  View log: cat {log_file}")
    print(f"  Stop training: kill {pid}")
    print()

if __name__ == '__main__':
    check_training_progress()
