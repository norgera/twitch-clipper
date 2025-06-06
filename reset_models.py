#!/usr/bin/env python3
"""
Reset ML models to start fresh with updated training logic.
Run this after updating the ML training code to clear old overfitted models.
"""

import os
import shutil
import subprocess
import signal
import time
from pathlib import Path

def find_api_processes():
    """Find running API server processes."""
    pids = []
    try:
        # Look for processes listening on port 8000
        result = subprocess.run(['lsof', '-i', ':8000'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                parts = line.split()
                if len(parts) > 1 and parts[0] == 'Python':
                    pids.append(int(parts[1]))
        
        # Also look for run_integrated_system.py processes
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'run_integrated_system.py' in line and 'grep' not in line:
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pids.append(int(parts[1]))
                        except (ValueError, IndexError):
                            continue
                            
        return list(set(pids))  # Remove duplicates
    except Exception as e:
        print(f"Warning: Could not check for running processes: {e}")
    return []

def stop_api_processes(pids):
    """Stop API server processes."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"🛑 Stopped process {pid}")
            time.sleep(1)  # Give it time to shut down
        except ProcessLookupError:
            print(f"⚠️  Process {pid} already stopped")
        except PermissionError:
            print(f"❌ Permission denied stopping process {pid}")

def reset_models():
    """Remove all saved ML models to start training fresh."""
    models_dir = Path('models')
    
    if not models_dir.exists():
        print("❌ No models directory found")
        return
    
    # Check for running API processes
    api_pids = find_api_processes()
    if api_pids:
        print(f"🔍 Found running API server processes: {api_pids}")
        response = input("⚠️  Stop running API servers to prevent model regeneration? (Y/n): ")
        if response.lower() not in ['n', 'no']:
            stop_api_processes(api_pids)
            print("⏳ Waiting for processes to fully stop...")
            time.sleep(2)
    
    # Count existing files
    model_files = list(models_dir.glob('*'))
    
    if not model_files:
        print("✅ No model files to reset")
        return
    
    print(f"🔍 Found {len(model_files)} model files:")
    for file in model_files:
        print(f"  - {file.name}")
    
    # Ask for confirmation
    response = input("\n⚠️  Reset all ML models? This will start training from scratch. (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        try:
            # Remove all files in models directory
            for file in model_files:
                file.unlink()
                print(f"🗑️  Deleted {file.name}")
            
            print(f"\n✅ Successfully reset {len(model_files)} model files")
            print("🚀 ML training will start fresh on next run")
            
            # Check if files reappear (indicating a still-running process)
            time.sleep(1)
            new_files = list(models_dir.glob('*'))
            if new_files:
                print(f"\n⚠️  Warning: {len(new_files)} model files reappeared!")
                print("   This means there's still a running process saving models.")
                print("   Try stopping all Python processes manually:")
                print("   - Check: lsof -i :8000")
                print("   - Stop: kill <PID>")
            
        except Exception as e:
            print(f"❌ Error resetting models: {e}")
    else:
        print("❌ Reset cancelled")

if __name__ == "__main__":
    reset_models() 