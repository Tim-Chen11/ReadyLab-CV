#!/usr/bin/env python
"""
Quick start script for the inference API
"""
import os
import sys
import subprocess
from pathlib import Path
import argparse


def find_best_checkpoint(experiments_dir="experiments"):
    """Find the most recent best checkpoint"""
    experiments_path = Path(experiments_dir)
    if not experiments_path.exists():
        return None
    
    # Look for best_checkpoint.pth files
    checkpoints = list(experiments_path.glob("*/checkpoints/best_checkpoint.pth"))
    
    if not checkpoints:
        # Try looking for any checkpoint
        checkpoints = list(experiments_path.glob("*/checkpoints/*.pth"))
    
    if checkpoints:
        # Return the most recent one
        return str(max(checkpoints, key=lambda p: p.stat().st_mtime))
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Start Decade Classifier API")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--multi-task", action="store_true", help="Use multi-task model")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    
    args = parser.parse_args()
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        print("No checkpoint specified, searching for best checkpoint...")
        checkpoint_path = find_best_checkpoint()
        
        if not checkpoint_path:
            print("❌ No checkpoint found!")
            print("Please specify a checkpoint with --checkpoint or train a model first")
            sys.exit(1)
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"✅ Using checkpoint: {checkpoint_path}")
    
    # Set environment variables
    os.environ["MODEL_CHECKPOINT"] = checkpoint_path
    if args.multi_task:
        os.environ["MULTI_TASK"] = "true"
        print("✅ Multi-task mode enabled")
    
    # Install requirements if needed
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("📦 Installing API requirements...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "api/requirements.txt"
        ])
    
    print(f"\n🚀 Starting API server on http://{args.host}:{args.port}")
    print("📖 API docs available at http://localhost:8000/docs")
    print("🌐 Web UI: Open ui/index.html in your browser")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Start server
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api.app:app",
            "--host", args.host,
            "--port", str(args.port),
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")


if __name__ == "__main__":
    main()