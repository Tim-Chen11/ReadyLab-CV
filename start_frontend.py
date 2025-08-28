#!/usr/bin/env python
"""Start the frontend with API server"""
import subprocess
import time
import webbrowser
import sys
from pathlib import Path

def main():
    print("Starting Decade Classifier Frontend...")
    print("=" * 40)
    
    # Start API server in background with multi-task model
    print("\nStep 1: Starting API server with multi-task model...")
    api_process = subprocess.Popen(
        [sys.executable, "start_api.py", "--multi-task"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print("API server starting on http://localhost:8000")
    print("Waiting for server to start...")
    time.sleep(3)
    
    # Open frontend
    print("\nStep 2: Opening frontend...")
    frontend_path = Path("ui/index.html").absolute()
    webbrowser.open(f"file://{frontend_path}")
    
    print("\n" + "=" * 40)
    print("Frontend is ready!")
    print("\nThe browser should open automatically.")
    print("If not, open ui/index.html manually.")
    print("\nPress Ctrl+C to stop the server.")
    
    try:
        # Keep the script running
        api_process.wait()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        api_process.terminate()
        print("Server stopped.")

if __name__ == "__main__":
    main()