#!/usr/bin/env python3
"""
Deployment script for Underwater Object Detection Gradio App
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🐬 Underwater Object Detection - Deployment Script")
    print("=" * 50)
    
    # Check if virtual environment exists
    if not os.path.exists("venv"):
        print("📦 Creating virtual environment...")
        if not run_command("python -m venv venv", "Creating virtual environment"):
            return False
    
    # Activate virtual environment and install dependencies
    if sys.platform == "win32":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        return False
    
    print("\n🚀 Starting the application...")
    print("📱 The app will be available at: http://127.0.0.1:7860 (default)")
    print("🌐 Public URL will be provided when the app starts")
    print("\nPress Ctrl+C to stop the application")
    
    # Start the application
    try:
        # Allow selecting entry file and port via env
        entry = os.getenv("ENTRY_FILE", "app_simple_professional.py")
        port = os.getenv("PORT", "7860")
        share = os.getenv("GRADIO_SHARE", "false")
        server_name = os.getenv("SERVER_NAME", "0.0.0.0")
        cmd = f"{activate_cmd} && set PORT={port} && set GRADIO_SHARE={share} && set SERVER_NAME={server_name} && python {entry}" if sys.platform == "win32" else f"{activate_cmd} && PORT={port} GRADIO_SHARE={share} SERVER_NAME={server_name} python {entry}"
        subprocess.run(cmd, shell=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main() 