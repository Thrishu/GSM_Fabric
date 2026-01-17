#!/usr/bin/env python3
"""
Startup script for Flask app - ensures all dependencies are installed first.
"""

import subprocess
import sys

def install_requirements():
    """Install requirements if not already installed."""
    packages = [
        'timm>=0.9.7',
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'Flask>=2.3',
        'opencv-python>=4.8',
    ]
    
    for package in packages:
        print(f"Checking {package}...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', package],
            capture_output=True
        )
        if result.returncode != 0:
            print(f"Failed to install {package}")
            print(result.stderr.decode())
        else:
            print(f"✓ {package}")

if __name__ == '__main__':
    print("=" * 60)
    print("Installing dependencies...")
    print("=" * 60)
    install_requirements()
    
    print("\n" + "=" * 60)
    print("Starting Flask app...")
    print("=" * 60 + "\n")
    
    # Import and run the app
    from app import app
    app.run(debug=True, host='127.0.0.1', port=5000)
