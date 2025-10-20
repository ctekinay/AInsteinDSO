#!/usr/bin/env python3
"""
Quick fix for WebSocket dependencies in 2025 Python environment
"""

import subprocess
import sys

def fix_websocket():
    """Install proper WebSocket dependencies for 2025."""
    print("🔧 Fixing WebSocket dependencies for Python 3.11+ (2025)")
    print("=" * 60)

    packages_to_install = [
        'uvicorn[standard]',
        'websockets',
        'python-multipart'
    ]

    for package in packages_to_install:
        print(f"📦 Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip3', 'install',
                '--upgrade', package
            ])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            print(f"   Try manually: pip3 install {package}")
            return False

    print("\n✅ All WebSocket dependencies installed!")
    print("🚀 Now restart with: python3 run_web_demo.py")
    return True

if __name__ == "__main__":
    fix_websocket()