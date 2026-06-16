#!/usr/bin/env bash
# Build Calcify.app on macOS
# Usage:  ./build_mac.sh
set -e

echo "==> Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "==> Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Building Calcify.app with PyInstaller..."
pyinstaller Calcify.spec --noconfirm --clean

echo ""
echo "==> Done!  Your app is at:  dist/Calcify.app"
echo "    Double-click it, or drag it into /Applications."
echo ""
echo "    First launch: macOS Gatekeeper may block an unsigned app."
echo "    Right-click the app -> Open -> Open, OR run:"
echo "      xattr -dr com.apple.quarantine dist/Calcify.app"
