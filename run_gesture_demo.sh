#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv_gesture"

if [ ! -d "$VENV_DIR" ]; then
  python3.11 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install -U pip
  "$VENV_DIR/bin/python" -m pip install mediapipe opencv-python pillow certifi
fi

exec "$VENV_DIR/bin/python" "$ROOT_DIR/gesture_hand_demo.py"
