#!/usr/bin/env bash
# Linux / macOS / Git-Bash launcher for install.py
set -euo pipefail
cd "$(dirname "$0")"

if [ "${EUID:-$(id -u)}" -eq 0 ]; then
  echo "Error: Do not run this script with sudo!"
  echo "Example:  bash install.sh"
  echo "          ./install.sh"
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Python3 not found. Please install it manually."
  exit 1
fi

exec "$PY" install.py "$@"
