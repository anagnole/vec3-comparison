#!/usr/bin/env bash
set -euo pipefail

# Stop the background API and Vite servers started by start_ui.sh
# Usage: ./scripts/stop_ui.sh

ROOT=$(cd "$(dirname "$0")"/.. && pwd)
mkdir -p "$ROOT/tmp"

if [ -f "$ROOT/tmp/vec3_api.pid" ]; then
  PID=$(cat "$ROOT/tmp/vec3_api.pid")
  echo "Stopping API PID $PID"
  kill "$PID" || true
  rm -f "$ROOT/tmp/vec3_api.pid"
else
  echo "No API PID file"
fi

if [ -f "$ROOT/tmp/vite.pid" ]; then
  PID=$(cat "$ROOT/tmp/vite.pid")
  echo "Stopping Vite PID $PID"
  kill "$PID" || true
  rm -f "$ROOT/tmp/vite.pid"
else
  echo "No Vite PID file"
fi

echo "You can inspect logs at:"
echo "  $ROOT/tmp/vec3_api.log"
echo "  $ROOT/tmp/vite.log"
