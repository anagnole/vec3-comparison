#!/usr/bin/env bash
set -euo pipefail

# Start both the API and the Vite web UI in background and save PIDs/logs.

ROOT=$(cd "$(dirname "$0")"/.. && pwd)
echo "Project root: $ROOT"

if [ -f "$ROOT/.venv/bin/activate" ]; then
  echo "Activating .venv..."
  source "$ROOT/.venv/bin/activate"
fi

echo "Installing API dependencies..."
cd "$ROOT/api"
npm install --silent

echo "Starting API (background)..."
cd "$ROOT/api"
nohup npm start > "$ROOT/tmp/vec3_api.log" 2>&1 &
API_PID=$!
echo "$API_PID" > "$ROOT/tmp/vec3_api.pid"
echo "API PID: $API_PID (log: tmp/vec3_api.log)"

echo "Installing web dependencies..."
cd "$ROOT/web"
npm install --silent

echo "Starting Vite dev server (background)..."
cd "$ROOT/web"
nohup npm run dev -- --host > "$ROOT/tmp/vite.log" 2>&1 &
VITE_PID=$!
echo "$VITE_PID" > "$ROOT/tmp/vite.pid"
echo "Vite PID: $VITE_PID (log: tmp/vite.log)"

echo "Done. UI: http://localhost:5173/  API: http://localhost:3001/"
