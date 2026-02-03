#!/bin/bash
set -e

echo "=== Resetting Docker environment ==="
cd "$(dirname "$0")/.."

echo "→ Stopping containers and removing volumes..."
docker compose down -v

echo "→ Starting fresh containers..."
docker compose up -d

echo "→ Waiting for services to be ready..."
sleep 10

echo ""
echo "=== Environment ready ==="
echo ""
echo "Docker stats baseline:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
echo ""
