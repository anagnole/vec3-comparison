#!/bin/bash
echo "Running pgvector smoke test..."
source .venv/bin/activate
python benchmarks/smoke_pgvector.py
