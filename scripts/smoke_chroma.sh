#!/bin/bash
echo "Running chroma smoke test..."
source .venv/bin/activate
python benchmarks/smoke_chroma.py
