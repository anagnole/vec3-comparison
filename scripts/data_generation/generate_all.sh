#!/bin/bash

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Generating datasets (10k, 100k, 200k, 500k)..."

python - <<EOF
from vec3.generate_data import DatasetGenerator

gen = DatasetGenerator(seed=42)

configs = [
    (10_000, 100, "data/10k"),
    (100_000, 100, "data/100k"),
    (200_000, 100, "data/200k"),
    (500_000, 100, "data/500k"),
]

for size, dim, path in configs:
    gen.generate_dataset(size=size, dim=dim, out_dir=path)

print("All datasets generated successfully.")
EOF
