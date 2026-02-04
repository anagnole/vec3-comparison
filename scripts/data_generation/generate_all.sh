#!/bin/bash

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Generating datasets..."

python - <<EOF
from vec3.generate_data import DatasetGenerator

gen = DatasetGenerator(seed=42)

configs = [
    # (size, dim, path)
    (10_000, 128, "data/10k"),
    (50_000, 128, "data/50k"),
    (100_000, 128, "data/100k"),
    (500_000, 128, "data/500k"),
    (1_000_000, 128, "data/1m"),
    (2_000_000, 128, "data/2m"),
    # Different dimensions
    (50_000, 32, "data/50k_32d"),
    (100_000, 32, "data/100k_32d"),
    (50_000, 1536, "data/50k_1536d"),
    (100_000, 768, "data/100k_768d"),
]

for size, dim, path in configs:
    gen.generate_dataset(size=size, dim=dim, out_dir=path)

print("All datasets generated successfully.")
EOF
