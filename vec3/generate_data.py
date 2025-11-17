import numpy as np
import json
import os

def generate_dataset(size: int, dim: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    vectors = np.random.randn(size, dim).astype(np.float32)
    labels = np.random.choice(["A","B","C"], size=size, p=[0.5,0.3,0.2])

    np.save(f"{out_dir}/vectors.npy", vectors)

    with open(f"{out_dir}/metadata.jsonl", "w") as f:
        for cls in labels:
            f.write(json.dumps({"cls": cls}) + "\n")

    print(f"Generated dataset of size {size} at {out_dir}")
