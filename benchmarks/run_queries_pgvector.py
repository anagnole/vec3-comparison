import numpy as np
import random
import os
from vec3.query_pgvector import run_queries

DATASETS = [
    ("10k", "data/10k/vectors.npy"),
    ("100k", "data/100k/vectors.npy"),
    ("200k", "data/200k/vectors.npy"),
    ("500k", "data/500k/vectors.npy"),
]
NUM_QUERIES = 100
CLASSES = ["A", "B", "C"] 

def generate_test_queries(path):
    if not os.path.exists(path):
        print(f"Dataset {path} not found! Skipping.")
        return []

    all_vectors = np.load(path)
    indices = np.random.choice(all_vectors.shape[0], NUM_QUERIES, replace=False)
    selected_vectors = all_vectors[indices]
    
    queries = []
    for i, vec in enumerate(selected_vectors):
        q = {"vector": vec.tolist()}
        if i % 2 == 0:
            q["filter"] = {"cls": random.choice(CLASSES)}
        queries.append(q)
    return queries

def main():
    print("--- Starting Pgvector Benchmark for ALL sizes ---")
    
    for name, path in DATASETS:
        print(f"\n[Dataset: {name}] Generating queries from {path}...")
        queries = generate_test_queries(path)
        if not queries: continue

        table_name = f"vectors_{name}"

        print(f"Running {len(queries)} queries on Pgvector (table={table_name})...")
        stats = run_queries(queries, table_name=table_name)
        
        print(f"Results for {name}:")
        print(f"  Mean Latency: {stats['mean']:.4f} s")
        print(f"  P99 Latency:  {stats['p99']:.4f} s")
        if stats['mean'] > 0:
            print(f"  QPS:          {1 / stats['mean']:.2f}")

if __name__ == "__main__":
    main()
