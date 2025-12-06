import numpy as np
import random
import os
from vec3.query_chroma import run_queries

DATASETS = [
    ("10k", "data/10k/vectors.npy", "vec3_10k"),
    ("100k", "data/100k/vectors.npy", "vec3_100k"),
    ("500k", "data/500k/vectors.npy", "vec3_500k")
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
    print("--- Starting Chroma Benchmark for ALL sizes ---")
    
    for name, path, collection_name in DATASETS:
        print(f"\n[Dataset: {name}] Collection: {collection_name}")
        queries = generate_test_queries(path)
        if not queries: continue

        print(f"Running {len(queries)} queries...")
        
        try:
            stats = run_queries(queries, collection_name=collection_name)
            
            print(f"Results for {name}:")
            print(f"  Mean Latency: {stats['mean']:.4f} s")
            print(f"  P99 Latency:  {stats['p99']:.4f} s")
            if stats['mean'] > 0:
                print(f"  QPS:          {1 / stats['mean']:.2f}")
        except Exception as e:
            print(f"Error running {name}: {e}")

if __name__ == "__main__":
    main()
