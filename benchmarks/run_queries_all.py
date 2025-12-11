import os
import json

import numpy as np

from vec3.plotting import plot_latency_bars
from vec3.query_chroma import run_queries as run_queries_chroma
from vec3.query_pgvector import run_queries as run_queries_pgvector

DATASETS = [
    ("10k", "data/10k/vectors.npy", "vec3_10k"),
    ("100k", "data/100k/vectors.npy", "vec3_100k"),
    ("500k", "data/500k/vectors.npy", "vec3_500k"),
]

NUM_QUERIES = 100
PLOTS_DIR = "results/plots"
RAW_DIR = "results/raw"


def generate_test_queries(path):
    if not os.path.exists(path):
        print(f"Dataset {path} not found! Skipping.")
        return []

    all_vectors = np.load(path)
    indices = np.random.choice(all_vectors.shape[0], NUM_QUERIES, replace=False)
    selected_vectors = all_vectors[indices]

    queries = []
    for vec in selected_vectors:
        q = {"vector": vec.tolist()}
        queries.append(q)
    return queries


def main():
    results = []

    print("--- Running Chroma & Pgvector query benchmarks ---")

    for name, path, collection_name in DATASETS:
        print(f"\n[Dataset: {name}] Generating queries from {path}...")
        queries = generate_test_queries(path)
        if not queries:
            continue

        print(f"Running {len(queries)} queries on Chroma (collection={collection_name})...")
        chroma_stats = run_queries_chroma(queries, collection_name=collection_name)
        print(
            f"Chroma {name}: mean={chroma_stats['mean']:.4f}s, p99={chroma_stats['p99']:.4f}s"
        )

        print(f"Running {len(queries)} queries on Pgvector (table=vectors)...")
        pg_stats = run_queries_pgvector(queries, table_name="vectors")
        print(f"Pgvector {name}: mean={pg_stats['mean']:.4f}s, p99={pg_stats['p99']:.4f}s")

        results.append({"db": "chroma", "dataset": name, "stats": chroma_stats})
        results.append({"db": "pgvector", "dataset": name, "stats": pg_stats})

    if not results:
        print("No results collected; nothing to plot.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    # Mean latency bar chart
    mean_path = os.path.join(PLOTS_DIR, "latency_mean_chroma_vs_pgvector.png")
    plot_latency_bars(
        results,
        metric="mean",
        title="Mean query latency by dataset size",
        output_path=mean_path,
    )
    print(f"Saved mean latency plot -> {mean_path}")

    # P99 latency bar chart
    p99_path = os.path.join(PLOTS_DIR, "latency_p99_chroma_vs_pgvector.png")
    plot_latency_bars(
        results,
        metric="p99",
        title="P99 query latency by dataset size",
        output_path=p99_path,
    )
    print(f"Saved p99 latency plot -> {p99_path}")

    # Save raw query benchmark results as JSON.
    raw_path = os.path.join(RAW_DIR, "query_results.json")
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved raw query results -> {raw_path}")


if __name__ == "__main__":
    main()
