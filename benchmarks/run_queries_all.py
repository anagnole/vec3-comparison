import os
import json
import random
import numpy as np

from vec3.plotting import plot_latency_bars
from vec3.query_chroma import run_queries as run_queries_chroma
from vec3.query_pgvector import run_queries as run_queries_pgvector

# We benchmark queries on the data ingested with this batch size
BATCH = 1000

DATASETS = [
    ("10k",  "data/10k/vectors.npy",  f"vec3_10k_b{BATCH}",  "vectors_10k"),
    ("100k", "data/100k/vectors.npy", f"vec3_100k_b{BATCH}", "vectors_100k"),
    ("200k", "data/200k/vectors.npy", f"vec3_200k_b{BATCH}", "vectors_200k"),
    ("500k", "data/500k/vectors.npy", f"vec3_500k_b{BATCH}", "vectors_500k"),
]

NUM_QUERIES = 100
PLOTS_DIR = "results/plots"
RAW_DIR = "results/raw"

CLASSES = ["A", "B", "C"]

def generate_test_queries(path: str, with_filters: bool):
    """
    Build a list of query dicts:
      {"vector": [...]} or {"vector": [...], "filter": {"cls": "A"}}
    """
    if not os.path.exists(path):
        print(f"Dataset {path} not found! Skipping.")
        return []

    all_vectors = np.load(path)
    indices = np.random.choice(all_vectors.shape[0], NUM_QUERIES, replace=False)
    selected_vectors = all_vectors[indices]

    queries = []
    for i, vec in enumerate(selected_vectors):
        q = {"vector": vec.tolist()}

        # If with_filters=True, put filter on ~50% queries (όπως έκανες πριν)
        if with_filters and (i % 2 == 0):
            q["filter"] = {"cls": random.choice(CLASSES)}

        queries.append(q)

    return queries


def run_one_mode(mode_tag: str, with_filters: bool):
    """
    Run one full benchmark mode (nofilter or filter).
    Saves plots + raw json with mode_tag in filename.
    """
    results = []

    print(f"\n--- Running query benchmarks mode='{mode_tag}' (batch={BATCH}) ---")

    for name, path, collection_name, pg_table in DATASETS:
        print(f"\n[Dataset: {name}] Generating queries from {path} (mode={mode_tag})...")
        queries = generate_test_queries(path, with_filters=with_filters)
        if not queries:
            continue

        print(f"Running {len(queries)} queries on Chroma (collection={collection_name})...")
        chroma_stats = run_queries_chroma(queries, collection_name=collection_name)
        print(f"Chroma {name}: mean={chroma_stats['mean']:.4f}s, p99={chroma_stats['p99']:.4f}s")

        print(f"Running {len(queries)} queries on Pgvector (table={pg_table})...")
        pg_stats = run_queries_pgvector(queries, table_name=pg_table, n_results=10)
        print(f"Pgvector {name}: mean={pg_stats['mean']:.4f}s, p99={pg_stats['p99']:.4f}s")

        results.append({"db": "chroma", "dataset": name, "mode": mode_tag, "stats": chroma_stats})
        results.append({"db": "pgvector", "dataset": name, "mode": mode_tag, "stats": pg_stats})

    if not results:
        print("No results collected; nothing to plot.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    # Mean latency bar chart
    mean_path = os.path.join(PLOTS_DIR, f"latency_mean_{mode_tag}_chroma_vs_pgvector_b{BATCH}.png")
    plot_latency_bars(
        results,
        metric="mean",
        title=f"Mean query latency by dataset size (mode={mode_tag}, batch={BATCH})",
        output_path=mean_path,
    )
    print(f"Saved mean latency plot -> {mean_path}")

    # P99 latency bar chart
    p99_path = os.path.join(PLOTS_DIR, f"latency_p99_{mode_tag}_chroma_vs_pgvector_b{BATCH}.png")
    plot_latency_bars(
        results,
        metric="p99",
        title=f"P99 query latency by dataset size (mode={mode_tag}, batch={BATCH})",
        output_path=p99_path,
    )
    print(f"Saved p99 latency plot -> {p99_path}")

    # Save raw JSON
    raw_path = os.path.join(RAW_DIR, f"query_results_{mode_tag}_b{BATCH}.json")
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved raw query results -> {raw_path}")


def main():

    SEED = 42
    
    # 1) No filters
    random.seed(SEED)
    np.random.seed(SEED)
    run_one_mode(mode_tag="nofilter", with_filters=False)

    # 2) With filters (cls)
    random.seed(SEED)
    np.random.seed(SEED)
    run_one_mode(mode_tag="filter", with_filters=True)


if __name__ == "__main__":
    main()
