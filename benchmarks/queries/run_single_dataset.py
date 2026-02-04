#!/usr/bin/env python3
import os
import sys
import json
import argparse
import subprocess
import threading
import time
import random
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vec3.query_chroma import run_queries as run_queries_chroma
from vec3.query_pgvector import run_queries as run_queries_pgvector
from vec3.metrics import compute_recall, compute_ground_truth

RESULTS_FILE = "results/raw/all_query_results.json"
NUM_QUERIES = 100
TOP_K_VALUES = [10, 50, 100]
CLASSES = ["A", "B", "C"]
WARMUP_QUERIES = 10  # Number of warmup queries after container restart

DATASET_MAP = {
    "10k": ("data/10k", 10000, 128),
    "50k": ("data/50k", 50000, 128),
    "100k": ("data/100k", 100000, 128),
    "500k": ("data/500k", 500000, 128),
    "1m": ("data/1m", 1000000, 128),
    "2m": ("data/2m", 2000000, 128),
    "50k_32d": ("data/50k_32d", 50000, 32),
    "100k_32d": ("data/100k_32d", 100000, 32),
    "100k_768d": ("data/100k_768d", 100000, 768),
    "50k_1536d": ("data/50k_1536d", 50000, 1536),
}


class DockerStatsMonitor:
    def __init__(self, container_name):
        self.container_name = container_name
        self.samples = []
        self.running = False
        self.thread = None

    def _collect(self):
        while self.running:
            try:
                result = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format",
                     "{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}",
                     self.container_name],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().split(",")
                    cpu = float(parts[0].replace("%", ""))
                    mem_percent = float(parts[2].replace("%", ""))
                    self.samples.append({
                        "cpu_percent": cpu,
                        "mem_percent": mem_percent,
                        "mem_usage": parts[1]
                    })
            except Exception:
                pass
            time.sleep(0.5)

    def start(self):
        self.samples = []
        self.running = True
        self.thread = threading.Thread(target=self._collect, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        return self.get_stats()

    def get_stats(self):
        if not self.samples:
            return {}
        cpu_vals = [s["cpu_percent"] for s in self.samples]
        mem_vals = [s["mem_percent"] for s in self.samples]
        return {
            "peak_mem_percent": max(mem_vals),
            "avg_mem_percent": sum(mem_vals) / len(mem_vals),
            "peak_mem_usage": max(self.samples, key=lambda x: x["mem_percent"])["mem_usage"],
            "avg_cpu_percent": sum(cpu_vals) / len(cpu_vals),
            "peak_cpu_percent": max(cpu_vals),
            "samples": len(self.samples)
        }


def restart_containers():
    print("  Restarting containers to clear cache...", end=" ", flush=True)
    try:
        subprocess.run(
            ["docker", "restart", "chroma_bench", "pgvector_bench"],
            capture_output=True, timeout=60
        )
        # Wait for containers to be ready
        time.sleep(5)
        
        # Wait for pgvector to accept connections
        for _ in range(30):
            try:
                result = subprocess.run(
                    ["docker", "exec", "pgvector_bench", "pg_isready", "-U", "user"],
                    capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    break
            except:
                pass
            time.sleep(1)
        
        # Wait for Chroma to be ready
        import chromadb
        for _ in range(30):
            try:
                client = chromadb.HttpClient(host="localhost", port=8000)
                client.heartbeat()
                break
            except:
                time.sleep(1)
        
        print("done")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def run_warmup(collection_name, table_name, queries, n_warmup=WARMUP_QUERIES):
    """Run warmup queries to warm up the database caches."""
    print(f"  Running {n_warmup} warmup queries...", end=" ", flush=True)
    
    warmup_queries = queries[:n_warmup]
    
    # Warmup Chroma
    try:
        run_queries_chroma(warmup_queries, collection_name=collection_name, n_results=10)
    except:
        pass
    
    # Warmup pgvector
    try:
        run_queries_pgvector(warmup_queries, table_name=table_name, n_results=10)
    except:
        pass
    
    print("done")


def generate_queries(dataset_path, num_queries, with_filter=False, compute_gt=True, max_k=100, metric="euclidean"):
    vectors_path = os.path.join(dataset_path, "vectors.npy")
    if not os.path.exists(vectors_path):
        raise FileNotFoundError(f"Vectors not found: {vectors_path}")
    
    all_vectors = np.load(vectors_path)
    indices = np.random.choice(all_vectors.shape[0], num_queries, replace=False)
    selected = all_vectors[indices]
    
    queries = []
    for i, vec in enumerate(selected):
        q = {"vector": vec.tolist(), "query_idx": i}
        if with_filter and (i % 2 == 0):
            q["filter"] = {"cls": random.choice(CLASSES)}
        queries.append(q)
    
    ground_truth = None
    if compute_gt:
        print(f"  Computing ground truth (k={max_k})...", end=" ", flush=True)
        ground_truth = compute_ground_truth(selected, all_vectors, k=max_k, metric=metric)
        print("done")
    
    return queries, ground_truth


def run_chroma_queries(queries, collection_name, top_k, monitor, ground_truth=None):
    monitor.start()
    try:
        if ground_truth is not None:
            stats, retrieved_ids = run_queries_chroma(
                queries, collection_name=collection_name, n_results=top_k, return_ids=True
            )
            # Compute recall for each query
            recalls = []
            for i, (retrieved, gt) in enumerate(zip(retrieved_ids, ground_truth)):
                recalls.append(compute_recall(retrieved, gt[:top_k]))
            stats["recall_mean"] = float(np.mean(recalls))
            stats["recall_min"] = float(np.min(recalls))
            stats["recall_max"] = float(np.max(recalls))
        else:
            stats = run_queries_chroma(queries, collection_name=collection_name, n_results=top_k)
    finally:
        docker_stats = monitor.stop()
    return stats, docker_stats


def run_pgvector_queries(queries, table_name, top_k, monitor, ground_truth=None, metric="euclidean"):
    monitor.start()
    try:
        if ground_truth is not None:
            stats, retrieved_ids = run_queries_pgvector(
                queries, table_name=table_name, n_results=top_k, metric=metric, return_ids=True
            )
            # Compute recall for each query
            # pgvector uses 1-indexed IDs, ground truth uses 0-indexed
            recalls = []
            for i, (retrieved, gt) in enumerate(zip(retrieved_ids, ground_truth)):
                # Convert 1-indexed pgvector IDs to 0-indexed
                retrieved_0idx = [rid - 1 for rid in retrieved]
                recalls.append(compute_recall(retrieved_0idx, gt[:top_k]))
            stats["recall_mean"] = float(np.mean(recalls))
            stats["recall_min"] = float(np.min(recalls))
            stats["recall_max"] = float(np.max(recalls))
        else:
            stats = run_queries_pgvector(queries, table_name=table_name, n_results=top_k, metric=metric)
    finally:
        docker_stats = monitor.stop()
    return stats, docker_stats


def run_query_benchmark(dataset_name, with_filter=False, index_type="ivfflat", 
                        index_params=None, metric="euclidean", restart=True):

    if dataset_name not in DATASET_MAP:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {list(DATASET_MAP.keys())}")
        return None
    
    dataset_path, num_vectors, dimensions = DATASET_MAP[dataset_name]
    mode = "filter" if with_filter else "nofilter"
    
    print(f"\n{'='*60}")
    print(f"Query Benchmark: {dataset_name} ({num_vectors:,} vectors, {dimensions}d)")
    print(f"Mode: {mode}, Index: {index_type}, Metric: {metric}")
    print(f"Queries: {NUM_QUERIES}, Top-K values: {TOP_K_VALUES}")
    print(f"{'='*60}")
    
    if restart:
        restart_containers()
    
    random.seed(42)
    np.random.seed(42)
    
    queries, ground_truth = generate_queries(
        dataset_path, NUM_QUERIES, with_filter=with_filter, 
        compute_gt=True, max_k=max(TOP_K_VALUES), metric=metric
    )
    
    collection_name = f"bench_{dataset_name}"  
    if index_type == "hnsw":
        table_name = f"vectors_{dataset_name}_hnsw"
    else:
        table_name = f"vectors_{dataset_name}_ivf"
    
    run_warmup(collection_name, table_name, queries)
    
    chroma_monitor = DockerStatsMonitor("chroma_bench")
    pg_monitor = DockerStatsMonitor("pgvector_bench")
    
    chroma_results = []
    pgvector_results = []
    
    for top_k in TOP_K_VALUES:
        print(f"\n--- Top-K = {top_k} ---")
        
        print(f"  Chroma ({collection_name})...", end=" ", flush=True)
        try:
            c_stats, c_docker = run_chroma_queries(
                queries, collection_name, top_k, chroma_monitor, ground_truth
            )
            recall_str = f", recall={c_stats.get('recall_mean', 0):.3f}" if 'recall_mean' in c_stats else ""
            print(f"mean={c_stats['mean']*1000:.2f}ms, p99={c_stats['p99']*1000:.2f}ms{recall_str}")
            chroma_results.append({
                "top_k": top_k,
                "num_queries": NUM_QUERIES,
                "latency": {k: v for k, v in c_stats.items() if k in ['mean', 'p50', 'p95', 'p99']},
                "recall": {k: v for k, v in c_stats.items() if 'recall' in k},
                "docker_stats": c_docker
            })
        except Exception as e:
            print(f"ERROR: {e}")
            chroma_results.append({"top_k": top_k, "error": str(e)})
        
        print(f"  pgvector ({table_name})...", end=" ", flush=True)
        try:
            p_stats, p_docker = run_pgvector_queries(
                queries, table_name, top_k, pg_monitor, ground_truth, metric
            )
            recall_str = f", recall={p_stats.get('recall_mean', 0):.3f}" if 'recall_mean' in p_stats else ""
            print(f"mean={p_stats['mean']*1000:.2f}ms, p99={p_stats['p99']*1000:.2f}ms{recall_str}")
            pgvector_results.append({
                "top_k": top_k,
                "num_queries": NUM_QUERIES,
                "latency": {k: v for k, v in p_stats.items() if k in ['mean', 'p50', 'p95', 'p99']},
                "recall": {k: v for k, v in p_stats.items() if 'recall' in k},
                "docker_stats": p_docker
            })
        except Exception as e:
            print(f"ERROR: {e}")
            pgvector_results.append({"top_k": top_k, "error": str(e)})
    
    return {
        "dataset": dataset_name,
        "dataset_path": dataset_path,
        "vectors": num_vectors,
        "dimensions": dimensions,
        "mode": mode,
        "index_type": index_type,
        "index_params": index_params or {},
        "metric": metric,
        "num_queries": NUM_QUERIES,
        "timestamp": datetime.now().isoformat(),
        "chroma": chroma_results,
        "pgvector": pgvector_results
    }


def load_existing_results(results_file=None):
    """Load existing results or create new structure."""
    path = results_file or RESULTS_FILE
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "experiment": "query_benchmark",
        "created": datetime.now().isoformat(),
        "runs": []
    }


def save_results(data, results_file=None):
    """Save results to file."""
    path = results_file or RESULTS_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run query benchmark for a single dataset")
    parser.add_argument("dataset", help=f"Dataset to benchmark: {list(DATASET_MAP.keys())}")
    parser.add_argument("--filter", action="store_true", help="Run with metadata filters")
    parser.add_argument("--both", action="store_true", help="Run both filter and nofilter modes")
    parser.add_argument("--fresh", action="store_true", help="Start fresh results file")
    parser.add_argument("-i", "--index-type", default="ivfflat", 
                        help="Index type (for tracking): ivfflat, hnsw")
    parser.add_argument("--hnsw-m", type=int, default=16, help="HNSW m parameter")
    parser.add_argument("--hnsw-ef", type=int, default=64, help="HNSW ef_construction parameter")
    parser.add_argument("--lists", type=int, default=100, help="IVFFlat lists parameter")
    parser.add_argument("--metric", default="euclidean", 
                        choices=["euclidean", "cosine", "inner_product"],
                        help="Distance metric")
    parser.add_argument("--no-restart", action="store_true", 
                        help="Skip container restart (for testing)")
    parser.add_argument("--results-file", type=str, default=None,
                        help="Custom results file path (default: results/raw/all_query_results.json)")
    args = parser.parse_args()
    
    results_file = args.results_file or RESULTS_FILE
    
    if args.fresh and os.path.exists(results_file):
        os.remove(results_file)
        print(f"Removed existing {results_file}")
    
    # Build index params based on type
    if args.index_type == "hnsw":
        index_params = {"m": args.hnsw_m, "ef_construction": args.hnsw_ef}
    else:
        index_params = {"lists": args.lists}
    
    modes = []
    if args.both:
        modes = [False, True]
    else:
        modes = [args.filter]
    
    all_data = load_existing_results(results_file=results_file)
    
    for with_filter in modes:
        result = run_query_benchmark(
            args.dataset, 
            with_filter=with_filter,
            index_type=args.index_type,
            index_params=index_params,
            metric=args.metric,
            restart=not args.no_restart
        )
        if result:
            existing_idx = None
            for i, run in enumerate(all_data["runs"]):
                if (run["dataset"] == result["dataset"] and 
                    run["mode"] == result["mode"] and
                    run.get("index_type") == result["index_type"] and
                    run.get("index_params") == result["index_params"] and
                    run.get("metric") == result["metric"]):
                    existing_idx = i
                    break
            
            if existing_idx is not None:
                all_data["runs"][existing_idx] = result
                print(f"\nUpdated existing run for {result['dataset']} ({result['mode']}, {result['index_type']})")
            else:
                all_data["runs"].append(result)
                print(f"\nAdded new run for {result['dataset']} ({result['mode']}, {result['index_type']})")
            
            save_results(all_data, results_file=results_file)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for with_filter in modes:
        mode = "filter" if with_filter else "nofilter"
        run = next((r for r in all_data["runs"] 
                   if r["dataset"] == args.dataset and r["mode"] == mode and
                   r.get("index_type") == args.index_type), None)
        if run and "chroma" in run:
            print(f"\n{args.dataset} ({mode}, {args.index_type}):")
            for c, p in zip(run["chroma"], run["pgvector"]):
                if "error" not in c and "error" not in p:
                    c_recall = c.get('recall', {}).get('recall_mean', 0)
                    p_recall = p.get('recall', {}).get('recall_mean', 0)
                    print(f"  top_k={c['top_k']:3d}: Chroma {c['latency']['mean']*1000:6.2f}ms (recall={c_recall:.3f}) | "
                          f"pgvector {p['latency']['mean']*1000:6.2f}ms (recall={p_recall:.3f})")


if __name__ == "__main__":
    main()
