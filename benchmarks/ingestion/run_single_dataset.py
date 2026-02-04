#!/usr/bin/env python3
import os
import sys
import json
import argparse
import subprocess
import threading
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vec3.ingest_chroma import ingest_chroma
from vec3.ingest_pgvector import ingest_pgvector

BATCH_SIZES = [500, 1000, 5000]
RESULTS_FILE = "results/raw/all_ingestion_results.json"
DEFAULT_BATCH_SIZE = 1000

DATASET_MAP = {
    "10k": ("data/10k", 10000, 128),
    "50k": ("data/50k", 50000, 128),
    "100k": ("data/100k", 100000, 128),
    "500k": ("data/500k", 500000, 128),
    "1m": ("data/1m", 1000000, 128),
    "50k_32d": ("data/50k_32d", 50000, 32),
    "100k_32d": ("data/100k_32d", 100000, 32),
    "100k_768d": ("data/100k_768d", 100000, 768),
    "50k_1536d": ("data/50k_1536d", 50000, 1536),
    "2m": ("data/2m", 2000000, 128),
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
                    if len(parts) >= 3:
                        cpu = parts[0].replace("%", "")
                        mem_usage = parts[1].split("/")[0].strip()
                        mem_pct = parts[2].replace("%", "")
                        self.samples.append({
                            "timestamp": datetime.now().isoformat(),
                            "cpu_percent": float(cpu) if cpu else 0,
                            "mem_usage": mem_usage,
                            "mem_percent": float(mem_pct) if mem_pct else 0,
                        })
            except Exception:
                pass
            time.sleep(2)

    def start(self):
        self.samples = []
        self.running = True
        self.thread = threading.Thread(target=self._collect, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
        return self.get_summary()

    def get_summary(self):
        if not self.samples:
            return {"peak_mem_percent": 0, "avg_cpu_percent": 0, "samples": 0}
        return {
            "peak_mem_percent": max(s["mem_percent"] for s in self.samples),
            "peak_mem_usage": max(self.samples, key=lambda s: s["mem_percent"])["mem_usage"],
            "avg_cpu_percent": sum(s["cpu_percent"] for s in self.samples) / len(self.samples),
            "samples": len(self.samples),
        }


def load_results(results_file=None, fresh=False):
    """Load existing results or create new structure."""
    path = results_file or RESULTS_FILE
    if not fresh and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "experiment": "ingestion_benchmark",
        "created": datetime.now().isoformat(),
        "environment": {
            "chroma_mem_limit": "4g",
            "pgvector_mem_limit": "4g",
            "cpus_limit": "6",
        },
        "runs": []
    }


def save_results(data, results_file=None):
    """Save results to file."""
    path = results_file or RESULTS_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {path}")


def clear_chroma():
    import chromadb
    try:
        client = chromadb.HttpClient(host="localhost", port=8000)
        for col in client.list_collections():
            client.delete_collection(col.name)
        print("  Cleared all Chroma collections")
    except Exception as e:
        print(f"  Warning: Could not clear Chroma: {e}")


def run_benchmark(dataset_key, batch_size=1000, index_type="ivfflat", 
                  lists=100, hnsw_m=16, hnsw_ef_construction=64,
                  results_file=None, fresh=False):
    if dataset_key not in DATASET_MAP:
        print(f"Unknown dataset: {dataset_key}")
        print(f"Available: {list(DATASET_MAP.keys())}")
        sys.exit(1)

    dataset_path, num_vectors, dimensions = DATASET_MAP[dataset_key]

    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_key} ({num_vectors:,} vectors, {dimensions}d)")
    print(f"Index type: {index_type}, batch_size: {batch_size}")
    if index_type == "ivfflat":
        print(f"IVFFlat params: lists={lists}")
    else:
        print(f"HNSW params: m={hnsw_m}, ef_construction={hnsw_ef_construction}")
    print(f"{'='*60}")

    if index_type == "hnsw":
        index_params = {"m": hnsw_m, "ef_construction": hnsw_ef_construction}
        table_name = f"vectors_{dataset_key}_hnsw"
    else:
        index_params = {"lists": lists}
        table_name = f"vectors_{dataset_key}_ivf"

    run_data = {
        "dataset": dataset_key,
        "dataset_path": dataset_path,
        "vectors": num_vectors,
        "dimensions": dimensions,
        "timestamp": datetime.now().isoformat(),
        "chroma": [],
        "pgvector": [],
    }

    # Chroma
    print(f"\n[Chroma HNSW] Starting ingestion (m={hnsw_m}, ef={hnsw_ef_construction})...")
    clear_chroma()
    collection_name = f"bench_{dataset_key}"

    monitor = DockerStatsMonitor("chroma_bench")
    monitor.start()

    try:
        result = ingest_chroma(dataset_path, collection_name, batch_size=batch_size,
                               hnsw_m=hnsw_m, hnsw_ef_construction=hnsw_ef_construction)
        stats = monitor.stop()

        run_data["chroma"].append({
            "batch_size": batch_size,
            "duration_sec": round(result["duration_sec"], 2),
            "vectors_per_sec": round(result["vectors_per_sec"], 1),
            "storage_mb": round(result["storage_bytes"] / (1024*1024), 2) if result.get("storage_bytes") else None,
            "docker_stats": stats,
            "index_type": "hnsw",
            "index_params": {"m": hnsw_m, "ef_construction": hnsw_ef_construction},
        })
        print(f"  ✓ {result['duration_sec']:.2f}s, {result['vectors_per_sec']:.0f} v/s, "
              f"storage: {result.get('storage_bytes', 0) / (1024*1024):.1f} MB, "
              f"peak mem: {stats['peak_mem_usage']}")
    except Exception as e:
        monitor.stop()
        print(f"  ✗ ERROR: {e}")
        run_data["chroma"].append({"batch_size": batch_size, "error": str(e), "index_type": "hnsw", "index_params": {"built_in": True}})

    # pgvector
    print(f"\n[pgvector {index_type}] Starting ingestion...")

    monitor = DockerStatsMonitor("pgvector_bench")
    monitor.start()

    try:
        result = ingest_pgvector(dataset_path, batch_size=batch_size, table_name=table_name,
                                  index_type=index_type, lists=lists,
                                  hnsw_m=hnsw_m, hnsw_ef_construction=hnsw_ef_construction)
        stats = monitor.stop()

        run_data["pgvector"].append({
            "batch_size": batch_size,
            "duration_sec": round(result["duration_sec"], 2),
            "duration_ingest_sec": round(result["duration_ingest_sec"], 2),
            "duration_index_sec": round(result["duration_index_sec"], 2) if result.get("duration_index_sec") else None,
            "vectors_per_sec": round(result["vectors_per_sec"], 1),
            "storage_mb": round(result["storage_bytes"] / (1024*1024), 2),
            "storage_before_index_mb": round(result["storage_before_index_bytes"] / (1024*1024), 2),
            "docker_stats": stats,
            "index_type": index_type,
            "index_params": index_params,
        })
        print(f"  ✓ {result['duration_sec']:.2f}s (insert: {result['duration_ingest_sec']:.2f}s, "
              f"index: {result.get('duration_index_sec', 0):.2f}s), "
              f"{result['vectors_per_sec']:.0f} v/s, "
              f"storage: {result['storage_bytes'] / (1024*1024):.1f} MB, "
              f"peak mem: {stats['peak_mem_usage']}")
    except Exception as e:
        monitor.stop()
        print(f"  ✗ ERROR: {e}")
        run_data["pgvector"].append({"batch_size": batch_size, "error": str(e), "index_type": index_type, "index_params": index_params})

    # Save results
    all_results = load_results(results_file=results_file, fresh=fresh)
    
    # Create unique key for this run (dataset + index_type + params)
    run_key = f"{dataset_key}_{index_type}_{batch_size}"
    if index_type == "hnsw":
        run_key += f"_m{hnsw_m}_ef{hnsw_ef_construction}"
    else:
        run_key += f"_lists{lists}"
    
    run_data["run_key"] = run_key
    
    # Remove any existing run with same key (to allow re-runs)
    all_results["runs"] = [r for r in all_results["runs"] if r.get("run_key") != run_key]
    all_results["runs"].append(run_data)
    all_results["last_updated"] = datetime.now().isoformat()
    
    save_results(all_results, results_file=results_file)

    print(f"\n{'='*60}")
    print(f"COMPLETED: {dataset_key} (index_type={index_type})")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Run ingestion benchmark for a single dataset")
    parser.add_argument("--dataset", "-d", required=True,
                        help=f"Dataset to benchmark: {list(DATASET_MAP.keys())}")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available datasets")
    parser.add_argument("--batch-size", "-b", type=int, default=1000,
                        help="Batch size (default: 1000)")
    parser.add_argument("--index-type", "-i", default="ivfflat", choices=["ivfflat", "hnsw"],
                        help="pgvector index type (default: ivfflat)")
    # IVFFlat params
    parser.add_argument("--lists", type=int, default=100,
                        help="IVFFlat lists parameter (default: 100)")
    # HNSW params (used for both Chroma and pgvector)
    parser.add_argument("--hnsw-m", type=int, default=16,
                        help="HNSW M parameter (default: 16)")
    parser.add_argument("--hnsw-ef", type=int, default=64,
                        help="HNSW ef_construction parameter (default: 64)")
    # Output options
    parser.add_argument("--results-file", type=str, default=None,
                        help="Custom results file path (default: results/raw/all_ingestion_results.json)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh results file (ignore existing)")
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for key, (path, n, dim) in DATASET_MAP.items():
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"  {exists} {key}: {n:,} vectors, {dim}d ({path})")
        sys.exit(0)

    run_benchmark(args.dataset, batch_size=args.batch_size, index_type=args.index_type,
                  lists=args.lists, hnsw_m=args.hnsw_m, hnsw_ef_construction=args.hnsw_ef,
                  results_file=args.results_file, fresh=args.fresh)


if __name__ == "__main__":
    main()