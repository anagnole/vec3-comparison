import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vec3.ingest_chroma import ingest_chroma
from vec3.ingest_pgvector import ingest_pgvector

DATASETS = [
    ("data/10k", "bench_10k"),
    ("data/100k", "bench_100k"),
    ("data/500k", "bench_500k"),
    ("data/1m", "bench_1m"),
    ("data/100k_768d", "bench_100k_768d"),
    ("data/50k_1536d", "bench_50k_1536d"),
]

BATCH_SIZES = [500, 1000, 5000]
NUM_RUNS = 3
OUTPUT_DIR = "results/raw"


def clear_chroma_collections():
    import chromadb
    client = chromadb.HttpClient(host="localhost", port=8000)
    for col in client.list_collections():
        client.delete_collection(col.name)
    time.sleep(1)


def run_single_benchmark(dataset_dir, collection_name, batch_size, run_num):
    results = {"run": run_num, "batch_size": batch_size}
    
    print(f"\n  [Run {run_num}] Chroma (batch={batch_size})...")
    try:
        r_chroma = ingest_chroma(dataset_dir, f"{collection_name}_b{batch_size}_r{run_num}", batch_size=batch_size)
        results["chroma"] = r_chroma
        print(f"    → {r_chroma['vectors']} vectors in {r_chroma['duration_sec']:.2f}s ({r_chroma['vectors_per_sec']:.0f} v/s)")
    except Exception as e:
        print(f"    → ERROR: {e}")
        results["chroma"] = {"error": str(e)}
    
    print(f"  [Run {run_num}] pgvector (batch={batch_size})...")
    try:
        r_pg = ingest_pgvector(dataset_dir, batch_size=batch_size, create_index=True)
        results["pgvector"] = r_pg
        print(f"    → {r_pg['vectors']} vectors in {r_pg['duration_sec']:.2f}s ({r_pg['vectors_per_sec']:.0f} v/s)")
    except Exception as e:
        print(f"    → ERROR: {e}")
        results["pgvector"] = {"error": str(e)}
    
    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {
        "timestamp": timestamp,
        "config": {
            "batch_sizes": BATCH_SIZES,
            "num_runs": NUM_RUNS,
            "datasets": [d[0] for d in DATASETS],
        },
        "benchmarks": []
    }
    
    for dataset_dir, collection_name in DATASETS:
        if not os.path.exists(dataset_dir):
            print(f"\n{'='*60}")
            print(f"SKIPPING: {dataset_dir} (not found)")
            print(f"{'='*60}")
            continue
        
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_dir}")
        print(f"{'='*60}")
        
        dataset_results = {
            "dataset": dataset_dir,
            "runs": []
        }
        
        for batch_size in BATCH_SIZES:
            for run_num in range(1, NUM_RUNS + 1):
                result = run_single_benchmark(dataset_dir, collection_name, batch_size, run_num)
                dataset_results["runs"].append(result)
        
        all_results["benchmarks"].append(dataset_results)
        
        interim_path = os.path.join(OUTPUT_DIR, f"ingestion_benchmark_{timestamp}_interim.json")
        with open(interim_path, "w") as f:
            json.dump(all_results, f, indent=2)
    
    final_path = os.path.join(OUTPUT_DIR, f"ingestion_benchmark_{timestamp}.json")
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {final_path}")
    
    print_summary(all_results)
    
    return final_path


def print_summary(results):
    print(f"\n{'='*100}")
    print("SUMMARY (averaged across runs)")
    print(f"{'='*100}")
    print(f"{'Dataset':<15} {'Batch':>6} | {'Chroma Time':>12} {'Chroma V/s':>12} {'Chroma MB':>10} | {'PG Time':>10} {'PG V/s':>10} {'PG MB':>8}")
    print("-"*100)
    
    for benchmark in results["benchmarks"]:
        dataset = os.path.basename(benchmark["dataset"])
        
        by_batch = {}
        for run in benchmark["runs"]:
            bs = run["batch_size"]
            if bs not in by_batch:
                by_batch[bs] = {"chroma": [], "pgvector": []}
            
            if "chroma" in run and "error" not in run["chroma"]:
                by_batch[bs]["chroma"].append(run["chroma"])
            if "pgvector" in run and "error" not in run["pgvector"]:
                by_batch[bs]["pgvector"].append(run["pgvector"])
        
        for bs in sorted(by_batch.keys()):
            chroma_runs = by_batch[bs]["chroma"]
            pg_runs = by_batch[bs]["pgvector"]
            
            if chroma_runs:
                c_time = sum(r["duration_sec"] for r in chroma_runs) / len(chroma_runs)
                c_vps = sum(r["vectors_per_sec"] for r in chroma_runs) / len(chroma_runs)
                c_storage = [r.get("storage_bytes") for r in chroma_runs if r.get("storage_bytes")]
                c_mb = (sum(c_storage) / len(c_storage) / (1024*1024)) if c_storage else 0
            else:
                c_time = c_vps = c_mb = 0
            
            if pg_runs:
                p_time = sum(r["duration_sec"] for r in pg_runs) / len(pg_runs)
                p_vps = sum(r["vectors_per_sec"] for r in pg_runs) / len(pg_runs)
                p_mb = sum(r["storage_bytes"] for r in pg_runs) / len(pg_runs) / (1024*1024)
            else:
                p_time = p_vps = p_mb = 0
            
            print(f"{dataset:<15} {bs:>6} | {c_time:>12.2f} {c_vps:>12.0f} {c_mb:>10.2f} | {p_time:>10.2f} {p_vps:>10.0f} {p_mb:>8.2f}")


if __name__ == "__main__":
    main()
