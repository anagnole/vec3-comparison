import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vec3.ingest_chroma import ingest_chroma
from vec3.ingest_pgvector import ingest_pgvector

DATASETS = [
    ("data/10k", "bench_10k"),
    ("data/100k", "bench_100k"),
    ("data/500k", "bench_500k"),
    ("data/100k_768d", "bench_100k_768d"),
    ("data/50k_1536d", "bench_50k_1536d"),
]

BATCH_SIZE = 1000
OUTPUT_DIR = "results/raw"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_results = []
    
    for dataset_dir, collection_name in DATASETS:
        if not os.path.exists(dataset_dir):
            print(f"Skipping {dataset_dir} - not found")
            continue
            
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_dir}")
        print(f"{'='*60}")
        
        print(f"\n[CHROMA] Ingesting {dataset_dir}...")
        try:
            r_chroma = ingest_chroma(dataset_dir, collection_name, batch_size=BATCH_SIZE)
            all_results.append(r_chroma)
            print(f"  Result: {r_chroma['vectors']} vectors, {r_chroma['duration_sec']:.2f}s, {r_chroma['vectors_per_sec']:.0f} v/s")
            if r_chroma.get('storage_bytes'):
                print(f"  Storage: {r_chroma['storage_bytes']/(1024*1024):.2f} MB")
        except Exception as e:
            print(f"  ERROR: {e}")
        
        print(f"\n[PGVECTOR] Ingesting {dataset_dir}...")
        try:
            r_pg = ingest_pgvector(dataset_dir, batch_size=BATCH_SIZE, create_index=True)
            all_results.append(r_pg)
            print(f"  Result: {r_pg['vectors']} vectors, {r_pg['duration_sec']:.2f}s, {r_pg['vectors_per_sec']:.0f} v/s")
            if r_pg.get('storage_bytes'):
                print(f"  Storage: {r_pg['storage_bytes']/(1024*1024):.2f} MB")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    output_path = os.path.join(OUTPUT_DIR, "ingestion_benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved results to {output_path}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Dataset':<20} {'DB':<10} {'Vectors':>10} {'Dim':>6} {'Time(s)':>10} {'V/s':>12} {'Storage(MB)':>12}")
    print("-"*80)
    for r in all_results:
        storage_mb = r.get('storage_bytes', 0) or 0
        storage_str = f"{storage_mb/(1024*1024):.2f}" if storage_mb else "N/A"
        print(f"{os.path.basename(r['dataset_dir']):<20} {r['db']:<10} {r['vectors']:>10} {r.get('dimensions', 'N/A'):>6} {r['duration_sec']:>10.2f} {r['vectors_per_sec']:>12.0f} {storage_str:>12}")


if __name__ == "__main__":
    main()
