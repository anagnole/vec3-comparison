import json
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = "results/raw/ingestion_benchmark_results.json"
PLOTS_DIR = "results/plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

with open(RESULTS_FILE) as f:
    results = json.load(f)

datasets_128d = ["10k", "100k", "500k"]
datasets_highdim = ["100k_768d", "50k_1536d"]

def get_result(db, dataset_suffix):
    for r in results:
        if r["db"] == db and r["dataset_dir"].endswith(dataset_suffix):
            return r
    return None

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

chroma_times = [get_result("chroma", d)["duration_sec"] for d in datasets_128d]
pg_times = [get_result("pgvector", d)["duration_sec"] for d in datasets_128d]
pg_ingest = [get_result("pgvector", d)["duration_ingest_sec"] for d in datasets_128d]
pg_index = [get_result("pgvector", d)["duration_index_sec"] for d in datasets_128d]

x = np.arange(len(datasets_128d))
width = 0.35

ax1 = axes[0, 0]
ax1.bar(x - width/2, chroma_times, width, label='Chroma', color='#2ecc71')
ax1.bar(x + width/2, pg_ingest, width, label='pgvector (insert)', color='#3498db')
ax1.bar(x + width/2, pg_index, width, bottom=pg_ingest, label='pgvector (index)', color='#e74c3c')
ax1.set_xlabel('Dataset')
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Ingestion Time (128 dimensions)')
ax1.set_xticks(x)
ax1.set_xticklabels(datasets_128d)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

chroma_vps = [get_result("chroma", d)["vectors_per_sec"] for d in datasets_128d]
pg_vps = [get_result("pgvector", d)["vectors_per_sec"] for d in datasets_128d]

ax2 = axes[0, 1]
ax2.bar(x - width/2, chroma_vps, width, label='Chroma', color='#2ecc71')
ax2.bar(x + width/2, pg_vps, width, label='pgvector', color='#3498db')
ax2.set_xlabel('Dataset')
ax2.set_ylabel('Vectors per second')
ax2.set_title('Ingestion Throughput (128 dimensions)')
ax2.set_xticks(x)
ax2.set_xticklabels(datasets_128d)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

pg_storage = [get_result("pgvector", d)["storage_bytes"]/(1024*1024) for d in datasets_128d]
raw_data_sizes = [10000*128*4/(1024*1024), 100000*128*4/(1024*1024), 500000*128*4/(1024*1024)]

ax3 = axes[1, 0]
ax3.bar(x - width/2, raw_data_sizes, width, label='Raw data size', color='#95a5a6')
ax3.bar(x + width/2, pg_storage, width, label='pgvector (table+index)', color='#3498db')
ax3.set_xlabel('Dataset')
ax3.set_ylabel('Size (MB)')
ax3.set_title('Storage Comparison (128 dimensions)')
ax3.set_xticks(x)
ax3.set_xticklabels(datasets_128d)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

dims = [128, 768, 1536]
chroma_times_dim = [
    get_result("chroma", "100k")["duration_sec"],
    get_result("chroma", "100k_768d")["duration_sec"],
    get_result("chroma", "50k_1536d")["duration_sec"] * 2
]
pg_times_dim = [
    get_result("pgvector", "100k")["duration_sec"],
    get_result("pgvector", "100k_768d")["duration_sec"],
    get_result("pgvector", "50k_1536d")["duration_sec"] * 2
]

x_dim = np.arange(len(dims))
ax4 = axes[1, 1]
ax4.bar(x_dim - width/2, chroma_times_dim, width, label='Chroma', color='#2ecc71')
ax4.bar(x_dim + width/2, pg_times_dim, width, label='pgvector', color='#3498db')
ax4.set_xlabel('Vector Dimensions')
ax4.set_ylabel('Time (seconds, normalized to 100k vectors)')
ax4.set_title('Dimension Impact on Ingestion Time')
ax4.set_xticks(x_dim)
ax4.set_xticklabels(dims)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ingestion_benchmark_summary.png"), dpi=150)
plt.close()

print("Generated: results/plots/ingestion_benchmark_summary.png")

fig2, ax = plt.subplots(figsize=(10, 6))

pg_data_only = [get_result("pgvector", d)["storage_before_index_bytes"]/(1024*1024) for d in datasets_128d]
pg_index_size = [(get_result("pgvector", d)["storage_bytes"] - get_result("pgvector", d)["storage_before_index_bytes"])/(1024*1024) for d in datasets_128d]

ax.bar(x, pg_data_only, width*0.8, label='Table data', color='#3498db')
ax.bar(x, pg_index_size, width*0.8, bottom=pg_data_only, label='IVFFLAT index', color='#e74c3c')
ax.plot(x, raw_data_sizes, 'ko--', label='Raw data size', markersize=8)
ax.set_xlabel('Dataset')
ax.set_ylabel('Size (MB)')
ax.set_title('pgvector Storage Breakdown vs Raw Data')
ax.set_xticks(x)
ax.set_xticklabels(datasets_128d)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "pgvector_storage_breakdown.png"), dpi=150)
plt.close()

print("Generated: results/plots/pgvector_storage_breakdown.png")

print("\n" + "="*90)
print("BENCHMARK RESULTS SUMMARY")
print("="*90)
print(f"{'Dataset':<15} {'Dim':>5} {'Vectors':>10} | {'Chroma':>10} {'pgvector':>10} | {'Speedup':>8} | {'PG Storage':>12}")
print(f"{'':15} {'':>5} {'':>10} | {'(sec)':>10} {'(sec)':>10} | {'':>8} | {'(MB)':>12}")
print("-"*90)

for r in results:
    if r["db"] == "chroma":
        dataset = os.path.basename(r["dataset_dir"])
        pg_r = get_result("pgvector", dataset)
        if pg_r:
            speedup = pg_r["duration_sec"] / r["duration_sec"]
            storage_mb = pg_r["storage_bytes"] / (1024*1024)
            print(f"{dataset:<15} {r['dimensions']:>5} {r['vectors']:>10} | {r['duration_sec']:>10.2f} {pg_r['duration_sec']:>10.2f} | {speedup:>7.2f}x | {storage_mb:>12.2f}")
