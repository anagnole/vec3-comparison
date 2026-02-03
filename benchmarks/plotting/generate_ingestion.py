import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_DIR = "results/raw"
PLOTS_DIR = "results/plots"


def load_latest_results():
    files = glob.glob(os.path.join(RESULTS_DIR, "ingestion_benchmark_*.json"))
    files = [f for f in files if "interim" not in f]
    if not files:
        raise FileNotFoundError("No benchmark results found")
    latest = max(files, key=os.path.getmtime)
    print(f"Loading: {latest}")
    with open(latest) as f:
        return json.load(f)


def aggregate_results(results):
    aggregated = defaultdict(lambda: defaultdict(lambda: {"chroma": [], "pgvector": []}))
    
    for benchmark in results["benchmarks"]:
        dataset = os.path.basename(benchmark["dataset"])
        for run in benchmark["runs"]:
            bs = run["batch_size"]
            if "chroma" in run and "error" not in run["chroma"]:
                aggregated[dataset][bs]["chroma"].append(run["chroma"])
            if "pgvector" in run and "error" not in run["pgvector"]:
                aggregated[dataset][bs]["pgvector"].append(run["pgvector"])
    
    return aggregated


def mean_std(values):
    if not values:
        return 0, 0
    return np.mean(values), np.std(values)


def plot_ingestion_time_comparison(aggregated, batch_size=1000):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    datasets_128d = [d for d in aggregated.keys() if "768" not in d and "1536" not in d]
    datasets_128d = sorted(datasets_128d, key=lambda x: int(x.replace("k", "000").replace("m", "000000")) if x[0].isdigit() else 0)
    
    if not datasets_128d:
        print("No 128-dim datasets found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(datasets_128d))
    width = 0.35
    
    chroma_times = []
    chroma_errs = []
    pg_ingest_times = []
    pg_ingest_errs = []
    pg_index_times = []
    
    for dataset in datasets_128d:
        data = aggregated[dataset][batch_size]
        
        c_runs = data["chroma"]
        if c_runs:
            times = [r["duration_sec"] for r in c_runs]
            m, s = mean_std(times)
            chroma_times.append(m)
            chroma_errs.append(s)
        else:
            chroma_times.append(0)
            chroma_errs.append(0)
        
        p_runs = data["pgvector"]
        if p_runs:
            ingest_times = [r["duration_ingest_sec"] for r in p_runs]
            index_times = [r.get("duration_index_sec", 0) or 0 for r in p_runs]
            m_i, s_i = mean_std(ingest_times)
            m_idx, _ = mean_std(index_times)
            pg_ingest_times.append(m_i)
            pg_ingest_errs.append(s_i)
            pg_index_times.append(m_idx)
        else:
            pg_ingest_times.append(0)
            pg_ingest_errs.append(0)
            pg_index_times.append(0)
    
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, chroma_times, width, yerr=chroma_errs, label='Chroma (HNSW)', color='#2ecc71', capsize=3)
    bars2 = ax1.bar(x + width/2, pg_ingest_times, width, yerr=pg_ingest_errs, label='pgvector (insert)', color='#3498db', capsize=3)
    bars3 = ax1.bar(x + width/2, pg_index_times, width, bottom=pg_ingest_times, label='pgvector (IVFFlat index)', color='#e74c3c')
    
    ax1.set_xlabel('Dataset Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title(f'Ingestion Time Comparison (batch_size={batch_size})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets_128d)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    chroma_vps = []
    pg_vps = []
    chroma_vps_err = []
    pg_vps_err = []
    
    for dataset in datasets_128d:
        data = aggregated[dataset][batch_size]
        
        c_runs = data["chroma"]
        if c_runs:
            vps = [r["vectors_per_sec"] for r in c_runs]
            m, s = mean_std(vps)
            chroma_vps.append(m)
            chroma_vps_err.append(s)
        else:
            chroma_vps.append(0)
            chroma_vps_err.append(0)
        
        p_runs = data["pgvector"]
        if p_runs:
            vps = [r["vectors_per_sec"] for r in p_runs]
            m, s = mean_std(vps)
            pg_vps.append(m)
            pg_vps_err.append(s)
        else:
            pg_vps.append(0)
            pg_vps_err.append(0)
    
    ax2 = axes[1]
    ax2.bar(x - width/2, chroma_vps, width, yerr=chroma_vps_err, label='Chroma', color='#2ecc71', capsize=3)
    ax2.bar(x + width/2, pg_vps, width, yerr=pg_vps_err, label='pgvector', color='#3498db', capsize=3)
    ax2.set_xlabel('Dataset Size')
    ax2.set_ylabel('Vectors per Second')
    ax2.set_title(f'Ingestion Throughput (batch_size={batch_size})')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets_128d)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"ingestion_comparison_batch{batch_size}.png"), dpi=150)
    plt.close()
    print(f"Saved: ingestion_comparison_batch{batch_size}.png")


def plot_storage_comparison(aggregated, batch_size=1000):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    datasets_128d = [d for d in aggregated.keys() if "768" not in d and "1536" not in d]
    datasets_128d = sorted(datasets_128d, key=lambda x: int(x.replace("k", "000").replace("m", "000000")) if x[0].isdigit() else 0)
    
    if not datasets_128d:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(datasets_128d))
    width = 0.25
    
    raw_sizes = []
    chroma_sizes = []
    pg_data_sizes = []
    pg_index_sizes = []
    
    for dataset in datasets_128d:
        data = aggregated[dataset][batch_size]
        
        c_runs = data["chroma"]
        if c_runs and c_runs[0].get("storage_bytes"):
            storage = [r["storage_bytes"] for r in c_runs if r.get("storage_bytes")]
            chroma_sizes.append(np.mean(storage) / (1024*1024) if storage else 0)
            n = c_runs[0]["vectors"]
            dim = c_runs[0]["dimensions"]
            raw_sizes.append(n * dim * 4 / (1024*1024))
        else:
            chroma_sizes.append(0)
            raw_sizes.append(0)
        
        p_runs = data["pgvector"]
        if p_runs:
            before_idx = [r["storage_before_index_bytes"] for r in p_runs]
            total = [r["storage_bytes"] for r in p_runs]
            pg_data_sizes.append(np.mean(before_idx) / (1024*1024))
            pg_index_sizes.append((np.mean(total) - np.mean(before_idx)) / (1024*1024))
        else:
            pg_data_sizes.append(0)
            pg_index_sizes.append(0)
    
    ax.bar(x - width, raw_sizes, width, label='Raw data (float32)', color='#95a5a6')
    ax.bar(x, chroma_sizes, width, label='Chroma (HNSW)', color='#2ecc71')
    ax.bar(x + width, pg_data_sizes, width, label='pgvector (table)', color='#3498db')
    ax.bar(x + width, pg_index_sizes, width, bottom=pg_data_sizes, label='pgvector (index)', color='#e74c3c')
    
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Storage (MB)')
    ax.set_title('Storage Comparison: Raw Data vs Database Storage')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_128d)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "storage_comparison.png"), dpi=150)
    plt.close()
    print("Saved: storage_comparison.png")


def plot_batch_size_impact(aggregated):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    test_dataset = None
    for d in ["100k", "500k", "10k"]:
        if d in aggregated:
            test_dataset = d
            break
    
    if not test_dataset:
        print("No suitable dataset for batch size analysis")
        return
    
    batch_sizes = sorted(aggregated[test_dataset].keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    chroma_times = []
    pg_times = []
    chroma_errs = []
    pg_errs = []
    
    for bs in batch_sizes:
        data = aggregated[test_dataset][bs]
        
        c_runs = data["chroma"]
        if c_runs:
            times = [r["duration_sec"] for r in c_runs]
            m, s = mean_std(times)
            chroma_times.append(m)
            chroma_errs.append(s)
        else:
            chroma_times.append(0)
            chroma_errs.append(0)
        
        p_runs = data["pgvector"]
        if p_runs:
            times = [r["duration_sec"] for r in p_runs]
            m, s = mean_std(times)
            pg_times.append(m)
            pg_errs.append(s)
        else:
            pg_times.append(0)
            pg_errs.append(0)
    
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    ax1 = axes[0]
    ax1.bar(x - width/2, chroma_times, width, yerr=chroma_errs, label='Chroma', color='#2ecc71', capsize=3)
    ax1.bar(x + width/2, pg_times, width, yerr=pg_errs, label='pgvector', color='#3498db', capsize=3)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Total Ingestion Time (seconds)')
    ax1.set_title(f'Batch Size Impact on Ingestion Time ({test_dataset} vectors)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    chroma_vps = []
    pg_vps = []
    
    for bs in batch_sizes:
        data = aggregated[test_dataset][bs]
        
        c_runs = data["chroma"]
        if c_runs:
            chroma_vps.append(np.mean([r["vectors_per_sec"] for r in c_runs]))
        else:
            chroma_vps.append(0)
        
        p_runs = data["pgvector"]
        if p_runs:
            pg_vps.append(np.mean([r["vectors_per_sec"] for r in p_runs]))
        else:
            pg_vps.append(0)
    
    ax2 = axes[1]
    ax2.plot(batch_sizes, chroma_vps, 'o-', label='Chroma', color='#2ecc71', linewidth=2, markersize=8)
    ax2.plot(batch_sizes, pg_vps, 's-', label='pgvector', color='#3498db', linewidth=2, markersize=8)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (vectors/sec)')
    ax2.set_title(f'Batch Size Impact on Throughput ({test_dataset} vectors)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "batch_size_impact.png"), dpi=150)
    plt.close()
    print("Saved: batch_size_impact.png")


def plot_dimension_impact(aggregated, batch_size=1000):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    dim_datasets = {}
    for d in aggregated.keys():
        if "768" in d:
            dim_datasets[768] = d
        elif "1536" in d:
            dim_datasets[1536] = d
        elif d == "100k":
            dim_datasets[128] = d
    
    if len(dim_datasets) < 2:
        print("Not enough dimension variants for comparison")
        return
    
    dims = sorted(dim_datasets.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    chroma_times = []
    pg_times = []
    vectors_counts = []
    
    for dim in dims:
        dataset = dim_datasets[dim]
        data = aggregated[dataset][batch_size]
        
        c_runs = data["chroma"]
        if c_runs:
            chroma_times.append(np.mean([r["duration_sec"] for r in c_runs]))
            vectors_counts.append(c_runs[0]["vectors"])
        else:
            chroma_times.append(0)
            vectors_counts.append(0)
        
        p_runs = data["pgvector"]
        if p_runs:
            pg_times.append(np.mean([r["duration_sec"] for r in p_runs]))
        else:
            pg_times.append(0)
    
    x = np.arange(len(dims))
    width = 0.35
    
    ax1 = axes[0]
    ax1.bar(x - width/2, chroma_times, width, label='Chroma', color='#2ecc71')
    ax1.bar(x + width/2, pg_times, width, label='pgvector', color='#3498db')
    ax1.set_xlabel('Vector Dimensions')
    ax1.set_ylabel('Ingestion Time (seconds)')
    ax1.set_title('Dimension Impact on Ingestion Time')
    ax1.set_xticks(x)
    labels = [f"{dim}d\n({dim_datasets[dim]})" for dim in dims]
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    chroma_storage = []
    pg_storage = []
    raw_storage = []
    
    for dim in dims:
        dataset = dim_datasets[dim]
        data = aggregated[dataset][batch_size]
        
        c_runs = data["chroma"]
        if c_runs:
            storage = [r.get("storage_bytes", 0) or 0 for r in c_runs]
            chroma_storage.append(np.mean(storage) / (1024*1024) if any(storage) else 0)
            n = c_runs[0]["vectors"]
            raw_storage.append(n * dim * 4 / (1024*1024))
        else:
            chroma_storage.append(0)
            raw_storage.append(0)
        
        p_runs = data["pgvector"]
        if p_runs:
            pg_storage.append(np.mean([r["storage_bytes"] for r in p_runs]) / (1024*1024))
        else:
            pg_storage.append(0)
    
    ax2 = axes[1]
    ax2.bar(x - width, raw_storage, width, label='Raw data', color='#95a5a6')
    ax2.bar(x, chroma_storage, width, label='Chroma', color='#2ecc71')
    ax2.bar(x + width, pg_storage, width, label='pgvector', color='#3498db')
    ax2.set_xlabel('Vector Dimensions')
    ax2.set_ylabel('Storage (MB)')
    ax2.set_title('Dimension Impact on Storage')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dimension_impact.png"), dpi=150)
    plt.close()
    print("Saved: dimension_impact.png")


def generate_latex_table(aggregated, batch_size=1000):
    print("\n" + "="*80)
    print("LATEX TABLE (for paper)")
    print("="*80)
    
    print(r"""
\begin{table}[H]
\centering
\caption{Ingestion Performance Comparison (batch\_size=1000)}
\label{tab:ingestion-results}
\begin{tabular}{lrrr|rrr|r}
\toprule
\textbf{Dataset} & \textbf{Vectors} & \textbf{Dim} & \textbf{Raw MB} & 
\multicolumn{3}{c|}{\textbf{Chroma}} & \multicolumn{1}{c}{\textbf{pgvector}} \\
& & & & Time(s) & V/s & MB & Time(s) \\
\midrule""")
    
    for dataset in sorted(aggregated.keys()):
        data = aggregated[dataset].get(batch_size, {})
        
        c_runs = data.get("chroma", [])
        p_runs = data.get("pgvector", [])
        
        if not c_runs and not p_runs:
            continue
        
        if c_runs:
            n = c_runs[0]["vectors"]
            dim = c_runs[0]["dimensions"]
            raw_mb = n * dim * 4 / (1024*1024)
            c_time = np.mean([r["duration_sec"] for r in c_runs])
            c_vps = np.mean([r["vectors_per_sec"] for r in c_runs])
            c_storage = [r.get("storage_bytes", 0) or 0 for r in c_runs]
            c_mb = np.mean(c_storage) / (1024*1024) if any(c_storage) else 0
        else:
            n = dim = raw_mb = c_time = c_vps = c_mb = 0
        
        if p_runs:
            p_time = np.mean([r["duration_sec"] for r in p_runs])
            p_vps = np.mean([r["vectors_per_sec"] for r in p_runs])
            p_mb = np.mean([r["storage_bytes"] for r in p_runs]) / (1024*1024)
        else:
            p_time = p_vps = p_mb = 0
        
        print(f"{dataset} & {n:,} & {dim} & {raw_mb:.1f} & {c_time:.2f} & {c_vps:,.0f} & {c_mb:.1f} & {p_time:.2f} \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


def main():
    results = load_latest_results()
    aggregated = aggregate_results(results)
    
    for batch_size in [500, 1000, 5000]:
        if any(batch_size in aggregated[d] for d in aggregated):
            plot_ingestion_time_comparison(aggregated, batch_size)
    
    plot_storage_comparison(aggregated)
    plot_batch_size_impact(aggregated)
    plot_dimension_impact(aggregated)
    generate_latex_table(aggregated)
    
    print(f"\nAll plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
