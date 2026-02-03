#!/usr/bin/env python3
import json
import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_FILE = "results/raw/all_ingestion_results.json"
PLOTS_DIR = "results/plots"


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def plot_throughput_comparison(data):
    runs = data["runs"]
    datasets = [r["dataset"] for r in runs]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    x = np.arange(len(datasets))
    width = 0.35
    
    chroma_vps = []
    pg_vps = []
    for run in runs:
        c_1000 = next((r for r in run["chroma"] if r["batch_size"] == 1000), None)
        p_1000 = next((r for r in run["pgvector"] if r["batch_size"] == 1000), None)
        chroma_vps.append(c_1000["vectors_per_sec"] if c_1000 and "vectors_per_sec" in c_1000 else 0)
        pg_vps.append(p_1000["vectors_per_sec"] if p_1000 and "vectors_per_sec" in p_1000 else 0)
    
    bars1 = ax1.bar(x - width/2, chroma_vps, width, label='Chroma (HNSW)', color='#2ecc71')
    bars2 = ax1.bar(x + width/2, pg_vps, width, label='pgvector (IVFFlat)', color='#3498db')
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Vectors per Second')
    ax1.set_title('Ingestion Throughput (batch_size=1000)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars1, chroma_vps):
        if val > 0:
            ax1.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, pg_vps):
        if val > 0:
            ax1.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=8)
    
    ax2 = axes[1]
    largest = runs[-1] 
    batch_sizes = [r["batch_size"] for r in largest["chroma"]]
    
    chroma_by_batch = [r["vectors_per_sec"] for r in largest["chroma"] if "vectors_per_sec" in r]
    pg_by_batch = [r["vectors_per_sec"] for r in largest["pgvector"] if "vectors_per_sec" in r]
    
    ax2.plot(batch_sizes[:len(chroma_by_batch)], chroma_by_batch, 'o-', label='Chroma', color='#2ecc71', linewidth=2, markersize=8)
    ax2.plot(batch_sizes[:len(pg_by_batch)], pg_by_batch, 's-', label='pgvector', color='#3498db', linewidth=2, markersize=8)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Vectors per Second')
    ax2.set_title(f'Batch Size Impact ({largest["dataset"]}, {largest["vectors"]:,} vectors)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "throughput_comparison.png"), dpi=150)
    plt.close()
    print("Saved: throughput_comparison.png")


def plot_storage_comparison(data):
    runs = data["runs"]
    datasets = [r["dataset"] for r in runs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(datasets))
    width = 0.25
    
    raw_sizes = []
    chroma_sizes = []
    pg_data_sizes = []
    pg_index_sizes = []
    
    for run in runs:
        raw_mb = run["vectors"] * run["dimensions"] * 4 / (1024*1024)
        raw_sizes.append(raw_mb)
        
        c_1000 = next((r for r in run["chroma"] if r["batch_size"] == 1000), None)
        chroma_sizes.append(c_1000["storage_mb"] if c_1000 and c_1000.get("storage_mb") else 0)
        
        p_1000 = next((r for r in run["pgvector"] if r["batch_size"] == 1000), None)
        if p_1000:
            pg_data_sizes.append(p_1000.get("storage_before_index_mb", 0))
            pg_index_sizes.append(p_1000.get("storage_mb", 0) - p_1000.get("storage_before_index_mb", 0))
        else:
            pg_data_sizes.append(0)
            pg_index_sizes.append(0)
    
    ax.bar(x - width, raw_sizes, width, label='Raw data (float32)', color='#95a5a6')
    ax.bar(x, chroma_sizes, width, label='Chroma (HNSW)', color='#2ecc71')
    ax.bar(x + width, pg_data_sizes, width, label='pgvector (table)', color='#3498db')
    ax.bar(x + width, pg_index_sizes, width, bottom=pg_data_sizes, label='pgvector (index)', color='#e74c3c')
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Storage (MB)')
    ax.set_title('Storage Comparison: Raw Data vs Database Storage')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['dataset']}\n({r['vectors']//1000}k×{r['dimensions']}d)" for r in runs])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "storage_comparison.png"), dpi=150)
    plt.close()
    print("Saved: storage_comparison.png")


def plot_memory_usage(data):
    runs = data["runs"]
    datasets = [r["dataset"] for r in runs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    chroma_peak = []
    pg_peak = []
    
    for run in runs:
        c_peaks = [r["docker_stats"]["peak_mem_percent"] for r in run["chroma"] 
                   if "docker_stats" in r and "peak_mem_percent" in r["docker_stats"]]
        p_peaks = [r["docker_stats"]["peak_mem_percent"] for r in run["pgvector"]
                   if "docker_stats" in r and "peak_mem_percent" in r["docker_stats"]]
        
        chroma_peak.append(max(c_peaks) if c_peaks else 0)
        pg_peak.append(max(p_peaks) if p_peaks else 0)
    
    chroma_gb = [p * 4 / 100 for p in chroma_peak]
    pg_gb = [p * 4 / 100 for p in pg_peak]
    
    bars1 = ax.bar(x - width/2, chroma_gb, width, label='Chroma', color='#2ecc71')
    bars2 = ax.bar(x + width/2, pg_gb, width, label='pgvector', color='#3498db')
    
    ax.axhline(y=4, color='red', linestyle='--', alpha=0.5, label='Memory limit (4GB)')
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Peak Memory Usage (GB)')
    ax.set_title('Peak Memory Usage During Ingestion')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['dataset']}\n({r['vectors']//1000}k)" for r in runs])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "memory_usage.png"), dpi=150)
    plt.close()
    print("Saved: memory_usage.png")


def plot_time_breakdown(data):
    runs = data["runs"]
    datasets = [r["dataset"] for r in runs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    chroma_total = []
    pg_insert = []
    pg_index = []
    
    for run in runs:
        c_1000 = next((r for r in run["chroma"] if r["batch_size"] == 1000), None)
        p_1000 = next((r for r in run["pgvector"] if r["batch_size"] == 1000), None)
        
        chroma_total.append(c_1000["duration_sec"] if c_1000 else 0)
        pg_insert.append(p_1000["duration_ingest_sec"] if p_1000 else 0)
        pg_index.append(p_1000["duration_index_sec"] if p_1000 and p_1000.get("duration_index_sec") else 0)
    
    ax.bar(x - width/2, chroma_total, width, label='Chroma (HNSW, built-in)', color='#2ecc71')
    ax.bar(x + width/2, pg_insert, width, label='pgvector (insert)', color='#3498db')
    ax.bar(x + width/2, pg_index, width, bottom=pg_insert, label='pgvector (IVFFlat index)', color='#e74c3c')
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Ingestion Time Breakdown (batch_size=1000)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['dataset']}\n({r['vectors']//1000}k)" for r in runs])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "time_breakdown.png"), dpi=150)
    plt.close()
    print("Saved: time_breakdown.png")


def plot_dimensionality_impact(data):
    """Plot comparing throughput and storage across different dimensions for same vector count."""
    runs = data["runs"]
    
    by_count = {}
    for run in runs:
        count = run["vectors"]
        if count not in by_count:
            by_count[count] = []
        by_count[count].append(run)
    
    multi_dim_counts = {k: v for k, v in by_count.items() if len(v) > 1}
    
    if not multi_dim_counts:
        print("Skipping dimensionality plot: need same vector count with different dimensions")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    
    ax1 = axes[0, 0]
    for idx, (count, runs_list) in enumerate(sorted(multi_dim_counts.items())):
        runs_sorted = sorted(runs_list, key=lambda r: r["dimensions"])
        dims = [r["dimensions"] for r in runs_sorted]
        vps = []
        for r in runs_sorted:
            c = next((x for x in r["chroma"] if x["batch_size"] == 1000), {})
            vps.append(c.get("vectors_per_sec", 0))
        ax1.plot(dims, vps, 'o-', label=f'{count//1000}k vectors', color=colors[idx % len(colors)], linewidth=2, markersize=8)
    
    ax1.set_xlabel('Vector Dimensions')
    ax1.set_ylabel('Vectors per Second')
    ax1.set_title('Chroma: Throughput vs Dimensionality')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    ax2 = axes[0, 1]
    for idx, (count, runs_list) in enumerate(sorted(multi_dim_counts.items())):
        runs_sorted = sorted(runs_list, key=lambda r: r["dimensions"])
        dims = [r["dimensions"] for r in runs_sorted]
        vps = []
        for r in runs_sorted:
            p = next((x for x in r["pgvector"] if x["batch_size"] == 1000), {})
            vps.append(p.get("vectors_per_sec", 0))
        ax2.plot(dims, vps, 's-', label=f'{count//1000}k vectors', color=colors[idx % len(colors)], linewidth=2, markersize=8)
    
    ax2.set_xlabel('Vector Dimensions')
    ax2.set_ylabel('Vectors per Second')
    ax2.set_title('pgvector: Throughput vs Dimensionality')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    ax3 = axes[1, 0]
    width = 0.35
    
    largest_count = max(multi_dim_counts.keys())
    runs_sorted = sorted(multi_dim_counts[largest_count], key=lambda r: r["dimensions"])
    dims = [r["dimensions"] for r in runs_sorted]
    x = np.arange(len(dims))
    
    chroma_storage = []
    pg_storage = []
    raw_storage = []
    
    for r in runs_sorted:
        c = next((x for x in r["chroma"] if x["batch_size"] == 1000), {})
        p = next((x for x in r["pgvector"] if x["batch_size"] == 1000), {})
        chroma_storage.append(c.get("storage_mb", 0))
        pg_storage.append(p.get("storage_mb", 0))
        raw_storage.append(r["vectors"] * r["dimensions"] * 4 / (1024*1024))
    
    ax3.bar(x - width, raw_storage, width, label='Raw data', color='#95a5a6')
    ax3.bar(x, chroma_storage, width, label='Chroma', color='#2ecc71')
    ax3.bar(x + width, pg_storage, width, label='pgvector', color='#3498db')
    ax3.set_xlabel('Vector Dimensions')
    ax3.set_ylabel('Storage (MB)')
    ax3.set_title(f'Storage vs Dimensionality ({largest_count//1000}k vectors)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(dims)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Chroma vs pgvector ratio by dimension
    ax4 = axes[1, 1]
    for idx, (count, runs_list) in enumerate(sorted(multi_dim_counts.items())):
        runs_sorted = sorted(runs_list, key=lambda r: r["dimensions"])
        dims = [r["dimensions"] for r in runs_sorted]
        ratios = []
        for r in runs_sorted:
            c = next((x for x in r["chroma"] if x["batch_size"] == 1000), {})
            p = next((x for x in r["pgvector"] if x["batch_size"] == 1000), {})
            c_vps = c.get("vectors_per_sec", 1)
            p_vps = p.get("vectors_per_sec", 1) or 1
            ratios.append(c_vps / p_vps)
        ax4.plot(dims, ratios, 'o-', label=f'{count//1000}k vectors', color=colors[idx % len(colors)], linewidth=2, markersize=8)
    
    ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Vector Dimensions')
    ax4.set_ylabel('Chroma / pgvector Throughput Ratio')
    ax4.set_title('Relative Performance (>1 = Chroma faster)')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dimensionality_impact.png"), dpi=150)
    plt.close()
    print("Saved: dimensionality_impact.png")


def plot_resource_usage(data):
    """Plot CPU and memory resource usage comparison."""
    runs = data["runs"]
    datasets = [r["dataset"] for r in runs]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    chroma_peak_mem = []
    pg_peak_mem = []
    chroma_avg_cpu = []
    pg_avg_cpu = []
    
    for run in runs:
        c = next((r for r in run["chroma"] if r["batch_size"] == 1000), {})
        p = next((r for r in run["pgvector"] if r["batch_size"] == 1000), {})
        
        c_stats = c.get("docker_stats", {})
        p_stats = p.get("docker_stats", {})
        
        chroma_peak_mem.append(c_stats.get("peak_mem_percent", 0) * 4 / 100)
        pg_peak_mem.append(p_stats.get("peak_mem_percent", 0) * 4 / 100)
        chroma_avg_cpu.append(c_stats.get("avg_cpu_percent", 0))
        pg_avg_cpu.append(p_stats.get("avg_cpu_percent", 0))
    
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width/2, chroma_peak_mem, width, label='Chroma', color='#2ecc71')
    bars2 = ax1.bar(x + width/2, pg_peak_mem, width, label='pgvector', color='#3498db')
    ax1.axhline(y=4, color='red', linestyle='--', alpha=0.5, label='Memory limit (4GB)')
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Peak Memory (GB)')
    ax1.set_title('Peak Memory Usage During Ingestion (batch_size=1000)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, chroma_avg_cpu, width, label='Chroma', color='#2ecc71')
    ax2.bar(x + width/2, pg_avg_cpu, width, label='pgvector', color='#3498db')
    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='1 CPU core')
    ax2.axhline(y=600, color='red', linestyle='--', alpha=0.5, label='CPU limit (6 cores)')
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Average CPU Usage (%)')
    ax2.set_title('Average CPU Usage During Ingestion (batch_size=1000)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    ax3 = axes[1, 0]
    
    runs_128d = [(r["dataset"], r["vectors"], r) for r in runs if r["dimensions"] == 128]
    runs_128d.sort(key=lambda x: x[1])
    
    if runs_128d:
        labels = [r[0] for r in runs_128d]
        vectors = [r[1] for r in runs_128d]
        
        chroma_mem_128 = []
        pg_mem_128 = []
        
        for _, _, run in runs_128d:
            c = next((r for r in run["chroma"] if r["batch_size"] == 1000), {})
            p = next((r for r in run["pgvector"] if r["batch_size"] == 1000), {})
            c_stats = c.get("docker_stats", {})
            p_stats = p.get("docker_stats", {})
            chroma_mem_128.append(c_stats.get("peak_mem_percent", 0) * 4 / 100)
            pg_mem_128.append(p_stats.get("peak_mem_percent", 0) * 4 / 100)
        
        ax3.plot(vectors, chroma_mem_128, 'o-', label='Chroma', color='#2ecc71', linewidth=2, markersize=8)
        ax3.plot(vectors, pg_mem_128, 's-', label='pgvector', color='#3498db', linewidth=2, markersize=8)
        ax3.axhline(y=4, color='red', linestyle='--', alpha=0.5, label='Memory limit')
        ax3.set_xlabel('Number of Vectors')
        ax3.set_ylabel('Peak Memory (GB)')
        ax3.set_title('Memory Scaling with Dataset Size (128d)')
        ax3.legend()
        ax3.grid(alpha=0.3)
        ax3.set_xscale('log')
    
    ax4 = axes[1, 1]
    
    chroma_efficiency = []
    pg_efficiency = []
    
    for run in runs:
        c = next((r for r in run["chroma"] if r["batch_size"] == 1000), {})
        p = next((r for r in run["pgvector"] if r["batch_size"] == 1000), {})
        
        c_vps = c.get("vectors_per_sec", 0)
        p_vps = p.get("vectors_per_sec", 0)
        c_cpu = c.get("docker_stats", {}).get("avg_cpu_percent", 1) or 1
        p_cpu = p.get("docker_stats", {}).get("avg_cpu_percent", 1) or 1
        
        chroma_efficiency.append(c_vps / c_cpu * 100)  # vectors per 1 CPU core
        pg_efficiency.append(p_vps / p_cpu * 100)
    
    ax4.bar(x - width/2, chroma_efficiency, width, label='Chroma', color='#2ecc71')
    ax4.bar(x + width/2, pg_efficiency, width, label='pgvector', color='#3498db')
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Vectors/sec per CPU core')
    ax4.set_title('CPU Efficiency (Throughput per CPU core)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "resource_usage.png"), dpi=150)
    plt.close()
    print("Saved: resource_usage.png")


def print_summary_table(data):
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    print(f"\n{'Dataset':<12} {'Vectors':>10} {'Dim':>5} | {'Chroma v/s':>12} {'Chroma MB':>10} | {'PG v/s':>10} {'PG MB':>8}")
    print("-" * 80)
    
    for run in data["runs"]:
        c = next((r for r in run["chroma"] if r["batch_size"] == 1000), {})
        p = next((r for r in run["pgvector"] if r["batch_size"] == 1000), {})
        
        print(f"{run['dataset']:<12} {run['vectors']:>10,} {run['dimensions']:>5} | "
              f"{c.get('vectors_per_sec', 0):>12,.0f} {c.get('storage_mb', 0):>10.1f} | "
              f"{p.get('vectors_per_sec', 0):>10,.0f} {p.get('storage_mb', 0):>8.1f}")


def filter_data(data, dimensions=None, sizes=None, datasets=None, min_vectors=None, max_vectors=None):
    runs = data["runs"]
    
    filtered = [r for r in runs if (
        (dimensions is None or r["dimensions"] in dimensions) and
        (sizes is None or r["vectors"] in sizes) and
        (datasets is None or r["dataset"] in datasets) and
        (min_vectors is None or r["vectors"] >= min_vectors) and
        (max_vectors is None or r["vectors"] <= max_vectors)
    )]
    
    return {"runs": filtered}


def get_available_filters(data):
    """
    Get available filter values from the data.
    Useful for discovering what dimensions/sizes/datasets are available.
    """
    runs = data["runs"]
    return {
        'dimensions': sorted(set(r["dimensions"] for r in runs)),
        'sizes': sorted(set(r["vectors"] for r in runs)),
        'size_labels': sorted(set(r["dataset"].split('_')[0] for r in runs), 
                              key=lambda x: int(x.replace('k', '000').replace('m', '000000'))),
        'datasets': [r["dataset"] for r in runs],
        'batch_sizes': sorted(set(b["batch_size"] for r in runs for b in r["chroma"])) if runs else [],
    }


def print_available_filters(data):
    """Print available filter options."""
    f = get_available_filters(data)
    print("\nAvailable filters:")
    print(f"  dimensions: {f['dimensions']}")
    print(f"  sizes (vectors): {f['sizes']}")
    print(f"  datasets: {f['datasets']}")
    print(f"  batch_sizes: {f['batch_sizes']}")



def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    data = load_results()
    print(f"Loaded {len(data['runs'])} datasets from {RESULTS_FILE}")
    
    print_available_filters(data)
    
    plot_throughput_comparison(data)
    plot_storage_comparison(data)
    plot_memory_usage(data)
    plot_time_breakdown(data)
    plot_dimensionality_impact(data)
    plot_resource_usage(data)
    print_summary_table(data)
    
    plot_throughput_comparison(filter_data(data, dimensions=[128]))
    plot_time_breakdown(filter_data(data, dimensions=[128]))
    plot_storage_comparison(filter_data(data, dimensions=[128]))


    
    print(f"\nAll plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
