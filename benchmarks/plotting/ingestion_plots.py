#!/usr/bin/env python3
import json
import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_FILE = "results/raw/all_ingestion_results.json"
PLOTS_DIR = "results/plots/ingestion"


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def plot_throughput_comparison(data, dimensions=None, pg_index_type=None, chroma_builtin=None, pg_lists=None):
    # Apply filtering
    filtered = filter_data(data, dimensions=dimensions, pg_index_type=pg_index_type, chroma_builtin=chroma_builtin, pg_lists=pg_lists)
    runs = filtered["runs"]
    
    if not runs:
        print("Skipping throughput comparison: no matching data")
        return
    
    datasets = [r["dataset"] for r in runs]
    
    # Build title suffix based on filters
    title_parts = []
    if dimensions:
        title_parts.append(f"{dimensions[0]}d")
    if pg_index_type:
        title_parts.append(f"pg:{pg_index_type}")
    filter_str = f" ({', '.join(title_parts)})" if title_parts else ""
    
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
    
    # Get index labels from first run
    pg_label = 'pgvector'
    chroma_label = 'Chroma'
    if runs:
        p_run = next((r for r in runs[0]["pgvector"] if r["batch_size"] == 1000), None)
        if p_run:
            idx_type = p_run.get("index_type", "")
            pg_label = f'pgvector ({idx_type.upper()})' if idx_type else 'pgvector'
    
    bars1 = ax1.bar(x - width/2, chroma_vps, width, label=f'{chroma_label} (HNSW)', color='#2ecc71')
    bars2 = ax1.bar(x + width/2, pg_vps, width, label=pg_label, color='#3498db')
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Vectors per Second')
    ax1.set_title(f'Ingestion Throughput{filter_str} (batch_size=1000)')
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


def plot_storage_comparison(data, dimensions=None, pg_index_type=None, chroma_builtin=None, pg_lists=None):
    # Apply filtering
    filtered = filter_data(data, dimensions=dimensions, pg_index_type=pg_index_type, chroma_builtin=chroma_builtin, pg_lists=pg_lists)
    runs = filtered["runs"]
    
    if not runs:
        print("Skipping storage comparison: no matching data")
        return
    
    datasets = [r["dataset"] for r in runs]
    
    # Build title suffix based on filters
    title_parts = []
    if dimensions:
        title_parts.append(f"{dimensions[0]}d")
    if pg_index_type:
        title_parts.append(f"pg:{pg_index_type}")
    filter_str = f" ({', '.join(title_parts)})" if title_parts else ""
    
    # Get index label from first run
    pg_index_label = 'index'
    if runs:
        p_run = next((r for r in runs[0]["pgvector"] if r["batch_size"] == 1000), None)
        if p_run:
            pg_index_label = p_run.get("index_type", "index").upper()
    
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
    ax.bar(x + width, pg_index_sizes, width, bottom=pg_data_sizes, label=f'pgvector ({pg_index_label})', color='#e74c3c')
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Storage (MB)')
    ax.set_title(f'Storage Comparison{filter_str}: Raw Data vs Database Storage')
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


def plot_time_breakdown(data, dimensions=None, pg_index_type=None, chroma_builtin=None, pg_lists=None):
    # Apply filtering
    filtered = filter_data(data, dimensions=dimensions, pg_index_type=pg_index_type, chroma_builtin=chroma_builtin, pg_lists=pg_lists)
    runs = filtered["runs"]
    
    if not runs:
        print("Skipping time breakdown: no matching data")
        return
    
    datasets = [r["dataset"] for r in runs]
    
    # Build title suffix based on filters
    title_parts = []
    if dimensions:
        title_parts.append(f"{dimensions[0]}d")
    if pg_index_type:
        title_parts.append(f"pg:{pg_index_type}")
    filter_str = f" ({', '.join(title_parts)})" if title_parts else ""
    
    # Get index label from first run
    pg_index_label = 'index'
    if runs:
        p_run = next((r for r in runs[0]["pgvector"] if r["batch_size"] == 1000), None)
        if p_run:
            pg_index_label = p_run.get("index_type", "index").upper()
    
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
    ax.bar(x + width/2, pg_index, width, bottom=pg_insert, label=f'pgvector ({pg_index_label})', color='#e74c3c')
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'Ingestion Time Breakdown{filter_str} (batch_size=1000)')
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


def plot_resource_usage(data, dimensions=None, pg_index_type=None, chroma_builtin=None, pg_lists=None):
    """Plot CPU and memory resource usage comparison."""
    # Apply filtering
    filtered = filter_data(data, dimensions=dimensions, pg_index_type=pg_index_type, chroma_builtin=chroma_builtin, pg_lists=pg_lists)
    runs = filtered["runs"]
    
    if not runs:
        print("Skipping resource usage: no matching data")
        return
    
    datasets = [r["dataset"] for r in runs]
    
    # Build title suffix based on filters
    title_parts = []
    if dimensions:
        title_parts.append(f"{dimensions[0]}d")
    if pg_index_type:
        title_parts.append(f"pg:{pg_index_type}")
    filter_str = f" ({', '.join(title_parts)})" if title_parts else ""
    
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
    ax1.set_title(f'Peak Memory Usage During Ingestion{filter_str} (batch_size=1000)')
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
    ax2.set_title(f'Average CPU Usage During Ingestion{filter_str} (batch_size=1000)')
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


def filter_data(data, 
                dimensions=None, 
                sizes=None, 
                datasets=None, 
                min_vectors=None, 
                max_vectors=None,
                pg_index_type=None,      # 'ivfflat', 'hnsw', 'none'
                chroma_index_type=None,  # 'hnsw' (only option)
                pg_lists=None,           # IVFFlat lists param (e.g., 100, 200)
                pg_hnsw_m=None,          # HNSW m param (e.g., 16, 32)
                pg_hnsw_ef=None,         # HNSW ef_construction (e.g., 64, 128)
                chroma_m=None,           # Chroma HNSW m param
                chroma_ef=None,          # Chroma HNSW ef_construction
                batch_sizes=None,        # List of batch sizes to keep
                chroma_builtin=None,     # True = built-in index, False = custom params
                custom_filter=None):     # lambda run: bool
    """
    Filter ingestion results flexibly.
    
    Args:
        data: Full results dict with 'runs' key
        dimensions: List of dimensions to include (e.g., [128, 768])
        sizes: List of vector counts to include (e.g., [50000, 100000])
        datasets: List of dataset names to include (e.g., ['50k', '100k'])
        min_vectors: Minimum vector count
        max_vectors: Maximum vector count
        pg_index_type: pgvector index type ('ivfflat', 'hnsw', 'none')
        chroma_index_type: Chroma index type ('hnsw')
        pg_lists: IVFFlat lists parameter value(s)
        pg_hnsw_m: pgvector HNSW m parameter value(s)
        pg_hnsw_ef: pgvector HNSW ef_construction value(s)
        chroma_m: Chroma HNSW m parameter value(s)
        chroma_ef: Chroma HNSW ef_construction value(s)
        batch_sizes: List of batch sizes to keep in results
        chroma_builtin: True for built-in index runs, False for custom
        custom_filter: Custom lambda function for additional filtering
    
    Returns:
        Filtered data dict with 'runs' key
    """
    runs = data["runs"]
    
    def match_index_params(result_list, index_type=None, lists=None, hnsw_m=None, hnsw_ef=None, builtin=None):
        """Check if any result in list matches the index criteria."""
        for r in result_list:
            params = r.get("index_params", {})
            
            # Check built-in
            if builtin is not None:
                is_builtin = params.get("built_in", False)
                if builtin != is_builtin:
                    continue
            
            # Check index type
            if index_type is not None:
                r_type = r.get("index_type", "")
                if isinstance(index_type, list):
                    if r_type not in index_type:
                        continue
                elif r_type != index_type:
                    continue
            
            # Check IVFFlat lists
            if lists is not None:
                r_lists = params.get("lists")
                if isinstance(lists, list):
                    if r_lists not in lists:
                        continue
                elif r_lists != lists:
                    continue
            
            # Check HNSW m
            if hnsw_m is not None:
                r_m = params.get("m")
                if isinstance(hnsw_m, list):
                    if r_m not in hnsw_m:
                        continue
                elif r_m != hnsw_m:
                    continue
            
            # Check HNSW ef_construction
            if hnsw_ef is not None:
                r_ef = params.get("ef_construction")
                if isinstance(hnsw_ef, list):
                    if r_ef not in hnsw_ef:
                        continue
                elif r_ef != hnsw_ef:
                    continue
            
            return True
        return False
    
    def filter_batch_sizes(result_list, allowed_batches):
        """Filter results to only include specified batch sizes."""
        if allowed_batches is None:
            return result_list
        return [r for r in result_list if r.get("batch_size") in allowed_batches]
    
    filtered = []
    for run in runs:
        # Basic filters
        if dimensions is not None and run["dimensions"] not in dimensions:
            continue
        if sizes is not None and run["vectors"] not in sizes:
            continue
        if datasets is not None and run["dataset"] not in datasets:
            continue
        if min_vectors is not None and run["vectors"] < min_vectors:
            continue
        if max_vectors is not None and run["vectors"] > max_vectors:
            continue
        
        # pgvector index filters
        if any(x is not None for x in [pg_index_type, pg_lists, pg_hnsw_m, pg_hnsw_ef]):
            if not match_index_params(run.get("pgvector", []), 
                                      index_type=pg_index_type,
                                      lists=pg_lists,
                                      hnsw_m=pg_hnsw_m,
                                      hnsw_ef=pg_hnsw_ef):
                continue
        
        # Chroma index filters
        if any(x is not None for x in [chroma_index_type, chroma_m, chroma_ef, chroma_builtin]):
            if not match_index_params(run.get("chroma", []),
                                      index_type=chroma_index_type,
                                      hnsw_m=chroma_m,
                                      hnsw_ef=chroma_ef,
                                      builtin=chroma_builtin):
                continue
        
        # Custom filter
        if custom_filter is not None and not custom_filter(run):
            continue
        
        # Create filtered run with batch size filtering
        filtered_run = run.copy()
        filtered_run["chroma"] = filter_batch_sizes(run.get("chroma", []), batch_sizes)
        filtered_run["pgvector"] = filter_batch_sizes(run.get("pgvector", []), batch_sizes)
        
        # Only include if there's still data after batch filtering
        if filtered_run["chroma"] or filtered_run["pgvector"]:
            filtered.append(filtered_run)
    
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


def plot_index_storage_breakdown(data, batch_size=1000):
    """
    Plot storage breakdown by index type for 100k datasets.
    Shows table size, index size, and index overhead.
    """
    runs = data["runs"]
    
    runs_100k = [r for r in runs if r["vectors"] == 100000]
    
    if not runs_100k:
        print("Skipping index storage plot: no 100k datasets found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Sort by dimensions
    runs_100k = sorted(runs_100k, key=lambda r: r["dimensions"])
    datasets = [r["dataset"] for r in runs_100k]
    dims = [r["dimensions"] for r in runs_100k]
    
    raw_sizes = []
    chroma_sizes = []
    pg_table_sizes = []
    pg_index_sizes = []
    pg_total_sizes = []
    
    for run in runs_100k:
        # Raw data size
        raw_mb = run["vectors"] * run["dimensions"] * 4 / (1024*1024)
        raw_sizes.append(raw_mb)
        
        # Chroma (batch_size=1000)
        c = next((r for r in run["chroma"] if r["batch_size"] == batch_size), None)
        chroma_sizes.append(c["storage_mb"] if c else 0)
        
        # pgvector (batch_size=1000)
        p = next((r for r in run["pgvector"] if r["batch_size"] == batch_size), None)
        if p:
            table_size = p.get("storage_before_index_mb", 0)
            total_size = p.get("storage_mb", 0)
            index_size = total_size - table_size
            pg_table_sizes.append(table_size)
            pg_index_sizes.append(index_size)
            pg_total_sizes.append(total_size)
        else:
            pg_table_sizes.append(0)
            pg_index_sizes.append(0)
            pg_total_sizes.append(0)
    
    # Plot 1: Storage breakdown by component
    ax1 = axes[0]
    x = np.arange(len(datasets))
    width = 0.2
    
    ax1.bar(x - 1.5*width, raw_sizes, width, label='Raw vectors', color='#95a5a6')
    ax1.bar(x - 0.5*width, chroma_sizes, width, label='Chroma (HNSW)', color='#2ecc71')
    ax1.bar(x + 0.5*width, pg_table_sizes, width, label='pgvector table', color='#3498db')
    ax1.bar(x + 0.5*width, pg_index_sizes, width, bottom=pg_table_sizes, label='pgvector IVFFlat index', color='#e74c3c')
    
    ax1.set_xlabel('Dataset (dimensions)')
    ax1.set_ylabel('Storage (MB)')
    ax1.set_title(f'Storage Breakdown (100k vectors, batch={batch_size})')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{d}d" for d in dims])
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Storage overhead ratio (vs raw data)
    ax2 = axes[1]
    
    chroma_overhead = [c/r if r > 0 else 0 for c, r in zip(chroma_sizes, raw_sizes)]
    pg_overhead = [p/r if r > 0 else 0 for p, r in zip(pg_total_sizes, raw_sizes)]
    
    ax2.bar(x - width/2, chroma_overhead, width, label='Chroma (HNSW)', color='#2ecc71')
    ax2.bar(x + width/2, pg_overhead, width, label='pgvector (IVFFlat)', color='#3498db')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Raw data size')
    
    ax2.set_xlabel('Dataset (dimensions)')
    ax2.set_ylabel('Storage / Raw Data Ratio')
    ax2.set_title('Storage Overhead vs Raw Data')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{d}d" for d in dims])
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (c, p) in enumerate(zip(chroma_overhead, pg_overhead)):
        ax2.annotate(f'{c:.2f}x', xy=(i - width/2, c), ha='center', va='bottom', fontsize=8)
        ax2.annotate(f'{p:.2f}x', xy=(i + width/2, p), ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Index overhead as percentage of total
    ax3 = axes[2]
    
    pg_index_pct = [idx/total*100 if total > 0 else 0 for idx, total in zip(pg_index_sizes, pg_total_sizes)]
    pg_table_pct = [100 - pct for pct in pg_index_pct]
    
    ax3.bar(x, pg_table_pct, width*2, label='Table data', color='#3498db')
    ax3.bar(x, pg_index_pct, width*2, bottom=pg_table_pct, label='IVFFlat index', color='#e74c3c')
    
    ax3.set_xlabel('Dataset (dimensions)')
    ax3.set_ylabel('Percentage of Total Storage')
    ax3.set_title('pgvector: Index vs Table Storage')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{d}d" for d in dims])
    ax3.legend(fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for i, (t_pct, i_pct) in enumerate(zip(pg_table_pct, pg_index_pct)):
        ax3.annotate(f'{t_pct:.0f}%', xy=(i, t_pct/2), ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        ax3.annotate(f'{i_pct:.0f}%', xy=(i, t_pct + i_pct/2), ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "index_storage_breakdown.png"), dpi=150)
    plt.close()
    print("Saved: index_storage_breakdown.png")


def plot_index_build_time(data, batch_size=1000):
    """
    Plot index build time and storage comparison across different index parameters.
    Focuses on 50k and 100k datasets, comparing HNSW and IVFFlat configurations.
    """
    runs = data["runs"]
    
    # Filter for 50k and 100k datasets with 128d (where we have multiple index configs)
    target_runs = [r for r in runs if r["vectors"] in [50000, 100000] and r["dimensions"] == 128]
    
    if not target_runs:
        print("Skipping index build time plot: no 50k/100k 128d datasets found")
        return
    
    # Collect all unique index configurations
    configs = []  # (size, index_type, params_str, index_time, storage_mb, total_time)
    
    for run in target_runs:
        size = run["vectors"]
        size_label = f"{size//1000}k"
        
        for p in run.get("pgvector", []):
            if p.get("batch_size") == batch_size:
                idx_type = p.get("index_type", "unknown")
                params = p.get("index_params", {})
                
                if idx_type == "hnsw":
                    params_str = f"m={params.get('m')}, ef={params.get('ef_construction')}"
                elif idx_type == "ivfflat":
                    params_str = f"lists={params.get('lists')}"
                else:
                    params_str = str(params)
                
                idx_time = p.get("duration_index_sec", 0)
                total_time = p.get("duration_ingest_sec", 0) + idx_time
                storage = p.get("storage_mb", 0)
                storage_before = p.get("storage_before_index_mb", 0)
                index_storage = storage - storage_before
                
                config_key = (size_label, "pgvector", idx_type, params_str)
                configs.append({
                    "size": size,
                    "size_label": size_label,
                    "db": "pgvector",
                    "index_type": idx_type,
                    "params_str": params_str,
                    "label": f"pg:{idx_type} ({params_str})",
                    "index_time": idx_time,
                    "total_time": total_time,
                    "index_pct": (idx_time / total_time * 100) if total_time > 0 else 0,
                    "storage_mb": storage,
                    "index_storage_mb": index_storage,
                })
        
        for c in run.get("chroma", []):
            if c.get("batch_size") == batch_size:
                params = c.get("index_params", {})
                
                if params.get("built_in"):
                    params_str = "built-in"
                else:
                    params_str = f"m={params.get('m')}, ef={params.get('ef_construction')}"
                
                total_time = c.get("duration_sec", 0)
                storage = c.get("storage_mb", 0)
                
                configs.append({
                    "size": size,
                    "size_label": size_label,
                    "db": "chroma",
                    "index_type": "hnsw",
                    "params_str": params_str,
                    "label": f"chroma:hnsw ({params_str})",
                    "index_time": total_time,  # Chroma builds index during insert
                    "total_time": total_time,
                    "index_pct": 100,  # All time is index+insert combined
                    "storage_mb": storage,
                    "index_storage_mb": storage,  # Chroma doesn't separate
                })
    
    if not configs:
        print("Skipping index build time plot: no configurations found")
        return
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors_pg = {'hnsw': '#e74c3c', 'ivfflat': '#3498db'}
    colors_chroma = '#2ecc71'
    
    # --- Plot 1: Index Build Time by Configuration (scatter/line) ---
    ax1 = axes[0, 0]
    
    # Group by size
    for size in [50000, 100000]:
        size_label = f"{size//1000}k"
        size_configs = [c for c in configs if c["size"] == size]
        
        # Sort by index time
        size_configs.sort(key=lambda x: x["index_time"])
        
        labels = [c["label"] for c in size_configs]
        times = [c["index_time"] for c in size_configs]
        colors = [colors_chroma if c["db"] == "chroma" else colors_pg.get(c["index_type"], "gray") for c in size_configs]
        
        y_pos = np.arange(len(labels))
        offset = 0.2 if size == 100000 else -0.2
        
        for i, (t, label, color) in enumerate(zip(times, labels, colors)):
            marker = 'o' if size == 50000 else 's'
            ax1.scatter(t, i, c=color, s=150, marker=marker, edgecolors='black', linewidths=0.5, zorder=3)
            ax1.annotate(f'{t:.1f}s', xy=(t, i), xytext=(5, 0), textcoords='offset points', 
                        fontsize=8, va='center')
        
        # Connect points for same size
        ax1.plot(times, y_pos, '--', alpha=0.3, color='gray')
    
    # Set labels from 50k configs (they should be similar)
    size_50k = [c for c in configs if c["size"] == 50000]
    size_50k.sort(key=lambda x: x["index_time"])
    ax1.set_yticks(range(len(size_50k)))
    ax1.set_yticklabels([c["label"] for c in size_50k], fontsize=9)
    ax1.set_xlabel('Index Build Time (seconds)', fontsize=10)
    ax1.set_title('Index Build Time Comparison (128d vectors)', fontsize=11)
    ax1.set_xscale('log')
    ax1.grid(axis='x', alpha=0.3)
    
    # Legend for sizes
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='50k vectors'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='100k vectors'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_chroma, markersize=10, label='Chroma'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_pg['hnsw'], markersize=10, label='pgvector HNSW'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_pg['ivfflat'], markersize=10, label='pgvector IVFFlat'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # --- Plot 2: Storage Comparison (horizontal lollipop) ---
    ax2 = axes[0, 1]
    
    # Use 50k data for cleaner comparison
    size_50k = [c for c in configs if c["size"] == 50000]
    size_50k.sort(key=lambda x: x["storage_mb"])
    
    labels = [c["label"] for c in size_50k]
    storage = [c["storage_mb"] for c in size_50k]
    colors = [colors_chroma if c["db"] == "chroma" else colors_pg.get(c["index_type"], "gray") for c in size_50k]
    
    y_pos = np.arange(len(labels))
    
    # Lollipop chart
    ax2.hlines(y=y_pos, xmin=0, xmax=storage, color=colors, alpha=0.7, linewidth=2)
    ax2.scatter(storage, y_pos, c=colors, s=150, edgecolors='black', linewidths=0.5, zorder=3)
    
    # Add raw data reference line
    raw_size = 50000 * 128 * 4 / (1024*1024)
    ax2.axvline(x=raw_size, color='gray', linestyle='--', alpha=0.5, label=f'Raw data ({raw_size:.1f} MB)')
    
    for i, s in enumerate(storage):
        overhead = (s / raw_size - 1) * 100
        ax2.annotate(f'{s:.1f} MB (+{overhead:.0f}%)', xy=(s, i), xytext=(5, 0), 
                    textcoords='offset points', fontsize=8, va='center')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel('Storage (MB)', fontsize=10)
    ax2.set_title('Storage Comparison (50k × 128d vectors)', fontsize=11)
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(axis='x', alpha=0.3)
    
    # --- Plot 3: Time vs Storage Trade-off (scatter) ---
    ax3 = axes[1, 0]
    
    for size in [50000, 100000]:
        size_label = f"{size//1000}k"
        size_configs = [c for c in configs if c["size"] == size]
        
        for c in size_configs:
            color = colors_chroma if c["db"] == "chroma" else colors_pg.get(c["index_type"], "gray")
            marker = 'o' if size == 50000 else 's'
            ax3.scatter(c["index_time"], c["storage_mb"], c=color, s=150, marker=marker,
                       edgecolors='black', linewidths=0.5, alpha=0.8, zorder=3)
            
            # Label only some points to avoid clutter
            if c["index_time"] > 5 or c["storage_mb"] > 70:
                ax3.annotate(c["params_str"], xy=(c["index_time"], c["storage_mb"]),
                           xytext=(5, 5), textcoords='offset points', fontsize=7)
    
    ax3.set_xlabel('Index Build Time (seconds)', fontsize=10)
    ax3.set_ylabel('Total Storage (MB)', fontsize=10)
    ax3.set_title('Time vs Storage Trade-off', fontsize=11)
    ax3.set_xscale('log')
    ax3.grid(alpha=0.3)
    ax3.legend(handles=legend_elements[:5], loc='upper left', fontsize=8)
    
    # --- Plot 4: Index Build Time as % of Total (for pgvector only) ---
    ax4 = axes[1, 1]
    
    # Only pgvector has meaningful separate index time
    pg_configs = [c for c in configs if c["db"] == "pgvector" and c["size"] == 50000]
    pg_configs.sort(key=lambda x: x["index_pct"], reverse=True)
    
    labels = [f"{c['index_type']} ({c['params_str']})" for c in pg_configs]
    pcts = [c["index_pct"] for c in pg_configs]
    times = [c["index_time"] for c in pg_configs]
    colors = [colors_pg.get(c["index_type"], "gray") for c in pg_configs]
    
    y_pos = np.arange(len(labels))
    
    # Horizontal lollipop
    ax4.hlines(y=y_pos, xmin=0, xmax=pcts, color=colors, alpha=0.7, linewidth=2)
    ax4.scatter(pcts, y_pos, c=colors, s=150, edgecolors='black', linewidths=0.5, zorder=3)
    
    for i, (pct, t) in enumerate(zip(pcts, times)):
        ax4.annotate(f'{pct:.1f}% ({t:.1f}s)', xy=(pct, i), xytext=(5, 0),
                    textcoords='offset points', fontsize=9, va='center')
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(labels, fontsize=9)
    ax4.set_xlabel('Index Build Time (% of Total Ingestion)', fontsize=10)
    ax4.set_title('pgvector: Index Build as % of Total (50k × 128d)', fontsize=11)
    ax4.set_xlim(0, 105)
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "index_build_time.png"), dpi=150)
    plt.close()
    print("Saved: index_build_time.png")



def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    data = load_results()
    data["runs"] = sorted(data["runs"], key=lambda r: r["vectors"])
    print(f"Loaded {len(data['runs'])} datasets from {RESULTS_FILE}")
    
    print_available_filters(data)
    
    plot_memory_usage(data)
    plot_dimensionality_impact(filter_data(data, pg_index_type="ivfflat", pg_lists=100))
    plot_index_storage_breakdown(data, batch_size=1000)
    plot_index_build_time(data, batch_size=1000)
    print_summary_table(data)
    
    plot_throughput_comparison(data, dimensions=[128], pg_index_type='ivfflat', pg_lists=100, chroma_builtin=True)
    plot_time_breakdown(data, dimensions=[128], pg_index_type='ivfflat', pg_lists=100, chroma_builtin=True)
    plot_storage_comparison(data, dimensions=[128], pg_index_type='ivfflat', pg_lists=100, chroma_builtin=True)
    plot_resource_usage(data, dimensions=[128], pg_index_type='ivfflat', pg_lists=100, chroma_builtin=True)

    
    print(f"\nAll plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
