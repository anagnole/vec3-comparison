#!/usr/bin/env python3
"""
Query benchmark plotting for Chroma vs pgvector comparison.

Generates plots for:
- Latency comparison by dataset size
- Recall comparison by dataset size  
- Latency vs Recall trade-off
- Top-K impact on latency and recall
- Filter impact analysis
- Index type comparison (when multiple available)
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_FILE = "results/raw/all_query_results.json"
PLOTS_DIR = "results/plots/queries"

# Color scheme
COLORS = {
    'chroma': '#2ecc71',
    'pgvector': '#3498db',
    'chroma_dark': '#27ae60',
    'pgvector_dark': '#2980b9',
    'hnsw': '#e74c3c',
    'ivfflat': '#9b59b6',
}


def load_results():
    """Load query benchmark results from JSON file."""
    with open(RESULTS_FILE) as f:
        return json.load(f)


def sort_runs_by_size(runs):
    """Sort runs by vector count for consistent ordering."""
    return sorted(runs, key=lambda r: r["vectors"])


def filter_data(data, modes=None, index_types=None, datasets=None, dimensions=None):
    """Filter runs by various criteria."""
    runs = data["runs"]
    filtered = []
    for r in runs:
        if modes and r["mode"] not in modes:
            continue
        if index_types and r.get("index_type") not in index_types:
            continue
        if datasets and r["dataset"] not in datasets:
            continue
        if dimensions and r["dimensions"] not in dimensions:
            continue
        filtered.append(r)
    return {"runs": filtered}


def get_available_filters(data):
    """Get unique values for all filter dimensions."""
    runs = data["runs"]
    return {
        'dimensions': sorted(set(r["dimensions"] for r in runs)),
        'sizes': sorted(set(r["vectors"] for r in runs)),
        'datasets': list(dict.fromkeys(r["dataset"] for r in runs)),
        'modes': sorted(set(r["mode"] for r in runs)),
        'index_types': sorted(set(r.get("index_type", "unknown") for r in runs)),
        'top_ks': sorted(set(tk["top_k"] for r in runs for tk in r["chroma"])),
    }


def get_metric(run, db, top_k, metric_path):
    """
    Extract a metric value from a run.
    
    Args:
        run: The run dict
        db: 'chroma' or 'pgvector'
        top_k: The top_k value to look for
        metric_path: Tuple like ('latency', 'mean') or ('recall', 'recall_mean')
    """
    results = run.get(db, [])
    entry = next((r for r in results if r.get("top_k") == top_k), None)
    if not entry:
        return None
    
    val = entry
    for key in metric_path:
        if isinstance(val, dict):
            val = val.get(key)
        else:
            return None
    return val



def plot_latency_comparison(data):

    runs = sort_runs_by_size([r for r in data["runs"] if r["mode"] == "nofilter"])
    if not runs:
        print("Skipping latency comparison: no nofilter runs")
        return
    
    # Deduplicate by dataset (take first if multiple index types)
    seen = set()
    unique_runs = []
    for r in runs:
        if r["dataset"] not in seen:
            seen.add(r["dataset"])
            unique_runs.append(r)
    runs = unique_runs
    
    datasets = [r["dataset"] for r in runs]
    top_ks = [10, 50, 100]
    
    for top_k in top_ks:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(datasets))
        width = 0.35
        
        chroma_lat = [get_metric(r, 'chroma', top_k, ('latency', 'mean')) or 0 for r in runs]
        pg_lat = [get_metric(r, 'pgvector', top_k, ('latency', 'mean')) or 0 for r in runs]
        
        # Convert to ms
        chroma_lat = [v * 1000 for v in chroma_lat]
        pg_lat = [v * 1000 for v in pg_lat]
        
        bars1 = ax.bar(x - width/2, chroma_lat, width, label='Chroma', color=COLORS['chroma'])
        bars2 = ax.bar(x + width/2, pg_lat, width, label='pgvector', color=COLORS['pgvector'])
        
        # Add value labels
        for bar, val in zip(bars1, chroma_lat):
            if val > 0:
                ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars2, pg_lat):
            if val > 0:
                ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Mean Latency (ms)')
        ax.set_title(f'Query Latency Comparison (top_k={top_k})')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"latency_k{top_k}.png"), dpi=150)
        plt.close()
        print(f"Saved: latency_k{top_k}.png")


def plot_recall_comparison(data):
    """
    Plot recall comparison between Chroma and pgvector.
    Generates separate plots for each top_k value.
    """
    runs = sort_runs_by_size([r for r in data["runs"] if r["mode"] == "nofilter"])
    if not runs:
        print("Skipping recall comparison: no nofilter runs")
        return
    
    # Deduplicate by dataset
    seen = set()
    unique_runs = []
    for r in runs:
        if r["dataset"] not in seen:
            seen.add(r["dataset"])
            unique_runs.append(r)
    runs = unique_runs
    
    datasets = [r["dataset"] for r in runs]
    top_ks = [10, 50, 100]
    
    for top_k in top_ks:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(datasets))
        width = 0.35
        
        chroma_recall = [get_metric(r, 'chroma', top_k, ('recall', 'recall_mean')) or 0 for r in runs]
        pg_recall = [get_metric(r, 'pgvector', top_k, ('recall', 'recall_mean')) or 0 for r in runs]
        
        bars1 = ax.bar(x - width/2, chroma_recall, width, label='Chroma', color=COLORS['chroma'])
        bars2 = ax.bar(x + width/2, pg_recall, width, label='pgvector', color=COLORS['pgvector'])
        
        # Add value labels
        for bar, val in zip(bars1, chroma_recall):
            if val > 0:
                ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars2, pg_recall):
            if val > 0:
                ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Recall@k')
        ax.set_title(f'Query Recall Comparison (top_k={top_k})')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"recall_k{top_k}.png"), dpi=150)
        plt.close()
        print(f"Saved: recall_k{top_k}.png")


def plot_latency_vs_recall(data):
    """
    Scatter plot of latency vs recall trade-off.
    Each point is a (database, dataset, top_k) combination.
    """
    runs = sort_runs_by_size([r for r in data["runs"] if r["mode"] == "nofilter"])
    if not runs:
        print("Skipping latency vs recall: no nofilter runs")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Collect all points
    chroma_points = []  # (latency_ms, recall, label)
    pg_points = []
    
    for run in runs:
        ds = run["dataset"]
        for entry in run.get("chroma", []):
            top_k = entry.get("top_k")
            lat = entry.get("latency", {}).get("mean")
            rec = entry.get("recall", {}).get("recall_mean")
            if lat and rec:
                chroma_points.append((lat * 1000, rec, f"{ds} k={top_k}"))
        
        for entry in run.get("pgvector", []):
            top_k = entry.get("top_k")
            lat = entry.get("latency", {}).get("mean")
            rec = entry.get("recall", {}).get("recall_mean")
            if lat and rec:
                pg_points.append((lat * 1000, rec, f"{ds} k={top_k}"))
    
    # Plot points
    if chroma_points:
        lats, recs, labels = zip(*chroma_points)
        ax.scatter(lats, recs, c=COLORS['chroma'], s=100, alpha=0.7, label='Chroma', edgecolors='white')
    
    if pg_points:
        lats, recs, labels = zip(*pg_points)
        ax.scatter(lats, recs, c=COLORS['pgvector'], s=100, alpha=0.7, label='pgvector', marker='s', edgecolors='white')
    
    ax.set_xlabel('Mean Latency (ms)')
    ax.set_ylabel('Recall@k')
    ax.set_title('Latency vs Recall Trade-off (nofilter mode)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Add ideal zone annotation
    ax.annotate('Ideal: low latency, high recall', xy=(0.05, 0.95), 
                xycoords='axes fraction', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "latency_vs_recall.png"), dpi=150)
    plt.close()
    print("Saved: latency_vs_recall.png")


def plot_topk_impact(data):
    runs = sort_runs_by_size([r for r in data["runs"] if r["mode"] == "nofilter"])
    if not runs:
        print("Skipping top-k impact: no nofilter runs")
        return
    
    # Deduplicate by dataset
    seen = set()
    unique_runs = []
    for r in runs:
        if r["dataset"] not in seen:
            seen.add(r["dataset"])
            unique_runs.append(r)
    runs = unique_runs
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(runs)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, run in enumerate(runs):
        top_ks = [e["top_k"] for e in run["chroma"]]
        lats = [e["latency"]["mean"] * 1000 for e in run["chroma"]]
        ax.plot(top_ks, lats, 'o-', label=run["dataset"], color=colors[i], linewidth=2, markersize=8)
    ax.set_xlabel('Top-K')
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title('Chroma: Latency vs Top-K')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "topk_chroma_latency.png"), dpi=150)
    plt.close()
    print("Saved: topk_chroma_latency.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, run in enumerate(runs):
        top_ks = [e["top_k"] for e in run["pgvector"]]
        lats = [e["latency"]["mean"] * 1000 for e in run["pgvector"]]
        ax.plot(top_ks, lats, 's-', label=run["dataset"], color=colors[i], linewidth=2, markersize=8)
    ax.set_xlabel('Top-K')
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title('pgvector: Latency vs Top-K')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "topk_pgvector_latency.png"), dpi=150)
    plt.close()
    print("Saved: topk_pgvector_latency.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, run in enumerate(runs):
        top_ks = [e["top_k"] for e in run["chroma"]]
        recs = [e.get("recall", {}).get("recall_mean", 0) for e in run["chroma"]]
        ax.plot(top_ks, recs, 'o-', label=run["dataset"], color=colors[i], linewidth=2, markersize=8)
    ax.set_xlabel('Top-K')
    ax.set_ylabel('Recall@k')
    ax.set_title('Chroma: Recall vs Top-K')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "topk_chroma_recall.png"), dpi=150)
    plt.close()
    print("Saved: topk_chroma_recall.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, run in enumerate(runs):
        top_ks = [e["top_k"] for e in run["pgvector"]]
        recs = [e.get("recall", {}).get("recall_mean", 0) for e in run["pgvector"]]
        ax.plot(top_ks, recs, 's-', label=run["dataset"], color=colors[i], linewidth=2, markersize=8)
    ax.set_xlabel('Top-K')
    ax.set_ylabel('Recall@k')
    ax.set_title('pgvector: Recall vs Top-K')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "topk_pgvector_recall.png"), dpi=150)
    plt.close()
    print("Saved: topk_pgvector_recall.png")


def plot_filter_impact(data):
    """
    Compare nofilter vs filter mode for latency and recall.
    Generates separate plots for each metric.
    """
    runs = data["runs"]
    
    # Find datasets that have both modes
    datasets_with_both = []
    for run in runs:
        ds = run["dataset"]
        has_nofilter = any(r["dataset"] == ds and r["mode"] == "nofilter" for r in runs)
        has_filter = any(r["dataset"] == ds and r["mode"] == "filter" for r in runs)
        if has_nofilter and has_filter and ds not in datasets_with_both:
            datasets_with_both.append(ds)
    
    if not datasets_with_both:
        print("Skipping filter impact: need both filter and nofilter runs")
        return
    
    # Sort by size
    size_map = {r["dataset"]: r["vectors"] for r in runs}
    datasets_with_both.sort(key=lambda d: size_map.get(d, 0))
    
    x = np.arange(len(datasets_with_both))
    width = 0.2
    top_k = 10  # Use top_k=10 for comparison
    
    # Collect data
    chroma_nf_lat, chroma_f_lat = [], []
    pg_nf_lat, pg_f_lat = [], []
    chroma_nf_rec, chroma_f_rec = [], []
    pg_nf_rec, pg_f_rec = [], []
    
    for ds in datasets_with_both:
        nf = next((r for r in runs if r["dataset"] == ds and r["mode"] == "nofilter"), None)
        f = next((r for r in runs if r["dataset"] == ds and r["mode"] == "filter"), None)
        
        chroma_nf_lat.append((get_metric(nf, 'chroma', top_k, ('latency', 'mean')) or 0) * 1000)
        chroma_f_lat.append((get_metric(f, 'chroma', top_k, ('latency', 'mean')) or 0) * 1000)
        pg_nf_lat.append((get_metric(nf, 'pgvector', top_k, ('latency', 'mean')) or 0) * 1000)
        pg_f_lat.append((get_metric(f, 'pgvector', top_k, ('latency', 'mean')) or 0) * 1000)
        
        chroma_nf_rec.append(get_metric(nf, 'chroma', top_k, ('recall', 'recall_mean')) or 0)
        chroma_f_rec.append(get_metric(f, 'chroma', top_k, ('recall', 'recall_mean')) or 0)
        pg_nf_rec.append(get_metric(nf, 'pgvector', top_k, ('recall', 'recall_mean')) or 0)
        pg_f_rec.append(get_metric(f, 'pgvector', top_k, ('recall', 'recall_mean')) or 0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, chroma_nf_lat, width, label='Chroma (nofilter)', color=COLORS['chroma'])
    ax.bar(x - 0.5*width, chroma_f_lat, width, label='Chroma (filter)', color=COLORS['chroma_dark'])
    ax.bar(x + 0.5*width, pg_nf_lat, width, label='pgvector (nofilter)', color=COLORS['pgvector'])
    ax.bar(x + 1.5*width, pg_f_lat, width, label='pgvector (filter)', color=COLORS['pgvector_dark'])
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title(f'Filter Impact on Latency (top_k={top_k})')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_with_both)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "filter_latency.png"), dpi=150)
    plt.close()
    print("Saved: filter_latency.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    chroma_ratio = [f/nf if nf > 0 else 0 for nf, f in zip(chroma_nf_lat, chroma_f_lat)]
    pg_ratio = [f/nf if nf > 0 else 0 for nf, f in zip(pg_nf_lat, pg_f_lat)]
    
    ax.bar(x - width/2, chroma_ratio, width*1.5, label='Chroma', color=COLORS['chroma'])
    ax.bar(x + width/2, pg_ratio, width*1.5, label='pgvector', color=COLORS['pgvector'])
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Latency Ratio (filter / nofilter)')
    ax.set_title('Filter Overhead (>1 = filter is slower)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_with_both)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "filter_overhead.png"), dpi=150)
    plt.close()
    print("Saved: filter_overhead.png")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, chroma_nf_rec, width, label='Chroma (nofilter)', color=COLORS['chroma'])
    ax.bar(x - 0.5*width, chroma_f_rec, width, label='Chroma (filter)', color=COLORS['chroma_dark'])
    ax.bar(x + 0.5*width, pg_nf_rec, width, label='pgvector (nofilter)', color=COLORS['pgvector'])
    ax.bar(x + 1.5*width, pg_f_rec, width, label='pgvector (filter)', color=COLORS['pgvector_dark'])
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Recall@k')
    ax.set_title(f'Filter Impact on Recall (top_k={top_k})')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_with_both)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "filter_recall.png"), dpi=150)
    plt.close()
    print("Saved: filter_recall.png")
    
    # Plot 4: Recall drop
    fig, ax = plt.subplots(figsize=(10, 6))
    chroma_drop = [nf - f for nf, f in zip(chroma_nf_rec, chroma_f_rec)]
    pg_drop = [nf - f for nf, f in zip(pg_nf_rec, pg_f_rec)]
    
    ax.bar(x - width/2, chroma_drop, width*1.5, label='Chroma', color=COLORS['chroma'])
    ax.bar(x + width/2, pg_drop, width*1.5, label='pgvector', color=COLORS['pgvector'])
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Recall Drop (nofilter - filter)')
    ax.set_title('Recall Degradation from Filtering (>0 = filter hurts recall)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_with_both)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "filter_recall_drop.png"), dpi=150)
    plt.close()
    print("Saved: filter_recall_drop.png")


def plot_scaling_analysis(data):
    runs = sort_runs_by_size([r for r in data["runs"] if r["mode"] == "nofilter"])
    if len(runs) < 2:
        print("Skipping scaling analysis: need at least 2 datasets")
        return
    
    # Deduplicate by dataset
    seen = set()
    unique_runs = []
    for r in runs:
        if r["dataset"] not in seen:
            seen.add(r["dataset"])
            unique_runs.append(r)
    runs = unique_runs
    
    top_k = 10
    sizes = [r["vectors"] for r in runs]
    labels = [r["dataset"] for r in runs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    chroma_lat = [(get_metric(r, 'chroma', top_k, ('latency', 'mean')) or 0) * 1000 for r in runs]
    pg_lat = [(get_metric(r, 'pgvector', top_k, ('latency', 'mean')) or 0) * 1000 for r in runs]
    
    ax.plot(sizes, chroma_lat, 'o-', label='Chroma', color=COLORS['chroma'], linewidth=2, markersize=10)
    ax.plot(sizes, pg_lat, 's-', label='pgvector', color=COLORS['pgvector'], linewidth=2, markersize=10)
    
    ax.set_xlabel('Number of Vectors')
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title(f'Latency Scaling with Dataset Size (top_k={top_k})')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add dataset labels
    for i, (s, l) in enumerate(zip(sizes, labels)):
        ax.annotate(l, (s, max(chroma_lat[i], pg_lat[i])), 
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "scaling_latency.png"), dpi=150)
    plt.close()
    print("Saved: scaling_latency.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    chroma_rec = [get_metric(r, 'chroma', top_k, ('recall', 'recall_mean')) or 0 for r in runs]
    pg_rec = [get_metric(r, 'pgvector', top_k, ('recall', 'recall_mean')) or 0 for r in runs]
    
    ax.plot(sizes, chroma_rec, 'o-', label='Chroma', color=COLORS['chroma'], linewidth=2, markersize=10)
    ax.plot(sizes, pg_rec, 's-', label='pgvector', color=COLORS['pgvector'], linewidth=2, markersize=10)
    
    ax.set_xlabel('Number of Vectors')
    ax.set_ylabel('Recall@k')
    ax.set_title(f'Recall Scaling with Dataset Size (top_k={top_k})')
    ax.set_xscale('log')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "scaling_recall.png"), dpi=150)
    plt.close()
    print("Saved: scaling_recall.png")


def plot_p99_latency(data):
    """
    Plot P99 tail latency comparison.
    """
    runs = sort_runs_by_size([r for r in data["runs"] if r["mode"] == "nofilter"])
    if not runs:
        print("Skipping P99 latency: no nofilter runs")
        return
    
    # Deduplicate by dataset
    seen = set()
    unique_runs = []
    for r in runs:
        if r["dataset"] not in seen:
            seen.add(r["dataset"])
            unique_runs.append(r)
    runs = unique_runs
    
    datasets = [r["dataset"] for r in runs]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(datasets))
    width = 0.25
    top_ks = [10, 50, 100]
    
    for i, top_k in enumerate(top_ks):
        chroma_p99 = [(get_metric(r, 'chroma', top_k, ('latency', 'p99')) or 0) * 1000 for r in runs]
        pg_p99 = [(get_metric(r, 'pgvector', top_k, ('latency', 'p99')) or 0) * 1000 for r in runs]
        
        offset = (i - 1) * width
        ax.bar(x + offset - width/4, chroma_p99, width/2, 
               label=f'Chroma k={top_k}' if i == 0 else '', color=COLORS['chroma'], alpha=0.4 + 0.2*i)
        ax.bar(x + offset + width/4, pg_p99, width/2,
               label=f'pgvector k={top_k}' if i == 0 else '', color=COLORS['pgvector'], alpha=0.4 + 0.2*i)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('P99 Latency (ms)')
    ax.set_title('P99 Tail Latency (nofilter)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['chroma'], label='Chroma'),
        Patch(facecolor=COLORS['pgvector'], label='pgvector'),
    ]
    ax.legend(handles=legend_elements)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "p99_latency.png"), dpi=150)
    plt.close()
    print("Saved: p99_latency.png")


def plot_throughput_comparison(data):
    """
    Plot queries per second (throughput) comparison.
    """
    runs = sort_runs_by_size([r for r in data["runs"] if r["mode"] == "nofilter"])
    if not runs:
        print("Skipping throughput: no nofilter runs")
        return
    
    # Deduplicate by dataset
    seen = set()
    unique_runs = []
    for r in runs:
        if r["dataset"] not in seen:
            seen.add(r["dataset"])
            unique_runs.append(r)
    runs = unique_runs
    
    datasets = [r["dataset"] for r in runs]
    top_k = 10
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    chroma_qps = []
    pg_qps = []
    
    for r in runs:
        c_lat = get_metric(r, 'chroma', top_k, ('latency', 'mean')) or 1
        p_lat = get_metric(r, 'pgvector', top_k, ('latency', 'mean')) or 1
        chroma_qps.append(1 / c_lat if c_lat > 0 else 0)
        pg_qps.append(1 / p_lat if p_lat > 0 else 0)
    
    bars1 = ax.bar(x - width/2, chroma_qps, width, label='Chroma', color=COLORS['chroma'])
    bars2 = ax.bar(x + width/2, pg_qps, width, label='pgvector', color=COLORS['pgvector'])
    
    # Add value labels
    for bar, val in zip(bars1, chroma_qps):
        ax.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, pg_qps):
        ax.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Queries per Second')
    ax.set_title(f'Query Throughput (top_k={top_k}, nofilter)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "throughput.png"), dpi=150)
    plt.close()
    print("Saved: throughput.png")


def plot_combined_summary(data):
    runs = sort_runs_by_size([r for r in data["runs"] if r["mode"] == "nofilter"])
    if not runs:
        print("Skipping combined summary: no nofilter runs")
        return
    
    # Deduplicate by dataset
    seen = set()
    unique_runs = []
    for r in runs:
        if r["dataset"] not in seen:
            seen.add(r["dataset"])
            unique_runs.append(r)
    runs = unique_runs
    
    datasets = [r["dataset"] for r in runs]
    top_k = 10
    x = np.arange(len(datasets))
    width = 0.35
    
    # Compute all metrics once
    chroma_lat = [(get_metric(r, 'chroma', top_k, ('latency', 'mean')) or 0) * 1000 for r in runs]
    pg_lat = [(get_metric(r, 'pgvector', top_k, ('latency', 'mean')) or 0) * 1000 for r in runs]
    chroma_rec = [get_metric(r, 'chroma', top_k, ('recall', 'recall_mean')) or 0 for r in runs]
    pg_rec = [get_metric(r, 'pgvector', top_k, ('recall', 'recall_mean')) or 0 for r in runs]
    chroma_qps = [1 / (get_metric(r, 'chroma', top_k, ('latency', 'mean')) or 1) for r in runs]
    pg_qps = [1 / (get_metric(r, 'pgvector', top_k, ('latency', 'mean')) or 1) for r in runs]
    speedup = [c / p if p > 0 else 0 for c, p in zip(chroma_lat, pg_lat)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, chroma_lat, width, label='Chroma', color=COLORS['chroma'])
    ax.bar(x + width/2, pg_lat, width, label='pgvector', color=COLORS['pgvector'])
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title(f'Query Latency Summary (top_k={top_k})')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "summary_latency.png"), dpi=150)
    plt.close()
    print("Saved: summary_latency.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, chroma_rec, width, label='Chroma', color=COLORS['chroma'])
    ax.bar(x + width/2, pg_rec, width, label='pgvector', color=COLORS['pgvector'])
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Recall@k')
    ax.set_title(f'Query Recall Summary (top_k={top_k})')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "summary_recall.png"), dpi=150)
    plt.close()
    print("Saved: summary_recall.png")
    
    # Plot 3: Throughput
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, chroma_qps, width, label='Chroma', color=COLORS['chroma'])
    ax.bar(x + width/2, pg_qps, width, label='pgvector', color=COLORS['pgvector'])
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Queries per Second')
    ax.set_title(f'Query Throughput Summary (top_k={top_k})')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "summary_throughput.png"), dpi=150)
    plt.close()
    print("Saved: summary_throughput.png")
    
    # Plot 4: Speedup ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_list = [COLORS['pgvector'] if s > 1 else COLORS['chroma'] for s in speedup]
    
    bars = ax.bar(x, speedup, width*1.5, color=colors_list)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Latency Ratio (Chroma / pgvector)')
    ax.set_title('Relative Speed (>1 = pgvector faster)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, speedup):
        label = f'{val:.1f}x'
        ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "summary_speedup.png"), dpi=150)
    plt.close()
    print("Saved: summary_speedup.png")


def print_summary_table(data):
    """Print a formatted summary table of all results."""
    print("\n" + "="*130)
    print("QUERY BENCHMARK SUMMARY")
    print("="*130)
    
    print(f"\n{'Dataset':<10} {'Mode':<10} {'Index':<8} {'Vectors':>10} {'Dim':>5} | "
          f"{'Chroma Lat':>10} {'Recall':>8} | {'PG Lat':>10} {'Recall':>8} | {'Speedup':>8}")
    print("-" * 130)
    
    for run in sorted(data["runs"], key=lambda r: (r["vectors"], r["mode"])):
        for top_k in [10, 50, 100]:
            c_lat = (get_metric(run, 'chroma', top_k, ('latency', 'mean')) or 0) * 1000
            c_rec = get_metric(run, 'chroma', top_k, ('recall', 'recall_mean')) or 0
            p_lat = (get_metric(run, 'pgvector', top_k, ('latency', 'mean')) or 0) * 1000
            p_rec = get_metric(run, 'pgvector', top_k, ('recall', 'recall_mean')) or 0
            speedup = c_lat / p_lat if p_lat > 0 else 0
            
            ds_label = run['dataset'] if top_k == 10 else ''
            mode_label = run['mode'] if top_k == 10 else ''
            idx_label = run.get('index_type', 'n/a') if top_k == 10 else ''
            vec_label = f"{run['vectors']:,}" if top_k == 10 else ''
            dim_label = str(run['dimensions']) if top_k == 10 else ''
            
            print(f"{ds_label:<10} {mode_label:<10} {idx_label:<8} {vec_label:>10} {dim_label:>5} | "
                  f"k={top_k:>3} {c_lat:>6.2f}ms {c_rec:>7.2%} | {p_lat:>6.2f}ms {p_rec:>7.2%} | {speedup:>7.1f}x")
        print("-" * 130)


def get_index_param_label(run):
    """Generate a label for index parameters."""
    idx_type = run.get("index_type", "unknown")
    params = run.get("index_params", {})
    
    if idx_type == "hnsw":
        m = params.get("m", "?")
        ef = params.get("ef_construction", "?")
        return f"HNSW(m={m},ef={ef})"
    elif idx_type == "ivfflat":
        lists = params.get("lists", "?")
        return f"IVFFlat(lists={lists})"
    else:
        return idx_type


def plot_index_params_latency(data):
    runs = [r for r in data["runs"] if r["mode"] == "nofilter"]
    
    # Group by dataset
    by_dataset = {}
    for run in runs:
        ds = run["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(run)
    
    # Find datasets with multiple index configs
    multi_config_datasets = {k: v for k, v in by_dataset.items() if len(v) > 1}
    
    if not multi_config_datasets:
        print("Skipping index params latency: no datasets with multiple index configurations")
        return
    
    for dataset, runs_list in multi_config_datasets.items():
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by index type then params
        runs_list = sorted(runs_list, key=lambda r: (r.get("index_type", ""), str(r.get("index_params", {}))))
        
        labels = [get_index_param_label(r) for r in runs_list]
        x = np.arange(len(labels))
        width = 0.35
        top_k = 10
        
        chroma_lat = [(get_metric(r, 'chroma', top_k, ('latency', 'mean')) or 0) * 1000 for r in runs_list]
        pg_lat = [(get_metric(r, 'pgvector', top_k, ('latency', 'mean')) or 0) * 1000 for r in runs_list]
        
        bars1 = ax.bar(x - width/2, chroma_lat, width, label='Chroma', color=COLORS['chroma'])
        bars2 = ax.bar(x + width/2, pg_lat, width, label='pgvector', color=COLORS['pgvector'])
        
        # Add value labels
        for bar, val in zip(bars1, chroma_lat):
            if val > 0:
                ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, pg_lat):
            if val > 0:
                ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Index Configuration')
        ax.set_ylabel('Mean Latency (ms)')
        ax.set_title(f'Index Parameter Impact on Latency ({dataset}, top_k={top_k})')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"index_params_latency_{dataset}.png"), dpi=150)
        plt.close()
        print(f"Saved: index_params_latency_{dataset}.png")


def plot_index_params_recall(data):
    runs = [r for r in data["runs"] if r["mode"] == "nofilter"]
    
    # Group by dataset
    by_dataset = {}
    for run in runs:
        ds = run["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(run)
    
    # Find datasets with multiple index configs
    multi_config_datasets = {k: v for k, v in by_dataset.items() if len(v) > 1}
    
    if not multi_config_datasets:
        print("Skipping index params recall: no datasets with multiple index configurations")
        return
    
    for dataset, runs_list in multi_config_datasets.items():
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by index type then params
        runs_list = sorted(runs_list, key=lambda r: (r.get("index_type", ""), str(r.get("index_params", {}))))
        
        labels = [get_index_param_label(r) for r in runs_list]
        x = np.arange(len(labels))
        width = 0.35
        top_k = 10
        
        chroma_rec = [get_metric(r, 'chroma', top_k, ('recall', 'recall_mean')) or 0 for r in runs_list]
        pg_rec = [get_metric(r, 'pgvector', top_k, ('recall', 'recall_mean')) or 0 for r in runs_list]
        
        bars1 = ax.bar(x - width/2, chroma_rec, width, label='Chroma', color=COLORS['chroma'])
        bars2 = ax.bar(x + width/2, pg_rec, width, label='pgvector', color=COLORS['pgvector'])
        
        # Add value labels
        for bar, val in zip(bars1, chroma_rec):
            if val > 0:
                ax.annotate(f'{val:.2%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, pg_rec):
            if val > 0:
                ax.annotate(f'{val:.2%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Index Configuration')
        ax.set_ylabel('Recall@k')
        ax.set_title(f'Index Parameter Impact on Recall ({dataset}, top_k={top_k})')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"index_params_recall_{dataset}.png"), dpi=150)
        plt.close()
        print(f"Saved: index_params_recall_{dataset}.png")


def plot_index_params_tradeoff(data):
    runs = [r for r in data["runs"] if r["mode"] == "nofilter"]
    
    # Group by dataset
    by_dataset = {}
    for run in runs:
        ds = run["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(run)
    
    # Find datasets with multiple index configs
    multi_config_datasets = {k: v for k, v in by_dataset.items() if len(v) > 1}
    
    if not multi_config_datasets:
        print("Skipping index params tradeoff: no datasets with multiple index configurations")
        return
    
    for dataset, runs_list in multi_config_datasets.items():
        fig, ax = plt.subplots(figsize=(10, 7))
        
        top_k = 10
        
        for run in runs_list:
            label = get_index_param_label(run)
            idx_type = run.get("index_type", "unknown")
            
            # Chroma point
            c_lat = (get_metric(run, 'chroma', top_k, ('latency', 'mean')) or 0) * 1000
            c_rec = get_metric(run, 'chroma', top_k, ('recall', 'recall_mean')) or 0
            
            # pgvector point
            p_lat = (get_metric(run, 'pgvector', top_k, ('latency', 'mean')) or 0) * 1000
            p_rec = get_metric(run, 'pgvector', top_k, ('recall', 'recall_mean')) or 0
            
            # Color based on index type
            if idx_type == "hnsw":
                marker_pg = 's'
                color_pg = COLORS['hnsw']
            else:
                marker_pg = '^'
                color_pg = COLORS['ivfflat']
            
            # Plot Chroma (always HNSW)
            ax.scatter(c_lat, c_rec, c=COLORS['chroma'], s=150, marker='o', 
                      edgecolors='black', linewidths=0.5, zorder=3)
            ax.annotate(f'Chroma\n{label}', xy=(c_lat, c_rec), xytext=(5, 5),
                       textcoords='offset points', fontsize=7, ha='left')
            
            # Plot pgvector
            ax.scatter(p_lat, p_rec, c=color_pg, s=150, marker=marker_pg,
                      edgecolors='black', linewidths=0.5, zorder=3)
            ax.annotate(f'pgvector\n{label}', xy=(p_lat, p_rec), xytext=(5, -10),
                       textcoords='offset points', fontsize=7, ha='left')
        
        ax.set_xlabel('Mean Latency (ms)')
        ax.set_ylabel('Recall@k')
        ax.set_title(f'Latency vs Recall Tradeoff by Index Configuration ({dataset}, top_k={top_k})')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
        ax.grid(alpha=0.3)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['chroma'], 
                   markersize=10, label='Chroma'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['hnsw'], 
                   markersize=10, label='pgvector HNSW'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['ivfflat'], 
                   markersize=10, label='pgvector IVFFlat'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"index_params_tradeoff_{dataset}.png"), dpi=150)
        plt.close()
        print(f"Saved: index_params_tradeoff_{dataset}.png")


def plot_index_type_comparison(data):
    runs = [r for r in data["runs"] if r["mode"] == "nofilter"]
    
    # Group by dataset
    by_dataset = {}
    for run in runs:
        ds = run["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(run)
    
    # Find datasets with both HNSW and IVFFlat
    datasets_with_both = {}
    for ds, runs_list in by_dataset.items():
        idx_types = set(r.get("index_type") for r in runs_list)
        if "hnsw" in idx_types and "ivfflat" in idx_types:
            datasets_with_both[ds] = runs_list
    
    if not datasets_with_both:
        print("Skipping index type comparison: no datasets with both HNSW and IVFFlat")
        return
    
    for dataset, runs_list in datasets_with_both.items():
        # Separate by index type
        hnsw_runs = [r for r in runs_list if r.get("index_type") == "hnsw"]
        ivfflat_runs = [r for r in runs_list if r.get("index_type") == "ivfflat"]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        top_k = 10
        
        # Collect all configs
        all_configs = []
        for r in hnsw_runs:
            params = r.get("index_params", {})
            all_configs.append({
                "label": f"HNSW(m={params.get('m')},ef={params.get('ef_construction')})",
                "type": "hnsw",
                "latency": (get_metric(r, 'pgvector', top_k, ('latency', 'mean')) or 0) * 1000,
                "recall": get_metric(r, 'pgvector', top_k, ('recall', 'recall_mean')) or 0
            })
        for r in ivfflat_runs:
            params = r.get("index_params", {})
            all_configs.append({
                "label": f"IVFFlat(lists={params.get('lists')})",
                "type": "ivfflat",
                "latency": (get_metric(r, 'pgvector', top_k, ('latency', 'mean')) or 0) * 1000,
                "recall": get_metric(r, 'pgvector', top_k, ('recall', 'recall_mean')) or 0
            })
        
        # Sort by latency
        all_configs.sort(key=lambda x: x["latency"])
        
        labels = [c["label"] for c in all_configs]
        latencies = [c["latency"] for c in all_configs]
        recalls = [c["recall"] for c in all_configs]
        colors = [COLORS['hnsw'] if c["type"] == "hnsw" else COLORS['ivfflat'] for c in all_configs]
        
        x = np.arange(len(labels))
        width = 0.35
        
        # Latency bars
        bars = ax.bar(x, latencies, width, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add recall as text on bars
        for i, (bar, rec) in enumerate(zip(bars, recalls)):
            ax.annotate(f'{rec:.1%}\nrecall', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Index Configuration')
        ax.set_ylabel('Mean Latency (ms)')
        ax.set_title(f'pgvector: HNSW vs IVFFlat Comparison ({dataset}, top_k={top_k})')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['hnsw'], edgecolor='black', label='HNSW'),
            Patch(facecolor=COLORS['ivfflat'], edgecolor='black', label='IVFFlat'),
        ]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"index_type_comparison_{dataset}.png"), dpi=150)
        plt.close()
        print(f"Saved: index_type_comparison_{dataset}.png")


def plot_distance_metric_latency(data):
    runs = [r for r in data["runs"] if r["mode"] == "nofilter"]
    
    # Group by dataset
    by_dataset = {}
    for run in runs:
        ds = run["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(run)
    
    # Find datasets with multiple distance metrics
    multi_metric_datasets = {}
    for ds, runs_list in by_dataset.items():
        metrics = set(r.get("metric", "euclidean") for r in runs_list)
        if len(metrics) > 1:
            multi_metric_datasets[ds] = runs_list
    
    if not multi_metric_datasets:
        print("Skipping distance metric latency: no datasets with multiple distance metrics")
        return
    
    for dataset, runs_list in multi_metric_datasets.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by metric
        by_metric = {}
        for run in runs_list:
            metric = run.get("metric", "euclidean")
            if metric not in by_metric:
                by_metric[metric] = run
        
        metrics = sorted(by_metric.keys())
        x = np.arange(len(metrics))
        width = 0.35
        top_k = 10
        
        chroma_lat = [(get_metric(by_metric[m], 'chroma', top_k, ('latency', 'mean')) or 0) * 1000 for m in metrics]
        pg_lat = [(get_metric(by_metric[m], 'pgvector', top_k, ('latency', 'mean')) or 0) * 1000 for m in metrics]
        
        bars1 = ax.bar(x - width/2, chroma_lat, width, label='Chroma', color=COLORS['chroma'])
        bars2 = ax.bar(x + width/2, pg_lat, width, label='pgvector', color=COLORS['pgvector'])
        
        # Add value labels
        for bar, val in zip(bars1, chroma_lat):
            if val > 0:
                ax.annotate(f'{val:.2f}ms', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars2, pg_lat):
            if val > 0:
                ax.annotate(f'{val:.2f}ms', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Distance Metric')
        ax.set_ylabel('Mean Latency (ms)')
        ax.set_title(f'Distance Metric Impact on Latency ({dataset}, top_k={top_k})')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"distance_metric_latency_{dataset}.png"), dpi=150)
        plt.close()
        print(f"Saved: distance_metric_latency_{dataset}.png")


def plot_distance_metric_recall(data):
    """
    Compare recall across different distance metrics (euclidean vs cosine).
    """
    runs = [r for r in data["runs"] if r["mode"] == "nofilter"]
    
    # Group by dataset
    by_dataset = {}
    for run in runs:
        ds = run["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(run)
    
    # Find datasets with multiple distance metrics
    multi_metric_datasets = {}
    for ds, runs_list in by_dataset.items():
        metrics = set(r.get("metric", "euclidean") for r in runs_list)
        if len(metrics) > 1:
            multi_metric_datasets[ds] = runs_list
    
    if not multi_metric_datasets:
        print("Skipping distance metric recall: no datasets with multiple distance metrics")
        return
    
    for dataset, runs_list in multi_metric_datasets.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by metric
        by_metric = {}
        for run in runs_list:
            metric = run.get("metric", "euclidean")
            if metric not in by_metric:
                by_metric[metric] = run
        
        metrics = sorted(by_metric.keys())
        x = np.arange(len(metrics))
        width = 0.35
        top_k = 10
        
        chroma_rec = [get_metric(by_metric[m], 'chroma', top_k, ('recall', 'recall_mean')) or 0 for m in metrics]
        pg_rec = [get_metric(by_metric[m], 'pgvector', top_k, ('recall', 'recall_mean')) or 0 for m in metrics]
        
        bars1 = ax.bar(x - width/2, chroma_rec, width, label='Chroma', color=COLORS['chroma'])
        bars2 = ax.bar(x + width/2, pg_rec, width, label='pgvector', color=COLORS['pgvector'])
        
        # Add value labels
        for bar, val in zip(bars1, chroma_rec):
            if val > 0:
                ax.annotate(f'{val:.1%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars2, pg_rec):
            if val > 0:
                ax.annotate(f'{val:.1%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Distance Metric')
        ax.set_ylabel('Recall@k')
        ax.set_title(f'Distance Metric Impact on Recall ({dataset}, top_k={top_k})')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"distance_metric_recall_{dataset}.png"), dpi=150)
        plt.close()
        print(f"Saved: distance_metric_recall_{dataset}.png")


def plot_distance_metric_comparison(data):
    runs = [r for r in data["runs"] if r["mode"] == "nofilter"]
    
    # Group by dataset
    by_dataset = {}
    for run in runs:
        ds = run["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(run)
    
    # Find datasets with multiple distance metrics
    multi_metric_datasets = {}
    for ds, runs_list in by_dataset.items():
        metrics = set(r.get("metric", "euclidean") for r in runs_list)
        if len(metrics) > 1:
            multi_metric_datasets[ds] = runs_list
    
    if not multi_metric_datasets:
        print("Skipping distance metric comparison: no datasets with multiple distance metrics")
        return
    
    for dataset, runs_list in multi_metric_datasets.items():
        # Group by metric
        by_metric = {}
        for run in runs_list:
            metric = run.get("metric", "euclidean")
            if metric not in by_metric:
                by_metric[metric] = run
        
        metrics = sorted(by_metric.keys())
        top_k = 10
        
        # Collect data
        data_points = []
        for metric in metrics:
            run = by_metric[metric]
            # Chroma
            c_lat = (get_metric(run, 'chroma', top_k, ('latency', 'mean')) or 0) * 1000
            c_rec = get_metric(run, 'chroma', top_k, ('recall', 'recall_mean')) or 0
            if c_lat > 0:
                data_points.append({
                    'db': 'Chroma',
                    'metric': metric.capitalize(),
                    'latency': c_lat,
                    'recall': c_rec
                })
            # pgvector
            p_lat = (get_metric(run, 'pgvector', top_k, ('latency', 'mean')) or 0) * 1000
            p_rec = get_metric(run, 'pgvector', top_k, ('recall', 'recall_mean')) or 0
            if p_lat > 0:
                data_points.append({
                    'db': 'pgvector',
                    'metric': metric.capitalize(),
                    'latency': p_lat,
                    'recall': p_rec
                })
        
        if not data_points:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot as scatter with annotations
        for dp in data_points:
            color = COLORS['chroma'] if dp['db'] == 'Chroma' else COLORS['pgvector']
            marker = 'o' if dp['db'] == 'Chroma' else 's'
            
            ax.scatter(dp['latency'], dp['recall'], c=color, s=200, marker=marker,
                      edgecolors='black', linewidths=0.5, zorder=3)
            ax.annotate(f"{dp['db']}\n{dp['metric']}", 
                       xy=(dp['latency'], dp['recall']), 
                       xytext=(8, 0), textcoords='offset points',
                       fontsize=9, ha='left', va='center')
        
        ax.set_xlabel('Mean Latency (ms)')
        ax.set_ylabel('Recall@k')
        ax.set_title(f'Distance Metric Impact: Latency vs Recall ({dataset}, top_k={top_k})')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
        ax.grid(alpha=0.3)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['chroma'], 
                   markersize=10, label='Chroma'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['pgvector'], 
                   markersize=10, label='pgvector'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"distance_metric_comparison_{dataset}.png"), dpi=150)
        plt.close()
        print(f"Saved: distance_metric_comparison_{dataset}.png")


def main():
    """Generate all query benchmark plots."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    data = load_results()
    print(f"Loaded {len(data['runs'])} query benchmark runs from {RESULTS_FILE}")
    
    filters = get_available_filters(data)
    print(f"\nAvailable data:")
    print(f"  Index types: {filters['index_types']}")
    print(f"  Dimensions: {filters['dimensions']}")
    print(f"  Sizes: {filters['sizes']}")
    print(f"  Modes: {filters['modes']}")
    print(f"  Datasets: {filters['datasets']}")
    print(f"  Top-K values: {filters['top_ks']}")
    
    print(f"\nGenerating plots...")
    
    plot_latency_comparison(data)
    plot_recall_comparison(data)
    plot_latency_vs_recall(data)
    plot_topk_impact(data)
    plot_filter_impact(data)
    plot_scaling_analysis(data)
    plot_p99_latency(data)
    plot_throughput_comparison(data)
    plot_combined_summary(data)
    plot_index_params_latency(data)
    plot_index_params_recall(data)
    plot_index_params_tradeoff(data)
    plot_index_type_comparison(data)
    plot_distance_metric_latency(data)
    plot_distance_metric_recall(data)
    plot_distance_metric_comparison(data)
    
    # Print summary table
    print_summary_table(data)
    
    print(f"\nAll plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
