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


# =============================================================================
# PLOT FUNCTIONS
# =============================================================================

def plot_latency_comparison(data):
    """
    Plot latency comparison between Chroma and pgvector.
    Side-by-side bars for each dataset, nofilter mode only.
    """
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
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    top_ks = [10, 50, 100]
    
    for ax, top_k in zip(axes, top_ks):
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
                           ha='center', va='bottom', fontsize=7)
        for bar, val in zip(bars2, pg_lat):
            if val > 0:
                ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=7)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Mean Latency (ms)')
        ax.set_title(f'Query Latency (top_k={top_k})')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "latency_comparison.png"), dpi=150)
    plt.close()
    print("Saved: latency_comparison.png")


def plot_recall_comparison(data):
    """
    Plot recall comparison between Chroma and pgvector.
    Side-by-side bars for each dataset, nofilter mode only.
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
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    top_ks = [10, 50, 100]
    
    for ax, top_k in zip(axes, top_ks):
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
                           ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, pg_recall):
            if val > 0:
                ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Recall@k')
        ax.set_title(f'Query Recall (top_k={top_k})')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "recall_comparison.png"), dpi=150)
    plt.close()
    print("Saved: recall_comparison.png")


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
    """
    Line plots showing how top_k affects latency and recall for each dataset.
    """
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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(runs)))
    
    # Chroma latency
    ax1 = axes[0, 0]
    for i, run in enumerate(runs):
        top_ks = [e["top_k"] for e in run["chroma"]]
        lats = [e["latency"]["mean"] * 1000 for e in run["chroma"]]
        ax1.plot(top_ks, lats, 'o-', label=run["dataset"], color=colors[i], linewidth=2, markersize=8)
    ax1.set_xlabel('Top-K')
    ax1.set_ylabel('Mean Latency (ms)')
    ax1.set_title('Chroma: Latency vs Top-K')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # pgvector latency
    ax2 = axes[0, 1]
    for i, run in enumerate(runs):
        top_ks = [e["top_k"] for e in run["pgvector"]]
        lats = [e["latency"]["mean"] * 1000 for e in run["pgvector"]]
        ax2.plot(top_ks, lats, 's-', label=run["dataset"], color=colors[i], linewidth=2, markersize=8)
    ax2.set_xlabel('Top-K')
    ax2.set_ylabel('Mean Latency (ms)')
    ax2.set_title('pgvector: Latency vs Top-K')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Chroma recall
    ax3 = axes[1, 0]
    for i, run in enumerate(runs):
        top_ks = [e["top_k"] for e in run["chroma"]]
        recs = [e.get("recall", {}).get("recall_mean", 0) for e in run["chroma"]]
        ax3.plot(top_ks, recs, 'o-', label=run["dataset"], color=colors[i], linewidth=2, markersize=8)
    ax3.set_xlabel('Top-K')
    ax3.set_ylabel('Recall@k')
    ax3.set_title('Chroma: Recall vs Top-K')
    ax3.set_ylim(0, 1.1)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # pgvector recall
    ax4 = axes[1, 1]
    for i, run in enumerate(runs):
        top_ks = [e["top_k"] for e in run["pgvector"]]
        recs = [e.get("recall", {}).get("recall_mean", 0) for e in run["pgvector"]]
        ax4.plot(top_ks, recs, 's-', label=run["dataset"], color=colors[i], linewidth=2, markersize=8)
    ax4.set_xlabel('Top-K')
    ax4.set_ylabel('Recall@k')
    ax4.set_title('pgvector: Recall vs Top-K')
    ax4.set_ylim(0, 1.1)
    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "topk_impact.png"), dpi=150)
    plt.close()
    print("Saved: topk_impact.png")


def plot_filter_impact(data):
    """
    Compare nofilter vs filter mode for latency and recall.
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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
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
    
    # Plot 1: Latency comparison
    ax1 = axes[0, 0]
    ax1.bar(x - 1.5*width, chroma_nf_lat, width, label='Chroma (nofilter)', color=COLORS['chroma'])
    ax1.bar(x - 0.5*width, chroma_f_lat, width, label='Chroma (filter)', color=COLORS['chroma_dark'])
    ax1.bar(x + 0.5*width, pg_nf_lat, width, label='pgvector (nofilter)', color=COLORS['pgvector'])
    ax1.bar(x + 1.5*width, pg_f_lat, width, label='pgvector (filter)', color=COLORS['pgvector_dark'])
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Mean Latency (ms)')
    ax1.set_title(f'Filter Impact on Latency (top_k={top_k})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets_with_both)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Latency overhead ratio
    ax2 = axes[0, 1]
    chroma_ratio = [f/nf if nf > 0 else 0 for nf, f in zip(chroma_nf_lat, chroma_f_lat)]
    pg_ratio = [f/nf if nf > 0 else 0 for nf, f in zip(pg_nf_lat, pg_f_lat)]
    
    ax2.bar(x - width/2, chroma_ratio, width, label='Chroma', color=COLORS['chroma'])
    ax2.bar(x + width/2, pg_ratio, width, label='pgvector', color=COLORS['pgvector'])
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Latency Ratio (filter / nofilter)')
    ax2.set_title('Filter Overhead (>1 = filter is slower)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets_with_both)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Recall comparison
    ax3 = axes[1, 0]
    ax3.bar(x - 1.5*width, chroma_nf_rec, width, label='Chroma (nofilter)', color=COLORS['chroma'])
    ax3.bar(x - 0.5*width, chroma_f_rec, width, label='Chroma (filter)', color=COLORS['chroma_dark'])
    ax3.bar(x + 0.5*width, pg_nf_rec, width, label='pgvector (nofilter)', color=COLORS['pgvector'])
    ax3.bar(x + 1.5*width, pg_f_rec, width, label='pgvector (filter)', color=COLORS['pgvector_dark'])
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Recall@k')
    ax3.set_title(f'Filter Impact on Recall (top_k={top_k})')
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets_with_both)
    ax3.set_ylim(0, 1.1)
    ax3.legend(fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Recall drop
    ax4 = axes[1, 1]
    chroma_drop = [nf - f for nf, f in zip(chroma_nf_rec, chroma_f_rec)]
    pg_drop = [nf - f for nf, f in zip(pg_nf_rec, pg_f_rec)]
    
    ax4.bar(x - width/2, chroma_drop, width, label='Chroma', color=COLORS['chroma'])
    ax4.bar(x + width/2, pg_drop, width, label='pgvector', color=COLORS['pgvector'])
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Recall Drop (nofilter - filter)')
    ax4.set_title('Recall Degradation from Filtering (>0 = filter hurts recall)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets_with_both)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "filter_impact.png"), dpi=150)
    plt.close()
    print("Saved: filter_impact.png")


def plot_scaling_analysis(data):
    """
    Show how latency and recall scale with dataset size.
    Line plots connecting datasets of increasing size.
    """
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
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    top_k = 10
    
    sizes = [r["vectors"] for r in runs]
    labels = [r["dataset"] for r in runs]
    
    # Latency scaling
    ax1 = axes[0]
    chroma_lat = [(get_metric(r, 'chroma', top_k, ('latency', 'mean')) or 0) * 1000 for r in runs]
    pg_lat = [(get_metric(r, 'pgvector', top_k, ('latency', 'mean')) or 0) * 1000 for r in runs]
    
    ax1.plot(sizes, chroma_lat, 'o-', label='Chroma', color=COLORS['chroma'], linewidth=2, markersize=10)
    ax1.plot(sizes, pg_lat, 's-', label='pgvector', color=COLORS['pgvector'], linewidth=2, markersize=10)
    
    ax1.set_xlabel('Number of Vectors')
    ax1.set_ylabel('Mean Latency (ms)')
    ax1.set_title(f'Latency Scaling (top_k={top_k}, nofilter)')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Add dataset labels
    for i, (s, l) in enumerate(zip(sizes, labels)):
        ax1.annotate(l, (s, max(chroma_lat[i], pg_lat[i])), 
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    
    # Recall scaling
    ax2 = axes[1]
    chroma_rec = [get_metric(r, 'chroma', top_k, ('recall', 'recall_mean')) or 0 for r in runs]
    pg_rec = [get_metric(r, 'pgvector', top_k, ('recall', 'recall_mean')) or 0 for r in runs]
    
    ax2.plot(sizes, chroma_rec, 'o-', label='Chroma', color=COLORS['chroma'], linewidth=2, markersize=10)
    ax2.plot(sizes, pg_rec, 's-', label='pgvector', color=COLORS['pgvector'], linewidth=2, markersize=10)
    
    ax2.set_xlabel('Number of Vectors')
    ax2.set_ylabel('Recall@k')
    ax2.set_title(f'Recall Scaling (top_k={top_k}, nofilter)')
    ax2.set_xscale('log')
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "scaling_analysis.png"), dpi=150)
    plt.close()
    print("Saved: scaling_analysis.png")


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
    """
    Create a 2x2 summary plot with key metrics.
    """
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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(datasets))
    width = 0.35
    
    # Plot 1: Mean latency
    ax1 = axes[0, 0]
    chroma_lat = [(get_metric(r, 'chroma', top_k, ('latency', 'mean')) or 0) * 1000 for r in runs]
    pg_lat = [(get_metric(r, 'pgvector', top_k, ('latency', 'mean')) or 0) * 1000 for r in runs]
    
    ax1.bar(x - width/2, chroma_lat, width, label='Chroma', color=COLORS['chroma'])
    ax1.bar(x + width/2, pg_lat, width, label='pgvector', color=COLORS['pgvector'])
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Mean Latency (ms)')
    ax1.set_title(f'Query Latency (top_k={top_k})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Recall
    ax2 = axes[0, 1]
    chroma_rec = [get_metric(r, 'chroma', top_k, ('recall', 'recall_mean')) or 0 for r in runs]
    pg_rec = [get_metric(r, 'pgvector', top_k, ('recall', 'recall_mean')) or 0 for r in runs]
    
    ax2.bar(x - width/2, chroma_rec, width, label='Chroma', color=COLORS['chroma'])
    ax2.bar(x + width/2, pg_rec, width, label='pgvector', color=COLORS['pgvector'])
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Recall@k')
    ax2.set_title(f'Query Recall (top_k={top_k})')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Throughput
    ax3 = axes[1, 0]
    chroma_qps = [1 / (get_metric(r, 'chroma', top_k, ('latency', 'mean')) or 1) for r in runs]
    pg_qps = [1 / (get_metric(r, 'pgvector', top_k, ('latency', 'mean')) or 1) for r in runs]
    
    ax3.bar(x - width/2, chroma_qps, width, label='Chroma', color=COLORS['chroma'])
    ax3.bar(x + width/2, pg_qps, width, label='pgvector', color=COLORS['pgvector'])
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Queries per Second')
    ax3.set_title(f'Query Throughput (top_k={top_k})')
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Speedup ratio
    ax4 = axes[1, 1]
    speedup = [c / p if p > 0 else 0 for c, p in zip(chroma_lat, pg_lat)]
    colors_list = [COLORS['pgvector'] if s > 1 else COLORS['chroma'] for s in speedup]
    
    bars = ax4.bar(x, speedup, width*1.5, color=colors_list)
    ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Latency Ratio (Chroma / pgvector)')
    ax4.set_title('Relative Speed (>1 = pgvector faster)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, speedup):
        label = f'{val:.1f}x'
        ax4.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "combined_summary.png"), dpi=150)
    plt.close()
    print("Saved: combined_summary.png")


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
    
    # Generate all plots
    plot_latency_comparison(data)
    plot_recall_comparison(data)
    plot_latency_vs_recall(data)
    plot_topk_impact(data)
    plot_filter_impact(data)
    plot_scaling_analysis(data)
    plot_p99_latency(data)
    plot_throughput_comparison(data)
    plot_combined_summary(data)
    
    # Print summary table
    print_summary_table(data)
    
    print(f"\nAll plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
