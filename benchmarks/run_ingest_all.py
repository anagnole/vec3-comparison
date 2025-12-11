from vec3.ingest_chroma import ingest_chroma
from vec3.ingest_pgvector import ingest_pgvector
from vec3.plotting import plot_ingest_bars
import os
import json


DATASETS = [
    # ("data/10k", "vec3_10k"),
    # ("data/100k", "vec3_100k"),
    # ("data/500k", "vec3_500k"),
    ("data/200k", "vec3_200k"),
]


def run_all_ingestion_batches(datasets,plot_title, batch_size=1000):
    """
    Run ingestion benchmarks for all datasets and return raw results.
    Handles:
      - Chroma ingestion
      - Pgvector ingestion
      - Plot creation
      - Raw JSON save
    """

    results = []
    total = len(datasets)

    # Ingestion loop
    for idx, (dataset_dir, collection_name) in enumerate(datasets, start=1):

        print(f"[ {idx}/{total} ] Ingesting Chroma for '{dataset_dir}'...")
        r_chroma = ingest_chroma(
            dataset_dir,
            collection_name=collection_name,
            batch_size=batch_size
        )
        results.append(r_chroma)

        print(f"[ {idx}/{total} ] Ingesting Pgvector for '{dataset_dir}'...")
        r_pg = ingest_pgvector(
            dataset_dir,
            batch_size=batch_size,
            create_index=True,
        )
        results.append(r_pg)

    # Print summary to console
    for r in results:
        print(
            f"{r['db']} | dataset={r['dataset_dir']} | vectors={r['vectors']} | "
            f"time={r['duration_sec']:.2f}s | vps={r['vectors_per_sec']:.2f}"
        )

    if not results:
        print("No results – nothing to plot.")
        return results

    # Prepare data for plotting
    plot_results = []
    for r in results:
        label = os.path.basename(r["dataset_dir"])
        plot_results.append({
            "db": r["db"],
            "dataset": label,
            "duration_sec": r["duration_sec"],
            "vectors_per_sec": r["vectors_per_sec"]
        })

    # Folder setup
    plots_dir = os.path.join("results", "plots")
    raw_dir = os.path.join("results", "raw")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # Plot ingestion time
    time_path = os.path.join(plots_dir, f"{plot_title}_ingest_time_chroma_vs_pgvector.png")
    plot_ingest_bars(
        plot_results,
        metric="duration_sec",
        title=f"{plot_title} - Ingestion time by dataset size",
        output_path=time_path,
    )
    print(f"Saved ingestion time plot -> {time_path}")

    # Plot throughput
    vps_path = os.path.join(plots_dir, f"{plot_title}_ingest_throughput_chroma_vs_pgvector.png")
    plot_ingest_bars(
        plot_results,
        metric="vectors_per_sec",
        title=f"{plot_title} - Ingestion throughput (vectors/sec) by dataset size",
        output_path=vps_path,
    )
    print(f"Saved ingestion throughput plot -> {vps_path}")

    # Save raw results JSON
    raw_path = os.path.join(raw_dir, f"{plot_title}_ingest_results.json")
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved ingestion results -> {raw_path}")

    return results


def main():
    # `batch_size` can be changed here or passed as CLI params later.
    run_all_ingestion_batches(DATASETS, "benchmark_run_small_batches_200", batch_size=500)
    run_all_ingestion_batches(DATASETS, "benchmark_run_200", batch_size=1000)
    run_all_ingestion_batches(DATASETS, "benchmark_run_large_batches_200", batch_size=5000)


if __name__ == "__main__":
    main()
