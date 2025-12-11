import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Load combined file
combined_path = "./results/raw/all_ingestion_results_combined.json"
with open(combined_path) as f:
    combined = json.load(f)

batches = {
    500: combined["batch_500"],
    1000: combined["batch_1000"],
    5000: combined["batch_5000"],
}

datasets = ["10k", "100k", "200k", "500k"]
plot_dir = "./results/plots/plots_pgvector_stacked"
os.makedirs(plot_dir, exist_ok=True)

def find_entry(data, db, label):
    for r in data:
        if r["db"] == db and r["dataset_dir"].endswith(label):
            return r
    return None

# -------------------------------------------------------
# 1) Stacked pgvector bars + chroma bars for each batch size
# -------------------------------------------------------
for batch_size, data in batches.items():

    x = np.arange(len(datasets))
    width = 0.35

    chroma_times = []
    pg_ingest_times = []
    pg_index_times = []

    for label in datasets:
        c = find_entry(data, "chroma", label)
        p = find_entry(data, "pgvector", label)

        chroma_times.append(c["duration_sec"] if c else np.nan)

        if p:
            ingest_time = p.get("duration_ingest_sec", p["duration_sec"])
            index_time = p.get("duration_index_sec", 0)
        else:
            ingest_time = index_time = np.nan

        pg_ingest_times.append(ingest_time)
        pg_index_times.append(index_time)

    # PLOT
    plt.figure(figsize=(10, 6))

    # Chroma = single bars
    plt.bar(
        x - width/2,
        chroma_times,
        width,
        label="Chroma (total)",
        edgecolor="black"
    )

    # Pgvector = stacked bar
    plt.bar(
        x + width/2,
        pg_ingest_times,
        width,
        label="pgvector ingest",
        color="#4C72B0"
    )
    plt.bar(
        x + width/2,
        pg_index_times,
        width,
        bottom=pg_ingest_times,
        label="pgvector index build",
        color="#DD8452"
    )

    plt.xticks(x, datasets)
    plt.ylabel("Time (s)")
    plt.title(f"Ingestion time (stacked pgvector) — batch={batch_size}")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(plot_dir, f"stacked_pgvector_batch_{batch_size}.png")
    plt.savefig(out_path)
    plt.close()

# -------------------------------------------------------
# 2) Batch-size effect per DB (stacked pgvector)
# -------------------------------------------------------
dbs = ["chroma", "pgvector"]

for db in dbs:

    for metric in ["duration_sec", "vectors_per_sec"]:
        plt.figure(figsize=(10, 6))
        x = np.arange(len(datasets))
        width = 0.25

        for i, (batch_size, data) in enumerate(sorted(batches.items())):
            if db == "chroma":
                vals = [find_entry(data, "chroma", d)["duration_sec"] for d in datasets]
                plt.bar(x + i*width, vals, width, label=f"batch={batch_size}")

            else:
                # stacked pgvector
                ingest_vals = []
                index_vals = []
                for label in datasets:
                    p = find_entry(data, "pgvector", label)
                    if p:
                        ingest_vals.append(p.get("duration_ingest_sec", p["duration_sec"]))
                        index_vals.append(p.get("duration_index_sec", 0))
                    else:
                        ingest_vals.append(np.nan)
                        index_vals.append(np.nan)

                plt.bar(x + i*width, ingest_vals, width,
                        label=f"pg ingest (batch={batch_size})",
                        color="#4C72B0")
                plt.bar(x + i*width, index_vals, width,
                        bottom=ingest_vals,
                        label=f"pg index (batch={batch_size})",
                        color="#DD8452")

        plt.xticks(x + width, datasets)
        ylabel = "Time (s)" if metric == "duration_sec" else "Vectors per sec"
        plt.ylabel(ylabel)
        plt.title(f"{db} vs batch size ({ylabel})")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(
            plot_dir, f"{db}_batch_effect_stacked_{'time' if metric=='duration_sec' else 'vps'}.png"
        )
        plt.savefig(out_path)
        plt.close()