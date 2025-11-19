from vec3.ingest_chroma import ingest_chroma
from vec3.ingest_pgvector import ingest_pgvector

DATASETS = [
    ("data/10k", "vec3_10k"),
    ("data/100k", "vec3_100k"),
    ("data/500k", "vec3_500k"),
]

def main():
    results = []

    for dataset_dir, collection_name in DATASETS:
        r_chroma = ingest_chroma(dataset_dir, collection_name=collection_name, batch_size=1000)
        results.append(r_chroma)

        r_pg = ingest_pgvector(dataset_dir, batch_size=1000)
        results.append(r_pg)

    for r in results:
        print(
            f"{r['db']} | dataset={r['dataset_dir']} | vectors={r['vectors']} | "
            f"time={r['duration_sec']:.2f}s | vps={r['vectors_per_sec']:.2f}"
        )

if __name__ == "__main__":
    main()
