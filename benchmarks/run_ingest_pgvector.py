import os
from vec3.ingest_pgvector import ingest_pgvector

DATASETS = [
    ("data/10k", "vec3_10k"),
    ("data/100k", "vec3_100k"),
    ("data/200k", "vec3_200k"),
    ("data/500k", "vec3_500k"),
]

BATCH_SIZE = 1000

def main():
    results = []

    for dataset_dir, _collection_name in DATASETS:
        ds_label = os.path.basename(dataset_dir)
        table_name = f"vectors_{ds_label}"

        r_pg = ingest_pgvector(dataset_dir, batch_size=BATCH_SIZE, table_name=table_name, create_index=True)
        results.append(r_pg)

    for r in results:
        print(
            f"{r['db']} | dataset={r['dataset_dir']} | vectors={r['vectors']} | "
            f"time={r['duration_sec']:.2f}s | vps={r['vectors_per_sec']:.2f} | table={r.get('table')}"
        )

if __name__ == "__main__":
    main()
