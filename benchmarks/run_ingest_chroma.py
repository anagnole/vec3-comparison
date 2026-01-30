from vec3.ingest_chroma import ingest_chroma

DATASETS = [
    ("data/10k", "vec3_10k"),
    ("data/100k", "vec3_100k"),
    ("data/200k", "vec3_200k"),
    ("data/500k", "vec3_500k"),
]

BATCH_SIZE = 1000

def main():
    results = []

    for dataset_dir, collection_name in DATASETS:
        chroma_collection = f"{collection_name}_b{BATCH_SIZE}"
        r_chroma = ingest_chroma(dataset_dir, collection_name=chroma_collection, batch_size=BATCH_SIZE)
        results.append(r_chroma)

    for r in results:
        print(
            f"{r['db']} | dataset={r['dataset_dir']} | vectors={r['vectors']} | "
            f"time={r['duration_sec']:.2f}s | vps={r['vectors_per_sec']:.2f}"
        )

if __name__ == "__main__":
    main()
