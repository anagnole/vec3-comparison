import psycopg2
import numpy as np
import json
import time

def load_pgvector(dataset_path: str):
    conn = psycopg2.connect("dbname=vecdb user=postgres password=postgres")
    cur = conn.cursor()

    vectors = np.load(f"{dataset_path}/vectors.npy")
    metadata = [json.loads(line) for line in open(f"{dataset_path}/metadata.jsonl")]

    start = time.perf_counter()

    for vec, meta in zip(vectors, metadata):
        cur.execute(
            "INSERT INTO vectors (embedding, cls) VALUES (%s, %s)",
            (vec.tolist(), meta["cls"])
        )

    conn.commit()
    duration = time.perf_counter() - start

    print(f"pgvector ingest: {duration:.2f}s")
    return duration
