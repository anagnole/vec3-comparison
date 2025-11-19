import os
import time
import json
import numpy as np
import psycopg2

def ingest_pgvector(dataset_dir: str, batch_size: int = 1000, dsn: str = "dbname=vecdb"):
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    cur.execute("TRUNCATE TABLE vectors;")
    conn.commit()

    vectors_path = os.path.join(dataset_dir, "vectors.npy")
    metadata_path = os.path.join(dataset_dir, "metadata.jsonl")

    vectors = np.load(vectors_path)
    metadata = [json.loads(line) for line in open(metadata_path)]

    n = len(vectors)
    start = time.perf_counter()

    idx = 0
    while idx < n:
        batch_vectors = vectors[idx:idx + batch_size]
        batch_meta = metadata[idx:idx + batch_size]

        args = [
            (v.tolist(), m["cls"])
            for v, m in zip(batch_vectors, batch_meta)
        ]

        cur.executemany(
            "INSERT INTO vectors (embedding, cls) VALUES (%s, %s)",
            args,
        )

        idx += batch_size

    conn.commit()
    duration = time.perf_counter() - start
    vps = n / duration if duration > 0 else 0.0

    cur.close()
    conn.close()

    return {
        "db": "pgvector",
        "dataset_dir": dataset_dir,
        "vectors": n,
        "duration_sec": duration,
        "vectors_per_sec": vps,
    }
