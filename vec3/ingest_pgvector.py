import os
import time
import json
import numpy as np
import psycopg2
from psycopg2 import sql

def get_db_connection():
    return psycopg2.connect(
        dbname="vecdb",
        user="user",
        host="localhost",
        port="5432"
    )

def ingest_pgvector(
    dataset_dir: str,
    batch_size: int = 1000,
    lists: int = 100,
    create_index: bool = True,
    table_name: str = "vectors",
):
    conn = get_db_connection()
    cur = conn.cursor()

    print(f"→ Resetting table '{table_name}'...")
    cur.execute(sql.SQL("TRUNCATE TABLE {}").format(sql.Identifier(table_name)))    
    conn.commit()

    vectors_path = os.path.join(dataset_dir, "vectors.npy")
    metadata_path = os.path.join(dataset_dir, "metadata.jsonl")

    vectors = np.load(vectors_path)
    metadata = [json.loads(line) for line in open(metadata_path)]

    n = len(vectors)
    print(f"→ Loading {n} vectors from {dataset_dir} into table {table_name} ...")

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
            sql.SQL("INSERT INTO {} (embedding, cls) VALUES (%s, %s)")
               .format(sql.Identifier(table_name)),
            args,
        )

        idx += batch_size
        if idx % (batch_size * 10) == 0:
            print(f"  inserted {idx}/{n}...")

    conn.commit()
    duration_ingest = time.perf_counter() - start
    vps_ingest = n / duration_ingest

    print(f"✓ Insert phase completed in {duration_ingest:.2f}s ({vps_ingest:.2f} v/s)")

    # INDEX CREATION
    index_time = None

    if create_index:
        print("→ Creating IVFFLAT index...")

        #make index name unique per table to avoid collisions
        index_name = f"{table_name}_embedding_idx"

        # Drop old index
        cur.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(index_name)))
        conn.commit()

        start_idx = time.perf_counter()

        cur.execute(
            sql.SQL("""
                CREATE INDEX {}
                ON {}
                USING ivfflat (embedding vector_l2_ops)
                WITH (lists = %s);
            """).format(
                sql.Identifier(index_name),
                sql.Identifier(table_name),
            ),
            (lists,)
        )
        conn.commit()

        # Required or PG will ignore the index
        cur.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(table_name)))
        conn.commit()

        index_time = time.perf_counter() - start_idx
        print(f"✓ Index built in {index_time:.2f}s")

    cur.close()
    conn.close()

    return {
        "db": "pgvector",
        "table": table_name,
        "dataset_dir": dataset_dir,
        "vectors": n,
        "duration_sec": duration_ingest + (index_time or 0),
        "duration_ingest_sec": duration_ingest,
        "duration_index_sec": index_time,
        "vectors_per_sec": vps_ingest,
    }