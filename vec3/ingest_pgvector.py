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


def get_table_size_bytes(cur, table_name="vectors"):
    cur.execute(f"SELECT pg_total_relation_size('{table_name}');")
    return cur.fetchone()[0]


def ingest_pgvector(
    dataset_dir: str,
    batch_size: int = 1000,
    lists: int = 100,
    create_index: bool = True,
    table_name: str = "vectors",
    index_type: str = "ivfflat",
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 64,
):
    conn = get_db_connection()
    cur = conn.cursor()

    vectors_path = os.path.join(dataset_dir, "vectors.npy")
    metadata_path = os.path.join(dataset_dir, "metadata.jsonl")

    vectors = np.load(vectors_path)
    metadata = [json.loads(line) for line in open(metadata_path)]

    n = len(vectors)
    dim = vectors.shape[1]

    print(f"→ Creating vector extension and recreating table with dimension {dim}...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(table_name)))
    create_sql = f"""
        CREATE TABLE "{table_name}" (
            id SERIAL PRIMARY KEY,
            embedding vector({dim}),
            cls TEXT
        );
    """
    cur.execute(create_sql)
    conn.commit()

    print(f"→ Loading {n} vectors (dim={dim}) into pgvector...")

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

    storage_before_index = get_table_size_bytes(cur, table_name)

    index_time = None
    index_name = f"{table_name}_embedding_idx"
    index_info = "none"

    if create_index and index_type != "none":
        cur.execute(sql.SQL("DROP INDEX IF EXISTS {};").format(sql.Identifier(index_name)))
        conn.commit()

        start_idx = time.perf_counter()

        if index_type == "hnsw":
            print(f"→ Creating HNSW index (m={hnsw_m}, ef_construction={hnsw_ef_construction})...")
            cur.execute(
                sql.SQL("""
                    CREATE INDEX {}
                    ON {}
                    USING hnsw (embedding vector_l2_ops)
                    WITH (m = %s, ef_construction = %s);
                """).format(
                    sql.Identifier(index_name),
                    sql.Identifier(table_name),
                ),
                (hnsw_m, hnsw_ef_construction)
            )
            index_info = f"hnsw(m={hnsw_m},ef={hnsw_ef_construction})"
        else:
            print(f"→ Creating IVFFlat index (lists={lists})...")
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
            index_info = f"ivfflat(lists={lists})"

        conn.commit()

        cur.execute(sql.SQL("ANALYZE {};").format(sql.Identifier(table_name)))
        conn.commit()

        index_time = time.perf_counter() - start_idx
        print(f"✓ Index built in {index_time:.2f}s")

    storage_total = get_table_size_bytes(cur, table_name)
    print(f"  Storage used: {storage_total / (1024*1024):.2f} MB (table+index)")

    cur.close()
    conn.close()

    return {
        "db": "pgvector",
        "table": table_name,
        "dataset_dir": dataset_dir,
        "vectors": n,
        "dimensions": dim,
        "batch_size": batch_size,
        "index_type": index_info,
        "duration_sec": duration_ingest + (index_time or 0),
        "duration_ingest_sec": duration_ingest,
        "duration_index_sec": index_time,
        "vectors_per_sec": vps_ingest,
        "storage_bytes": storage_total,
        "storage_before_index_bytes": storage_before_index,
    }