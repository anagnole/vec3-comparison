import time
import psycopg2
from psycopg2 import sql
from vec3.metrics import latency_stats

METRIC_OPERATORS = {
    "euclidean": "<->",      # L2 distance
    "cosine": "<=>",         # Cosine distance
    "inner_product": "<#>",  # Inner product (negative, so smaller = more similar)
}

def get_db_connection():
    return psycopg2.connect(
        dbname="vecdb",
        user="user", 
        host="localhost",
        port="5432"
    )

def run_queries(dataset_queries, table_name="vectors", n_results: int = 10, 
                metric: str = "euclidean", return_ids: bool = False):
    conn = get_db_connection()
    cur = conn.cursor()
    latencies = []
    all_retrieved_ids = [] if return_ids else None
    
    operator = METRIC_OPERATORS.get(metric, "<->")

    q_no_filter = sql.SQL("""
        SELECT id, embedding {op} %s::vector AS distance
        FROM {table}
        ORDER BY distance
        LIMIT %s;
    """).format(table=sql.Identifier(table_name), op=sql.SQL(operator))

    q_with_filter = sql.SQL("""
        SELECT id, embedding {op} %s::vector AS distance
        FROM {table}
        WHERE cls = %s
        ORDER BY distance
        LIMIT %s;
    """).format(table=sql.Identifier(table_name), op=sql.SQL(operator))

    # warmup (not timed)
    first = dataset_queries[0]
    if first.get("filter"):
        cur.execute(q_with_filter, (first["vector"], first["filter"]["cls"], n_results))
    else:
        cur.execute(q_no_filter, (first["vector"], n_results))
    cur.fetchall()

    for q in dataset_queries:
        vector = q["vector"]
        start = time.perf_counter()

        if "filter" in q and q["filter"]:
            filter_cls = q["filter"].get("cls")
            cur.execute(q_with_filter, (vector, filter_cls, n_results))
        else:
            cur.execute(q_no_filter, (vector, n_results))

        results = cur.fetchall()
        latencies.append(time.perf_counter() - start)
        
        if return_ids:
            all_retrieved_ids.append([row[0] for row in results])

    cur.close()
    conn.close()
    
    stats = latency_stats(latencies)
    if return_ids:
        return stats, all_retrieved_ids
    return stats
