import time
import psycopg2
from vec3.metrics import latency_stats

def get_db_connection():
    return psycopg2.connect(
        dbname="vecdb",
        user="user", 
        host="localhost",
        port="5432"
    )

def run_queries(dataset_queries, table_name="vectors"):
    conn = psycopg2.connect("dbname=vecdb")
    cur = conn.cursor()
    latencies = []

    for q in dataset_queries:
        vector = str(q["vector"])
        start = time.perf_counter()
        
        if "filter" in q and q["filter"]:
            filter_cls = q["filter"].get("cls")
            cur.execute(f"""
                SELECT id, embedding <-> %s::vector AS distance
                FROM {table_name}
                WHERE cls = %s
                ORDER BY distance LIMIT 10;
            """, (vector, filter_cls))
        else:
            cur.execute(f"""
                SELECT id, embedding <-> %s::vector AS distance
                FROM {table_name}
                ORDER BY distance LIMIT 10;
            """, (vector,))
            
        _ = cur.fetchall()
        latencies.append(time.perf_counter() - start)

    cur.close()
    conn.close()
    return latency_stats(latencies)
