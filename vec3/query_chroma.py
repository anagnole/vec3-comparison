import chromadb
import time
from vec3.metrics import latency_stats

def run_queries(dataset_queries, collection_name: str):
    client = chromadb.Client()
    col = client.get_or_create_collection(collection_name)

    latencies = []

    for q in dataset_queries:
        start = time.perf_counter()
        col.query(
            query_embeddings=[q["vector"]],
            n_results=10,
            where=q.get("filter")
        )
        latencies.append(time.perf_counter() - start)

    return latency_stats(latencies)
