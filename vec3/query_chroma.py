import chromadb
import time
from vec3.metrics import latency_stats

def run_queries(dataset_queries, collection_name: str, n_results: int = 10):
    client = chromadb.HttpClient(host="localhost", port=8000)


    try:
        col = client.get_collection(collection_name)
    except Exception as e:
        print(f"Error: Collection '{collection_name}' not found locally.")
        print("Make sure you ran the ingestion script first locally.")
        raise e

    latencies = []

    # warmup (not timed)
    first = dataset_queries[0]
    warm_args = {"query_embeddings": [first["vector"]], "n_results": n_results}
    if first.get("filter"):
        warm_args["where"] = first["filter"]
    col.query(**warm_args)

    for q in dataset_queries:
        query_args = {
            "query_embeddings": [q["vector"]],
            "n_results": n_results,
        }
        
        if q.get("filter"):
            query_args["where"] = q["filter"]

        start = time.perf_counter()
        col.query(**query_args)
        latencies.append(time.perf_counter() - start)

    return latency_stats(latencies)
