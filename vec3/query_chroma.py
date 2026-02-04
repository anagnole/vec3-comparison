import chromadb
import time
from vec3.metrics import latency_stats

def run_queries(dataset_queries, collection_name: str, n_results: int = 10,
                return_ids: bool = False):
    client = chromadb.HttpClient(host="localhost", port=8000)

    try:
        col = client.get_collection(collection_name)
    except Exception as e:
        print(f"Error: Collection '{collection_name}' not found locally.")
        print("Make sure you ran the ingestion script first locally.")
        raise e

    latencies = []
    all_retrieved_ids = [] if return_ids else None

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
        result = col.query(**query_args)
        latencies.append(time.perf_counter() - start)
        
        if return_ids:
            ids = result.get("ids", [[]])[0]
            all_retrieved_ids.append([int(id_str) for id_str in ids])

    stats = latency_stats(latencies)
    if return_ids:
        return stats, all_retrieved_ids
    return stats
