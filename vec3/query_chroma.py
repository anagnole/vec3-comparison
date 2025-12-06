import chromadb
import time
from vec3.metrics import latency_stats

def run_queries(dataset_queries, collection_name: str):
    client = chromadb.HttpClient(host="localhost", port=8000)
    try:
        col = client.get_collection(collection_name)
    except Exception as e:
        print(f"Error: Collection '{collection_name}' not found locally.")
        print("Make sure you ran the ingestion script first locally.")
        raise e

    latencies = []

    for q in dataset_queries:
        start = time.perf_counter()
        
        query_args = {
            "query_embeddings": [q["vector"]],
            "n_results": 10
        }
        
        if "filter" in q and q["filter"]:
            query_args["where"] = q["filter"]

        col.query(**query_args)
        latencies.append(time.perf_counter() - start)

    return latency_stats(latencies)
