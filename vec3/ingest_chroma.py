import os
import time
import json
import numpy as np
import chromadb

def ingest_chroma(dataset_dir: str, collection_name: str, batch_size: int = 1000):
    client = chromadb.HttpClient(host="localhost", port=8000)
    collection = client.get_or_create_collection(collection_name)

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
        batch_ids = [str(idx + i) for i in range(len(batch_vectors))]

        collection.add(
            ids=batch_ids,
            embeddings=batch_vectors.tolist(),
            metadatas=batch_meta,
        )

        idx += batch_size

    duration = time.perf_counter() - start
    vps = n / duration if duration > 0 else 0.0

    return {
        "db": "chroma",
        "collection": collection_name,
        "dataset_dir": dataset_dir,
        "vectors": n,
        "duration_sec": duration,
        "vectors_per_sec": vps,
    }
