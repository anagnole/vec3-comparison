import chromadb
import numpy as np
import json
import time

def load_chroma(dataset_path: str, collection_name: str):
    client = chromadb.Client()
    col = client.get_or_create_collection(collection_name)

    vectors = np.load(f"{dataset_path}/vectors.npy")
    metadata = [json.loads(line) for line in open(f"{dataset_path}/metadata.jsonl")]

    start = time.perf_counter()

    batch = 500
    for i in range(0, len(vectors), batch):
        v = vectors[i:i+batch].tolist()
        m = metadata[i:i+batch]
        ids = [str(i+j) for j in range(len(v))]
        col.add(ids=ids, embeddings=v, metadatas=m)

    duration = time.perf_counter() - start
    print(f"Chroma ingest: {duration:.2f}s")
    return duration
