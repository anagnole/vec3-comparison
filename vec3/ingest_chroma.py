import os
import time
import json
import subprocess
import numpy as np
import chromadb


def get_chroma_storage_bytes():
    try:
        result = subprocess.run(
            ["docker", "exec", "chroma_bench", "du", "-sb", "/data"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            return int(result.stdout.split()[0])
    except Exception as e:
        print(f"  Warning: Could not get Chroma storage: {e}")
    return None


def ingest_chroma(dataset_dir: str, collection_name: str, batch_size: int = 1000,
                  hnsw_m: int = None, hnsw_ef_construction: int = None, hnsw_ef_search: int = None):
    client = chromadb.HttpClient(host="localhost", port=8000)

    for col in client.list_collections():
        if col.name == collection_name:
            client.delete_collection(collection_name)
            break

    time.sleep(0.5)
    storage_before = get_chroma_storage_bytes()
    print(f"  Storage before: {storage_before} bytes" if storage_before else "  Storage before: N/A")

    # Build HNSW metadata if parameters provided
    metadata = {}
    if hnsw_m is not None:
        metadata["hnsw:M"] = hnsw_m
    if hnsw_ef_construction is not None:
        metadata["hnsw:construction_ef"] = hnsw_ef_construction
    if hnsw_ef_search is not None:
        metadata["hnsw:search_ef"] = hnsw_ef_search
    
    if metadata:
        print(f"  HNSW params: {metadata}")
        collection = client.create_collection(collection_name, metadata=metadata)
    else:
        collection = client.create_collection(collection_name)

    vectors_path = os.path.join(dataset_dir, "vectors.npy")
    metadata_path = os.path.join(dataset_dir, "metadata.jsonl")

    vectors = np.load(vectors_path)
    with open(metadata_path, "r") as f:
        metadata = [json.loads(line) for line in f]

    n = len(vectors)
    dim = vectors.shape[1]
    print(f"→ Loading {n} vectors (dim={dim}) into Chroma collection '{collection_name}'...")

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
        if idx % (batch_size * 10) == 0:
            print(f"  inserted {idx}/{n}...")

    duration = time.perf_counter() - start
    vps = n / duration if duration > 0 else 0.0

    time.sleep(1)
    storage_after = get_chroma_storage_bytes()
    print(f"  Storage after: {storage_after} bytes" if storage_after else "  Storage after: N/A")
    
    storage_bytes = None
    if storage_before is not None and storage_after is not None:
        storage_bytes = storage_after - storage_before

    print(f"✓ Chroma ingestion completed in {duration:.2f}s ({vps:.2f} v/s)")
    if storage_bytes and storage_bytes > 0:
        print(f"  Storage used: {storage_bytes / (1024*1024):.2f} MB")

    return {
        "db": "chroma",
        "collection": collection_name,
        "dataset_dir": dataset_dir,
        "vectors": n,
        "dimensions": dim,
        "batch_size": batch_size,
        "index_type": "HNSW",
        "duration_sec": duration,
        "vectors_per_sec": vps,
        "storage_bytes": storage_bytes,
    }
