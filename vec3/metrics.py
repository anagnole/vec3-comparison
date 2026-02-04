import numpy as np

def latency_stats(latencies):
    arr = np.array(latencies)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def compute_recall(retrieved_ids: list, ground_truth_ids: list) -> float:
    if not ground_truth_ids:
        return 1.0
    retrieved_set = set(retrieved_ids)
    gt_set = set(ground_truth_ids[:len(retrieved_ids)])  # Compare same k
    return len(retrieved_set & gt_set) / len(gt_set)


def compute_ground_truth(query_vectors: np.ndarray, all_vectors: np.ndarray, 
                         k: int, metric: str = "euclidean") -> list:
    ground_truth = []
    
    for query in query_vectors:
        if metric == "euclidean":
            # L2 distance
            distances = np.linalg.norm(all_vectors - query, axis=1)
            indices = np.argsort(distances)[:k]
        elif metric == "cosine":
            # Cosine similarity (higher is better, so negate for argsort)
            query_norm = query / (np.linalg.norm(query) + 1e-10)
            all_norms = all_vectors / (np.linalg.norm(all_vectors, axis=1, keepdims=True) + 1e-10)
            similarities = np.dot(all_norms, query_norm)
            indices = np.argsort(-similarities)[:k]  # Descending
        elif metric == "inner_product":
            # Inner product (higher is better)
            products = np.dot(all_vectors, query)
            indices = np.argsort(-products)[:k]  # Descending
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        ground_truth.append(indices.tolist())
    
    return ground_truth
