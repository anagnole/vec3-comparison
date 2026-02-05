import os
import json
import numpy as np
from typing import List, Dict, Tuple


class DatasetGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(self.seed)

    def generate_vectors(self, size: int, dim: int, distribution: str = "gaussian") -> np.ndarray:
        if distribution == "gaussian":
            return np.random.randn(size, dim).astype(np.float32)
        elif distribution == "uniform":
            return np.random.uniform(-1, 1, size=(size, dim)).astype(np.float32)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    def generate_metadata(self, size: int, classes: List[str], class_distribution: List[float]) -> List[Dict]:
        class_distribution = np.array(class_distribution)
        class_distribution = class_distribution / class_distribution.sum()  # normalize

        sampled = np.random.choice(classes, size=size, p=class_distribution)
        return [{"cls": c} for c in sampled]

    def save_vectors(self, vectors: np.ndarray, out_dir: str):
        path = os.path.join(out_dir, "vectors.npy")
        np.save(path, vectors)

    def save_metadata(self, metadata: List[Dict], out_dir: str):
        path = os.path.join(out_dir, "metadata.jsonl")
        with open(path, "w") as f:
            for entry in metadata:
                f.write(json.dumps(entry) + "\n")

    def generate_dataset(self,
                         size: int,
                         dim: int,
                         out_dir: str,
                         distribution: str = "gaussian",
                         classes: List[str] = ["A", "B", "C"],
                         class_distribution: List[float] = [0.1, 0.3, 0.6]):

        os.makedirs(out_dir, exist_ok=True)

        vectors = self.generate_vectors(size=size, dim=dim, distribution=distribution)
        metadata = self.generate_metadata(size=size, classes=classes, class_distribution=class_distribution)

        self.save_vectors(vectors, out_dir)
        self.save_metadata(metadata, out_dir)

        print(f"Generated dataset at {out_dir}: size={size}, dim={dim}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic vector datasets.")

    parser.add_argument("--size", type=int, required=True, help="Number of vectors to generate")
    parser.add_argument("--dim", type=int, required=True, help="Dimensionality of vectors")
    parser.add_argument("--out", type=str, required=True, help="Output directory")

    parser.add_argument("--distribution", type=str, default="gaussian",
                        choices=["gaussian", "uniform"],
                        help="Vector distribution type")

    parser.add_argument("--classes", nargs="+", default=["A", "B", "C"],
                        help="List of class labels")

    parser.add_argument("--class-dist", nargs="+", type=float, default=[0.1, 0.3, 0.6],
                        help="Class distribution percentages (must match number of classes)")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Initialize generator
    gen = DatasetGenerator(seed=args.seed)

    # Validate class distribution length
    if len(args.classes) != len(args.class_dist):
        raise ValueError("classes and class-dist must have the same length")

    gen.generate_dataset(
        size=args.size,
        dim=args.dim,
        out_dir=args.out,
        distribution=args.distribution,
        classes=args.classes,
        class_distribution=args.class_dist,
    )
