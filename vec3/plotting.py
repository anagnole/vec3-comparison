import os
from typing import Dict, List

import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def plot_latency_bars(
	results: List[Dict],
	metric: str,
	title: str,
	output_path: str,
) -> None:
	"""Create a grouped bar chart for a given latency metric.

	Each item in ``results`` is expected to have:
	  - ``db``: name of the backend (e.g. "chroma", "pgvector")
	  - ``dataset``: label for the dataset size (e.g. "10k")
	  - ``stats``: dict with latency metrics (e.g. {"mean": ..., "p99": ...})
	"""

	ensure_dir(os.path.dirname(output_path))

	# Collect unique datasets and backends in stable order.
	datasets = []
	backends = []
	for r in results:
		if r["dataset"] not in datasets:
			datasets.append(r["dataset"])
		if r["db"] not in backends:
			backends.append(r["db"])

	# Build a matrix: backends x datasets
	values = {db: [] for db in backends}
	for ds in datasets:
		for db in backends:
			stat = next(
				(r["stats"].get(metric) for r in results if r["db"] == db and r["dataset"] == ds),
				None,
			)
			values[db].append(stat if stat is not None else 0.0)

	x = range(len(datasets))
	width = 0.8 / max(len(backends), 1)

	plt.figure(figsize=(8, 5))
	for idx, db in enumerate(backends):
		offset = (idx - (len(backends) - 1) / 2) * width
		plt.bar([xi + offset for xi in x], values[db], width=width, label=db)

	plt.xticks(list(x), datasets)
	plt.ylabel(f"{metric} latency (s)")
	plt.title(title)
	plt.legend()
	plt.tight_layout()

	plt.savefig(output_path)
	plt.close()


def plot_ingest_bars(
	results: List[Dict],
	metric: str,
	title: str,
	output_path: str,
)	-> None:
	"""Create a grouped bar chart for ingestion metrics.

	Each item in ``results`` is expected to have:
	  - ``db``: name of the backend (e.g. "chroma", "pgvector")
	  - ``dataset``: label for the dataset size (e.g. "10k")
	  - the metric key directly on the dict (e.g. "duration_sec" or "vectors_per_sec").
	"""

	ensure_dir(os.path.dirname(output_path))

	datasets = []
	backends = []
	for r in results:
		if r["dataset"] not in datasets:
			datasets.append(r["dataset"])
		if r["db"] not in backends:
			backends.append(r["db"])

	values = {db: [] for db in backends}
	for ds in datasets:
		for db in backends:
			val = next(
				(r.get(metric) for r in results if r["db"] == db and r["dataset"] == ds),
				None,
			)
			values[db].append(val if val is not None else 0.0)

	x = range(len(datasets))
	width = 0.8 / max(len(backends), 1)

	plt.figure(figsize=(8, 5))
	for idx, db in enumerate(backends):
		offset = (idx - (len(backends) - 1) / 2) * width
		plt.bar([xi + offset for xi in x], values[db], width=width, label=db)

	plt.xticks(list(x), datasets)
	plt.ylabel(metric)
	plt.title(title)
	plt.legend()
	plt.tight_layout()

	plt.savefig(output_path)
	plt.close()

