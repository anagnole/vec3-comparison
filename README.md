  # Vec3 Comparison: Chroma vs pgvector Benchmark Suite

A benchmarking project to compare **Chroma** and **pgvector** vector databases, developed as part of the *Analysis and Design of Information Systems* course at NTUA.

## Overview

This project provides an end-to-end benchmarking pipeline for evaluating two popular vector database solutions:

| Database | Type | Index Support |
|----------|------|---------------|
| **Chroma** | Purpose-built vector DB | HNSW (built-in) |
| **pgvector** | PostgreSQL extension | IVFFlat, HNSW |

### What We Measure

- **Ingestion Performance**: Throughput (vectors/sec), time breakdown, storage footprint
- **Query Performance**: Latency (mean, P50, P99), recall@k, throughput (QPS)
- **Resource Usage**: CPU utilization, memory consumption
- **Scalability**: Performance across dataset sizes (10K → 2M vectors)
- **Index Comparison**: HNSW vs IVFFlat with various parameters

---

## Repository Structure

```
vec3-comparison/
├── api/                    # REST API for running benchmarks (Node.js/Express)
├── benchmarks/
│   ├── ingestion/          # Ingestion benchmark scripts
│   ├── queries/            # Query benchmark scripts
│   └── plotting/           # Visualization scripts (matplotlib)
├── data/                   # Generated datasets (vectors.npy, metadata.jsonl)
├── docs/                   # Project documentation
├── paper/                  # LaTeX research paper
├── results/
│   ├── raw/                # JSON benchmark results
│   ├── plots/              # Generated visualizations
│   └── web/                # Results for web UI
├── scripts/
│   ├── benchmarks/         # Full benchmark suite runners
│   ├── data_generation/    # Dataset generation scripts
│   ├── db/                 # Database management (start, stop, reset)
│   └── ui/                 # Web UI start/stop scripts
├── vec3/                   # Core Python library
│   ├── generate_data.py    # Dataset generator
│   ├── ingest_chroma.py    # Chroma ingestion
│   ├── ingest_pgvector.py  # pgvector ingestion
│   ├── query_chroma.py     # Chroma queries
│   ├── query_pgvector.py   # pgvector queries
│   └── metrics.py          # Recall calculation
├── web/                    # React frontend for visualizing results
├── docker-compose.yml      # Database containers configuration
└── requirements.txt        # Python dependencies
```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Docker** and **Docker Compose**
- **Node.js 18+** (for Web UI, optional)

### 1. Clone and Setup Environment

```bash
git clone https://github.com/yourusername/vec3-comparison.git
cd vec3-comparison

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Start Database Containers

```bash
# Start both Chroma and pgvector containers
./scripts/db/start.sh

# Verify containers are running
docker ps
```

This starts:
- **pgvector** on `localhost:5432` (PostgreSQL with vector extension)
- **Chroma** on `localhost:8000` (HTTP API)

Both containers are configured with:
- 4GB memory limit
- 6 CPU cores
- Persistent volumes for data

### 3. Generate Test Datasets

```bash
# Generate all standard datasets
./scripts/data_generation/generate_all.sh
```

This creates datasets in `data/`:

| Dataset | Vectors | Dimensions | Size |
|---------|---------|------------|------|
| `10k` | 10,000 | 128 | ~5 MB |
| `50k` | 50,000 | 128 | ~25 MB |
| `100k` | 100,000 | 128 | ~50 MB |
| `500k` | 500,000 | 128 | ~250 MB |
| `1m` | 1,000,000 | 128 | ~500 MB |
| `2m` | 2,000,000 | 128 | ~1 GB |
| `50k_32d` | 50,000 | 32 | ~6 MB |
| `100k_32d` | 100,000 | 32 | ~12 MB |
| `50k_1536d` | 50,000 | 1536 | ~300 MB |
| `100k_768d` | 100,000 | 768 | ~300 MB |

### 4. Run Benchmarks

#### Run Full Benchmark Suite

```bash
# Run all ingestion benchmarks
./scripts/benchmarks/run_all_ingestion.sh

# Run all query benchmarks
./scripts/benchmarks/run_all_queries.sh
```

#### Run Individual Benchmarks

```bash
# Ingestion: Single dataset with IVFFlat index
.venv/bin/python benchmarks/ingestion/run_single_dataset.py -d 100k -i ivfflat --lists 100

# Ingestion: With HNSW index
.venv/bin/python benchmarks/ingestion/run_single_dataset.py -d 100k -i hnsw --hnsw-m 16 --hnsw-ef 64

# Query: Run queries on ingested data
.venv/bin/python benchmarks/queries/run_single_dataset.py 100k --both -i hnsw --hnsw-m 16 --hnsw-ef 64
```

### 5. Generate Plots

```bash
# Generate ingestion plots
.venv/bin/python benchmarks/plotting/ingestion_plots.py

# Generate query plots
.venv/bin/python benchmarks/plotting/queries_plots.py
```

Plots are saved to `results/plots/ingestion/` and `results/plots/queries/`.

---

## Web UI (Optional)

A React-based dashboard for visualizing results and running benchmarks interactively.

### Setup Web UI

```bash
# Install API dependencies
cd api && npm install && cd ..

# Install frontend dependencies
cd web && npm install && cd ..
```

### Start Web UI

```bash
./scripts/ui/start.sh
```

Access at: **http://localhost:5173**

![alt text](image.png)

---

## Database Management

```bash
# Start databases
./scripts/db/start.sh

# Stop databases
./scripts/db/stop.sh

# Restart databases
./scripts/db/restart.sh

# Reset databases (delete all data)
./scripts/db/reset_and_wait.sh
```

---

## Benchmark Configuration

### Index Types

**pgvector** supports two index types:

| Index | Parameters | Best For |
|-------|------------|----------|
| **IVFFlat** | `--lists N` | Large datasets, faster build |
| **HNSW** | `--hnsw-m M --hnsw-ef EF` | Higher recall, slower build |

**Chroma** uses HNSW by default (built during ingestion).

### Batch Sizes

Benchmarks test multiple batch sizes: `100, 500, 1000, 5000`

### Query Modes

- `nofilter`: Pure vector similarity search
- `filter`: Vector search with metadata filtering (e.g., `cls = 'A'`)

---

## Results

Results are stored in JSON format:

- `results/raw/all_ingestion_results.json` - Ingestion benchmarks
- `results/raw/all_query_results.json` - Query benchmarks

### Sample Ingestion Result

```json
{
  "dataset": "100k",
  "vectors": 100000,
  "dimensions": 128,
  "chroma": [{
    "batch_size": 1000,
    "duration_sec": 45.2,
    "vectors_per_sec": 2212,
    "storage_mb": 89.5
  }],
  "pgvector": [{
    "batch_size": 1000,
    "index_type": "hnsw",
    "duration_ingest_sec": 12.3,
    "duration_index_sec": 8.7,
    "vectors_per_sec": 4762,
    "storage_mb": 67.2
  }]
}
```

---

## Key Findings

Based on our benchmarks:

| Metric | Winner | Notes |
|--------|--------|-------|
| **Ingestion Speed** | pgvector | 2-5x faster (separate index build) |
| **Query Latency** | pgvector | Lower latency across all dataset sizes |
| **Recall** | Chroma | Consistently higher recall@k |
| **Storage Efficiency** | pgvector | Lower storage overhead |
| **Ease of Setup** | Chroma | No schema/index configuration needed |

See [paper/main.tex](paper/main.tex) for detailed analysis.

---

## Project Structure Details

### Core Library (`vec3/`)

| File | Description |
|------|-------------|
| `generate_data.py` | Synthetic dataset generator (Gaussian vectors + metadata) |
| `ingest_chroma.py` | Chroma ingestion with Docker stats monitoring |
| `ingest_pgvector.py` | pgvector ingestion with index building |
| `query_chroma.py` | Chroma query execution with recall calculation |
| `query_pgvector.py` | pgvector query execution |
| `metrics.py` | Recall@k calculation using brute-force ground truth |

### Benchmark Scripts (`benchmarks/`)

| File | Description |
|------|-------------|
| `ingestion/run_single_dataset.py` | Run ingestion benchmark for one dataset |
| `queries/run_single_dataset.py` | Run query benchmark for one dataset |
| `plotting/ingestion_plots.py` | Generate all ingestion visualizations |
| `plotting/queries_plots.py` | Generate all query visualizations |

---

## Docker Configuration

Both databases run with identical resource constraints for fair comparison:

```yaml
deploy:
  resources:
    limits:
      cpus: '6'
      memory: 4g
    reservations:
      cpus: '2'
      memory: 2g
```

---

## Troubleshooting

### Databases won't start
```bash
# Check Docker is running
docker info

# Check for port conflicts
lsof -i :5432  # pgvector
lsof -i :8000  # Chroma
```

### Out of memory during large dataset ingestion
```bash
# Increase Docker memory limit in docker-compose.yml
# Or use smaller batch sizes
```

### Connection refused errors
```bash
# Wait for databases to fully initialize
./scripts/db/reset_and_wait.sh
```

---

## License

This project is part of academic coursework at NTUA. See [LICENSE](LICENSE) for details.

---

## Authors

- Advanced Database Systems Course Project, NTUA 2025-2026

