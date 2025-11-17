  # vec3-comparison

  
This repository contains the code, documentation, and experimental results for our **Vector Database Benchmarking Project**, part of the *Data Management Systems* coursework.

We compare two open-source vector database systems:
-  **Chroma** — a purpose-built vector database optimized for embedding search.
-  **pgvector** — a PostgreSQL extension that adds vector similarity search.

---

## Project Goals

1. **Installation & Setup** — Deploy both DBs locally and on Okeanos; document setup and (if possible) distributed configurations.  
2. **Data Generation & Loading** — Generate large, high-dimensional vector datasets and load them into both databases, measuring ingestion time and storage footprint.  
3. **Query Benchmarking** — Execute a set of similarity queries (L2, cosine, inner product) with and without metadata filters.  
4. **Performance Measurement** — Record latency, throughput, CPU/memory usage, and index build efficiency for meaningful comparison.

---

##  Repository Structure

# Setup

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

