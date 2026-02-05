# Vector Database Comparison Project — Charter

## 1. Project Overview
Goal: Benchmark and compare two open-source vector database systems — **Chroma** and **pgvector** — across ingestion, query performance, and storage efficiency.  
This work follows the four main requirements specified in the course brief.

---

## 2. Objectives
- Understand the architecture and design trade-offs between a **purpose-built vector DB (Chroma)** and a **relational DB extension (pgvector)**.
- Build a reproducible benchmarking pipeline that generates embeddings, loads them into both databases, executes similarity queries, and collects performance metrics.
- Produce quantitative results (ingest speed, latency, throughput, storage, CPU/RAM usage) and a qualitative comparison of strengths/weaknesses.

---

## 3. Scope (aligned with course requirements)

### Requirement 1 — Installation & Setup
- Install and configure **Chroma** (embedded and server mode if possible) and **pgvector** (PostgreSQL + extension) on local and **Okeanos** VMs.
- Document environment specifications: OS, CPU/RAM, storage, DB versions.
- If any database offers a cluster/distributed mode, attempt or describe setup steps.

### Requirement 2 — Data Generation & Loading
- Generate or obtain large vector datasets that:
  - Do **not** fit fully in main memory.
  - Vary in dimensionality (e.g., 512, 768, 1536).
  - Are loaded identically into both DBs.
- Measure ingestion time (small/medium/large sets) and storage footprint.
- Record compression/indexing efficiency if visible.

### Requirement 3 — Query Generation
- Define a shared query workload that uses:
  - Common distance metrics (L2, cosine, inner product if available).
  - Filtered and unfiltered queries (e.g., metadata field `cls ∈ {A,B,C}`).
- Use the same set of query vectors for both DBs.
- Base workload design partly on [Qdrant ANN filtering benchmark](https://github.com/qdrant/ann-filtering-benchmark-datasets).

### Requirement 4 — Measurement & Comparison
- Measure and compare:
  - Query latency (avg, p95, p99)
  - Throughput (queries/sec)
  - CPU and memory utilization (where possible)
  - Index build time and disk usage
- Produce plots and interpretation of results.

---

## 4. Deliverables
1. Working installations of both DB systems (local and Okeanos)
2. Benchmark scripts (data generation, ingestion, query workloads)
3. Raw metrics (JSON/CSV) and visual results (plots)
4. Written report (2–4 pages) with analysis and recommendations

---

## 5. Roles & Responsibilities
| Member | Role | Focus |
|---------|------|-------|
| **Person A** | Chroma Lead | Installation, ingestion, query benchmarking |
| **Person B** | pgvector Lead | Installation, indexing, query benchmarking |
| Both | Coordinators | Data design, metric collection, analysis & writing |

---

## 6. Milestones
| Milestone | Description | Target |
|------------|--------------|---------|
| M1 | Environments ready, smoke tests run 
| M2 | Data generation & ingestion benchmarks 
| M3 | Query benchmarks & index tests 
| M4 | Results visualization & final report 

---

## 7. References
- [Chroma Docs](https://docs.trychroma.com/)
- [pgvector Docs](https://github.com/pgvector/pgvector)
- [Qdrant Benchmarks](https://github.com/qdrant/vector-db-benchmark)
