# Reading Guide – Vec3 Vector Database Comparison

This document lists recommended readings and reference materials to build a solid understanding of vector databases, embeddings, approximate nearest-neighbor (ANN) search, and the two systems compared in this project: Chroma and pgvector.

---

## 1. Core Concepts: Vector Databases and ANN Search

| Topic | Resource | Purpose |
|--------|-----------|----------|
| Introduction to vector databases | https://www.pinecone.io/learn/vector-database/ | Overview of how embeddings, similarity search, and vector databases relate. |
| Approximate nearest neighbor (ANN) search | https://www.pinecone.io/learn/ann/ | Explains the motivation and algorithms behind ANN indexing. |
| Distance metrics | https://qdrant.tech/documentation/concepts/distance/ | Defines cosine, Euclidean (L2), and inner-product similarities. |
| Indexing methods | https://milvus.io/docs/index_selection.md | Describes common index structures such as HNSW, IVF, and PQ. |

---

## 2. Chroma Documentation and Benchmarks

| Area | Resource | Purpose |
|-------|-----------|----------|
| Official documentation | https://docs.trychroma.com | API reference, installation instructions, and usage examples. |
| Source code repository | https://github.com/chroma-core/chroma | To inspect configuration options and understand implementation details. |
| Benchmark framework | https://github.com/qdrant/vector-db-benchmark | Contains ready-made scripts for data generation and query benchmarking that support Chroma. |
| Integration example | https://python.langchain.com/docs/integrations/vectorstores/chroma | Demonstrates how Chroma is used in practice for embedding storage and retrieval. |

---

## 3. pgvector Documentation and Performance

| Area | Resource | Purpose |
|-------|-----------|----------|
| Official repository and README | https://github.com/pgvector/pgvector | Describes operators, index types, and configuration parameters. |
| Quick-start guide | https://supabase.com/blog/pgvector-getting-started | Step-by-step tutorial for installation and first queries. |
| Performance tuning | https://supabase.com/blog/pgvector-performance | Practical advice on tuning IVFFLAT parameters (`lists`, `probes`). |
| Example queries | https://ankane.org/pgvector | Concise examples of SQL syntax for vector operations. |
| HNSW support discussion | https://github.com/pgvector/pgvector/pull/134 | Background on upcoming HNSW indexing in pgvector. |

---

## 4. Benchmarking Methodology

| Topic | Resource | Purpose |
|--------|-----------|----------|
| General benchmark suite | https://github.com/qdrant/vector-db-benchmark | Reference implementation of data generation, ingestion, and query benchmarks. |
| Benchmarking best practices | https://zilliz.com/blog/benchmarking-vector-databases | Guidance on avoiding bias and interpreting latency/throughput results. |
| ANN library background (FAISS) | https://faiss.ai/index.html | Explains the FAISS library used internally by several vector databases. |
| System metrics collection | https://psutil.readthedocs.io/en/latest/ | Documentation for measuring CPU and memory usage in Python. |

---

## 5. Datasets and Example Data Sources

| Source | Description |
|---------|-------------|
| https://github.com/qdrant/ann-filtering-benchmark-datasets | Public datasets (Fashion-MNIST, GloVe, etc.) for filtered ANN benchmarks. |
| http://corpus-texmex.irisa.fr/ | SIFT1M dataset, a standard benchmark with one million vectors. |
| https://laion.ai/blog/laion-400-open-dataset/ | Large-scale image embeddings (LAION) for realistic testing. |
| https://huggingface.co/datasets | Repository of open datasets, including pre-computed embeddings. |

---

## 6. Optional In-Depth Reading

| Topic | Resource |
|--------|-----------|
| Efficient similarity search (HNSW paper) | https://arxiv.org/abs/1603.09320 |
| Vector databases and embeddings infrastructure | https://weaviate.io/blog/vector-databases-embeddings-infrastructure |
| Cosine vs Euclidean distance | https://towardsdatascience.com/cosine-similarity-vs-euclidean-distance-67f15e72e826 |
| PostgreSQL index internals | https://www.postgresql.org/docs/current/indexes.html |

---

## 7. Suggested Reading Schedule

| Day | Focus | Reading Targets |
|------|--------|----------------|
| Day 1 | Fundamentals | Pinecone introduction and ANN overview |
| Day 2 | Chroma system | Chroma documentation and benchmark examples |
| Day 3 | pgvector system | pgvector README, quick-start, and performance tuning |
| Day 4 | Benchmark design | Qdrant benchmark suite and Zilliz benchmarking guide |
| Weekend | Deep dive | Optional papers and Milvus/Weaviate resources |

---

End of file.
