# Benchmark Protocol & Configuration

*(Fill out as experimentation progresses)*

## 1. Environment Details
| Parameter | Chroma | pgvector |
|------------|---------|-----------|
| Host OS |  |  |
| CPU / RAM |  |  |
| Storage Type |  |  |
| Python / Postgres version |  |  |
| DB version |  |  |
| Deployment | local / Okeanos / cluster | local / Okeanos / cluster |

---

## 2. Dataset Configuration
- **Source**: synthetic random floats or public ANN dataset
- **Sizes**: small (10k), medium (100k), large (500k–1M)
- **Dimensions**: 512 (base), optional 768 / 1536
- **Type**: float32
- **Metadata**: `cls ∈ {A, B, C}` with distribution 50/30/20
- **Storage parity**: identical vectors/labels in both DBs

---

## 3. Ingestion Plan
| Parameter | Value |
|------------|--------|
| Batch size | 1,000 vectors |
| Warmup runs | none (timed once per dataset) |
| Metrics | total time, vectors/sec, disk size, peak RAM |
| Index build (pgvector) | measure time + size |
| Index build (Chroma) | if available, note implicit indexing behavior |

---

## 4. Query Workload
| Setting | Value |
|----------|--------|
| Number of queries | 200 (per dataset) |
| Query vector source | sampled from same dataset |
| `k` (neighbors) | 10 (base), 100 (optional) |
| Distance metrics | L2, cosine, (inner product optional) |
| Filters | none, `cls='A'`, `cls='C'` |
| Repeats | 3 runs per workload |
| Warmup | first 10 queries ignored |
| Concurrency | single client (optional sweep 1,4,8) |

---

## 5. Index Configurations (pgvector)
| Name | Metric | Params |
|-------|---------|---------|
| Baseline | none | sequential scan |
| IVFFLAT (L2) | L2 | `lists = 100`, `probes = 10` |
| IVFFLAT (cosine) | cosine | same parameters |
| HNSW (optional) | cosine/L2 | record parameters if tested |

---

## 6. Metrics to Record
- **Ingestion:** total seconds, vectors/sec, peak RAM, disk GB
- **Query:** latency (avg/p50/p95/p99), QPS, CPU%, memory%
- **Storage:** base + index size
- **Indexing:** build time, index size
- **Notes:** normalization method for cosine, caching effects

---

## 7. Result Artifacts
| Artifact | Format | Location |
|-----------|---------|-----------|
| Raw metrics | JSON / CSV | `results/raw/` |
| Plots | PNG / PDF | `results/plots/` |
| System specs | Markdown | `docs/env.md` |
| Report | Markdown / PDF | `docs/findings.md` |

---

## 8. Hygiene
- Fixed random seed for reproducibility
- Run on the same machine (or identical VMs) for both DBs
- Document cache-clearing or warmup methods
- Record timestamps for all measurements
