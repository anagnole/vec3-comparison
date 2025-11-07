# Team Instructions & Workflow

## 1. Setup
1. Create GitHub repo `vec3-comparison`.
2. Add folders: `docs/`, `benchmarks/`, `common/`, `results/`.
3. Create project board in YouTrack or GitHub Projects with columns:
   - Backlog → In progress → Review → Done.
4. Create four Epics:
   - EPIC-1 Installation & Setup
   - EPIC-2 Data Generation & Loading
   - EPIC-3 Query Workloads
   - EPIC-4 Measurement & Comparison

---

## 2. Reading Plan

### Person A — *Chroma Lead*
- Learn: installation (local/server), collection model, supported metrics.
- Read: [Chroma docs](https://docs.trychroma.com/).
- Output: short bullet notes in `docs/reading-notes.md`.

### Person B — *pgvector Lead*
- Learn: operators `<->`, `<#>`, `<=>`; indexes IVFFLAT/HNSW; Postgres setup.
- Read: [pgvector repo](https://github.com/pgvector/pgvector) & examples.
- Output: notes in same file.

### Both
- Read: general vector DB & ANN concepts (HNSW, IVF).
- Read: [Qdrant filtering benchmark README](https://github.com/qdrant/ann-filtering-benchmark-datasets).
- Meet for 30 min to summarize understanding.

---

## 3. Task Outline (matches course requirements)

### EPIC-1: Installation & Setup
- T1: Write `docs/charter.md` & `docs/protocol.md`.
- T2: Decide Okeanos VM specs; request or prepare local equivalents.
- T3-A: Install Chroma locally; verify insert/query.
- T3-B: Install Postgres + pgvector; verify insert/query.
- T4: Document environment details in `docs/env.md`.

**Done when:** both DBs operational locally and on Okeanos, with smoke test (1k vectors).

---

### EPIC-2: Data Generation & Loading
- T5: Define dataset plan (sizes, dimensions, label skew).
- T6: Generate vectors (random float32) using fixed seed.
- T7-A: Load into Chroma; record ingest time & storage.
- T7-B: Load into pgvector; record ingest time & storage.
- T8: Store metrics (CSV/JSON) in `results/raw/`.

**Done when:** both DBs loaded with identical datasets (10k & 100k at minimum).

---

### EPIC-3: Query Workloads
- T9: Lock workload matrix (metrics, filters, k values).
- T10-A: Verify Chroma supports chosen metrics/filters.
- T10-B: Verify pgvector supports same metrics; plan index build.
- T11-B: Create IVFFLAT index; record lists/probes values.

**Done when:** both DBs can run L2 & cosine queries, with and without filters.

---

### EPIC-4: Measurement & Comparison
- T12: Define metric schema for results.
- T13-A: Run query workloads on Chroma (10k/100k); collect latency/QPS.
- T13-B: Run query workloads on pgvector (baseline + indexed); collect same.
- T14-B: Measure index build time & size.
- T15: Aggregate results; make plots; write short interpretation.
- T16: Draft report `docs/findings.md`.

**Done when:** plots exist for ingestion speed, latency, throughput, and storage; report drafted.

---

## 4. Cadence
| Meeting | Purpose | Duration |
|----------|----------|-----------|
| Kickoff (today) | Confirm roles & epics | 15 min |
| Reading sync | Summarize docs & finalize protocol | 30 min |
| Weekly stand-ups | Blockers + next tasks | 10 min |
| Milestone reviews | Validate metrics & plots | 30 min |

---

## 5. Milestones
| Milestone | Deliverable | Target |
|------------|--------------|--------|
| M1 | Setup + smoke test complete | Week 1 |
| M2 | Ingestion benchmarks done | Week 2 |
| M3 | Query benchmarks done | Week 3 |
| M4 | Plots & report finalized | Week 4 |

---

## 6. Notes
- Keep all metrics machine-comparable (same hardware where possible).
- Use fixed random seed for reproducibility.
- Document any parameter tuning (e.g., IVFFLAT lists/probes).
- Optional: explore distributed/clustered setup and record observations.
