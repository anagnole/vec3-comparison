# Dataset Generation Plan

## Overview
This document captures the **full design plan** for synthetic dataset generation for the Vec3 comparison project. It includes **everything we discussed**, from broad strategy to specific configuration decisions.

The generator will be modular, reproducible, flexible, and able to support multiple dataset families later. For now we begin with the baseline synthetic datasets.

---

## Goals of the Dataset Generator
1. Provide **fully reproducible synthetic datasets** for benchmarking Chroma and pgvector.
2. Allow **configurable size**, **dimensionality**, **category distributions**, and later **distribution families**.
3. Produce metadata suitable for **filtering queries**.
4. Be flexible enough to expand later with real datasets or alternative generation patterns.
5. Generate data in a format compatible with our ingestion scripts.

---

## What Other Repos Do (Context)
We reviewed the structure and patterns from:
- **qdrant/vector-db-benchmark**
- **qdrant/ann-filtering-benchmark-datasets**

These repos:
- Use synthetic random Gaussian vectors
- Use `.npy` and `.jsonl` for efficient storage
- Include metadata fields used for filtering
- Often generate multiple dataset sizes
- Provide several distributions and data shapes

We are following the same principles but making it cleaner, simpler, and more configurable.

---

## General Design Decisions
### ✔ Distribution Options
We will support multiple distributions *later*, but for now:
- **Start with: Standard Gaussian (normal) distribution**
- `np.random.randn(size, dim)`

### ✔ Dimensionality
Initial dimension:
- **100 dimensions**

Later we will add configurable dims (e.g., 128, 256, 512, 2048).

### ✔ Vector Density / Structure
For now:
- **Dense vectors (Gaussian)**
- No sparsity or structured patterns

Later: option for sparse, clustered, multimodal, etc.

### ✔ Dataset Sizes (Initial Batch)
We will generate three initial datasets:
- **10k vectors**
- **100k vectors**
- **500k vectors**

Later we may add 1M or other stress-test sizes depending on performance.

### ✔ Reproducibility
- **A fixed random seed** per dataset generation
- Seed stored in a variable for transparency

### ✔ Metadata / Filtering Fields
We want realistic category filtering. We decided:
- Metadata contains **only one field** for now: `cls`
- Category values: **"A", "B", "C"**

#### Category Distribution Options
Two options were considered:
- Equal distribution
- Skewed distribution

We chose:
- **Skewed distribution**:
  - **A = 10%**
  - **B = 30%**
  - **C = 60%**

This gives realistic filtering behavior.

### ✔ Metadata Format
We chose:
- **strings**: "A", "B", "C"

### ✔ File Output Format
We follow standard ANN benchmarking format:

1. **vectors.npy**
   - Shape: `(size, dim)`
   - dtype: `float32`

2. **metadata.jsonl**
   - One JSON object per line: `{ "cls": "A" }`
   - Efficient streaming format

### ✔ Directory Structure
```
data/<size>/
  vectors.npy
  metadata.jsonl
```

Example:
```
data/100k/vectors.npy
data/100k/metadata.jsonl
```

---

## Future Extensions (Planned)
We will later expand with:
- Different distributions: uniform, clustered, multimodal
- Different dimensions: 32, 128, 256, 512, 1024
- Different metadata schemas: more fields, numeric properties
- Real datasets (e.g., SIFT, DEEP1B, glove embeddings)
- Query generation scripts
- CLI for generating new datasets on demand

---

## Implementation Plan
### 1. Create a dedicated generator module:
`vec3/generate_data.py`

Contains:
- function: `generate_dataset(size, dim, out_dir, seed)`
- helper: `_generate_metadata(size)`
- helper: `_write_jsonl(path, metadata)`
- helper: `_write_vectors(path, vectors)`

### 2. Use fixed seed
We'll define a constant seed (e.g., `42`) at the start of the script.

### 3. Generate the three initial datasets
- 10k
- 100k
- 500k

### 4. Verify with small test run
- Quick smoke test on 1k vectors

---

## Summary of All Agreed Decisions
**Distribution:** Standard Gaussian

**Dimensionality:** 100

**Sizes:** 10k, 100k, 500k

**Categories:** "A", "B", "C"

**Category Distribution:** A=10%, B=30%, C=60%

**Metadata:** Only `cls`

**Output:** `.npy` + `.jsonl`

**Folder Structure:** `data/<size>/`

**Seed:** Yes, fixed

**No CLI or queries yet**

**Goal:** Simple first-generation datasets, expandable later

---

## Next Step
Begin implementing the dataset generator in `vec3/generate_data.py` following the above design.