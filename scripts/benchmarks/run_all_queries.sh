#!/bin/bash
# Complete Query Benchmark Suite
# 
# Phase 1: Size scaling with HNSW (m=16, ef=64) - all dataset sizes
# Phase 2: Index comparison on 50k dataset
# Phase 3: Distance metric comparison (optional)

set -e

PYTHON=".venv/bin/python"
RESET="./scripts/db/reset_and_wait.sh"
INGEST="benchmarks/ingestion/run_single_dataset.py"
QUERY="benchmarks/queries/run_single_dataset.py"

echo "============================================================"
echo "       COMPLETE QUERY BENCHMARK SUITE"
echo "============================================================"
echo ""
echo "This script will run comprehensive query benchmarks:"
echo "  Phase 1: Size scaling (HNSW m=16, ef=64) - 10k to 2m"
echo "  Phase 2: Index comparison on 50k"
echo "  Phase 3: Distance metrics (optional)"
echo ""

# Clear previous results
if [ "$1" == "--fresh" ]; then
    rm -f results/raw/all_query_results.json
    echo "Cleared previous query results"
fi

echo ""
echo "============================================================"
echo "PHASE 1: Size Scaling (HNSW m=16, ef=64)"
echo "============================================================"

PHASE1_DATASETS="10k 50k 100k 500k 1m 2m"
HNSW_M=16
HNSW_EF=64

for dataset in $PHASE1_DATASETS; do
    echo ""
    echo "--- Dataset: $dataset (HNSW m=$HNSW_M, ef=$HNSW_EF) ---"
    
    # Ingest with HNSW index
    $RESET && $PYTHON $INGEST -d $dataset -i hnsw --hnsw-m $HNSW_M --hnsw-ef $HNSW_EF
    
    # Run queries (both filter and nofilter)
    $PYTHON $QUERY $dataset --both -i hnsw --hnsw-m $HNSW_M --hnsw-ef $HNSW_EF
done

echo ""
echo "Phase 1 complete!"

echo ""
echo "============================================================"
echo "PHASE 2: Index Comparison (50k dataset)"
echo "============================================================"

PHASE2_DATASET="50k"

# --- IVFFlat with lists=100 ---
echo ""
echo "--- IVFFlat (lists=100) ---"
$RESET && $PYTHON $INGEST -d $PHASE2_DATASET -i ivfflat --lists 100
$PYTHON $QUERY $PHASE2_DATASET --both -i ivfflat --lists 100

# --- IVFFlat with lists=200 ---
echo ""
echo "--- IVFFlat (lists=200) ---"
$RESET && $PYTHON $INGEST -d $PHASE2_DATASET -i ivfflat --lists 200
$PYTHON $QUERY $PHASE2_DATASET --both -i ivfflat --lists 200

# --- HNSW with m=16, ef=64 (already done in Phase 1, skip re-ingest) ---
# If you want to re-run with fresh containers, uncomment:
# echo ""
# echo "--- HNSW (m=16, ef=64) ---"
# $RESET && $PYTHON $INGEST -d $PHASE2_DATASET -i hnsw --hnsw-m 16 --hnsw-ef 64
# $PYTHON $QUERY $PHASE2_DATASET --both -i hnsw --hnsw-m 16 --hnsw-ef 64

# --- HNSW with m=16, ef=128 ---
echo ""
echo "--- HNSW (m=16, ef=128) ---"
$RESET && $PYTHON $INGEST -d $PHASE2_DATASET -i hnsw --hnsw-m 16 --hnsw-ef 128
$PYTHON $QUERY $PHASE2_DATASET --both -i hnsw --hnsw-m 16 --hnsw-ef 128

# --- HNSW with m=32, ef=64 ---
echo ""
echo "--- HNSW (m=32, ef=64) ---"
$RESET && $PYTHON $INGEST -d $PHASE2_DATASET -i hnsw --hnsw-m 32 --hnsw-ef 64
$PYTHON $QUERY $PHASE2_DATASET --both -i hnsw --hnsw-m 32 --hnsw-ef 64

# --- HNSW with m=32, ef=128 ---
echo ""
echo "--- HNSW (m=32, ef=128) ---"
$RESET && $PYTHON $INGEST -d $PHASE2_DATASET -i hnsw --hnsw-m 32 --hnsw-ef 128
$PYTHON $QUERY $PHASE2_DATASET --both -i hnsw --hnsw-m 32 --hnsw-ef 128

echo ""
echo "Phase 2 complete!"

# ============================================================
# PHASE 3: Distance Metrics (Optional)
# ============================================================
# Uncomment to enable Phase 3
#
# echo ""
# echo "============================================================"
# echo "PHASE 3: Distance Metrics Comparison (50k, HNSW)"
# echo "============================================================"
# 
# # Note: For cosine/inner_product metrics, the index must be created
# # with the matching distance function. This requires modifying 
# # ingest_pgvector.py to support different metrics at index creation.
# 
# PHASE3_DATASET="50k"
# 
# # --- Euclidean (L2) - already covered above ---
# 
# # --- Cosine similarity ---
# echo ""
# echo "--- Cosine metric ---"
# # Would need to re-ingest with cosine index
# # $RESET && $PYTHON $INGEST -d $PHASE3_DATASET -i hnsw --hnsw-m 16 --hnsw-ef 64 --metric cosine
# # $PYTHON $QUERY $PHASE3_DATASET --both -i hnsw --hnsw-m 16 --hnsw-ef 64 --metric cosine
# 
# echo ""
# echo "Phase 3 complete!"

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "============================================================"
echo "ALL QUERY BENCHMARKS COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to: results/raw/all_query_results.json"
echo ""
echo "  $PYTHON benchmarks/plotting/queries_plots.py"
