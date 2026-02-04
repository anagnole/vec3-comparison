#!/bin/bash
set -e

PYTHON=".venv/bin/python"
RESET="./scripts/db/reset_and_wait.sh"
RUN="benchmarks/ingestion/run_single_dataset.py"

echo "=== Starting Ingestion Benchmarks ==="
echo "This will run all ingestion benchmarks sequentially."
echo ""

echo "=== Phase 1: Standard IVFFlat runs ==="

$RESET && $PYTHON $RUN -d 10k -i ivfflat --lists 100
$RESET && $PYTHON $RUN -d 50k -i ivfflat --lists 100
$RESET && $PYTHON $RUN -d 100k -i ivfflat --lists 100
$RESET && $PYTHON $RUN -d 500k -i ivfflat --lists 100
$RESET && $PYTHON $RUN -d 1m -i ivfflat --lists 100
$RESET && $PYTHON $RUN -d 2m -i ivfflat --lists 100

echo "=== Phase 2: Different dimensions ==="

$RESET && $PYTHON $RUN -d 50k_32d -i ivfflat --lists 100
$RESET && $PYTHON $RUN -d 100k_32d -i ivfflat --lists 100
$RESET && $PYTHON $RUN -d 50k_1536d -i ivfflat --lists 100
$RESET && $PYTHON $RUN -d 100k_768d -i ivfflat --lists 100

echo "=== Phase 3: HNSW parameter study (50k) ==="

$RESET && $PYTHON $RUN -d 50k -i hnsw --hnsw-m 16 --hnsw-ef 64
$RESET && $PYTHON $RUN -d 50k -i hnsw --hnsw-m 16 --hnsw-ef 128
$RESET && $PYTHON $RUN -d 50k -i hnsw --hnsw-m 32 --hnsw-ef 64
$RESET && $PYTHON $RUN -d 50k -i hnsw --hnsw-m 32 --hnsw-ef 128

echo "=== Phase 4: IVFFlat parameter study ==="

$RESET && $PYTHON $RUN -d 50k -i ivfflat --lists 200

echo "=== Phase 5: HNSW on larger datasets ==="

$RESET && $PYTHON $RUN -d 10k -i hnsw --hnsw-m 16 --hnsw-ef 64
$RESET && $PYTHON $RUN -d 100k -i hnsw --hnsw-m 16 --hnsw-ef 64
$RESET && $PYTHON $RUN -d 500k -i hnsw --hnsw-m 16 --hnsw-ef 64
$RESET && $PYTHON $RUN -d 1m -i hnsw --hnsw-m 16 --hnsw-ef 64

echo ""
echo "=== All ingestion benchmarks complete ==="
echo "Results saved to: results/raw/all_ingestion_results.json"
