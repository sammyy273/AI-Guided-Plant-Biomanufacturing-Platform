#!/bin/bash
# Run validation benchmark for all real species sequentially.
# Usage:
#   bash scripts/run_benchmark_all_species.sh
# Optional env vars:
#   RUNS=3 ITERATIONS=2 VARIANTS=5 MODE=benchmark
#   PROMOTER_GENERATION_DEVICE=cuda PROMOTER_EMBEDDING_DEVICE=cpu

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MODE="${MODE:-benchmark}"
RUNS="${RUNS:-3}"
ITERATIONS="${ITERATIONS:-2}"
VARIANTS="${VARIANTS:-5}"
SPECIES_LIST=(
  arabidopsis
  nbenthamiana
  ntobacum
  by2_cells
  tomato
  soybean
  rice
  maize
  wheat
)

export PROMOTER_GENERATION_DEVICE="${PROMOTER_GENERATION_DEVICE:-cuda}"
export PROMOTER_EMBEDDING_DEVICE="${PROMOTER_EMBEDDING_DEVICE:-cpu}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

LOG_DIR="$PROJECT_DIR/logs/benchmark_batch_${TIMESTAMP}"
OUT_ROOT="$PROJECT_DIR/outputs/validation_batch_${TIMESTAMP}"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

echo "============================================================"
echo "BENCHMARK BATCH: All Species"
echo "Timestamp: $TIMESTAMP"
echo "Mode: $MODE"
echo "Runs: $RUNS"
echo "Iterations/run: $ITERATIONS"
echo "Variants/iteration: $VARIANTS"
echo "Generation device: $PROMOTER_GENERATION_DEVICE"
echo "Embedding device: $PROMOTER_EMBEDDING_DEVICE"
echo "Logs: $LOG_DIR"
echo "Outputs: $OUT_ROOT"
echo "============================================================"
echo ""

TOTAL=0
SUCCESS=0
FAIL=0

for SPECIES in "${SPECIES_LIST[@]}"; do
  TOTAL=$((TOTAL + 1))
  LOG_FILE="$LOG_DIR/${SPECIES}.log"
  OUT_DIR="$OUT_ROOT/${SPECIES}"
  mkdir -p "$OUT_DIR"
  : > "$LOG_FILE"

  echo "[$(date '+%H:%M:%S')] Starting ${SPECIES} (${TOTAL}/${#SPECIES_LIST[@]})"
  echo "  Log: $LOG_FILE"
  echo "  Output: $OUT_DIR"
  echo "  Command: python -u scripts/validation.py --species $SPECIES --mode $MODE --runs $RUNS --iterations $ITERATIONS --variants $VARIANTS --output-dir $OUT_DIR"

  if python -u scripts/validation.py \
      --species "$SPECIES" \
      --mode "$MODE" \
      --runs "$RUNS" \
      --iterations "$ITERATIONS" \
      --variants "$VARIANTS" \
      --output-dir "$OUT_DIR" \
      2>&1 | tee "$LOG_FILE"; then
    echo "  DONE — $SPECIES"
    SUCCESS=$((SUCCESS + 1))
  else
    EXIT_CODE=$?
    echo "  FAILED ($EXIT_CODE) — $SPECIES"
    echo "  See: $LOG_FILE"
    FAIL=$((FAIL + 1))
  fi

  python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
  sleep 2
  echo ""
done

echo "============================================================"
echo "BENCHMARK BATCH COMPLETE"
echo "============================================================"
echo "Total:   $TOTAL"
echo "Success: $SUCCESS"
echo "Fail:    $FAIL"
echo "Logs:    $LOG_DIR"
echo "Outputs: $OUT_ROOT"
