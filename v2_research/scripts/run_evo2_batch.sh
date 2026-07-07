#!/bin/bash
# Evo2-only promoter generation batch run across all species
# Runs auto_loop_v2.py with --models evo2 (no d3lm, no mutational baseline)
# Each species gets 5 iterations of Evo2 generation + scoring
set -euo pipefail

export NVIDIA_API_KEY='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'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

SPECIES=(
    arabidopsis
    nbenthamiana
    rice
    tomato
    maize
    wheat
    soybean
    ntobacum
    by2_cells
)

ITERATIONS=5
VARIANTS=10       # Evo2 API calls per iteration (10 variants × 5 iter = 50 API calls/species)
MODELS="evo2"

echo "============================================================"
echo "Evo2 Batch Promoter Generation"
echo "============================================================"
echo "Species: ${SPECIES[*]}"
echo "Iterations per species: $ITERATIONS"
echo "Variants per iteration: $VARIANTS"
echo "Models: $MODELS"
echo "Total API calls: $(( ${#SPECIES[@]} * ITERATIONS * VARIANTS ))"
echo "Started: $(date)"
echo ""

RESULTS=()
FAILURES=()

for species in "${SPECIES[@]}"; do
    echo ""
    echo "============================================================"
    echo ">>> Starting: $species"
    echo "============================================================"

    OUTDIR="outputs/evo2_only_${species}_$(date +%Y%m%d_%H%M%S)"

    if python3 scripts/auto_loop_v2.py \
        --species "$species" \
        --iterations "$ITERATIONS" \
        --models "$MODELS" \
        --variants "$VARIANTS" \
        --output-dir "$OUTDIR" \
        2>&1 | tee "logs/evo2_batch_${species}.log"; then

        # Extract best composite score from summary
        BEST=$(python3 -c "
import json
with open('$OUTDIR/loop_summary.json') as f:
    s = json.load(f)
print(f'{s[\"best_composite_score\"]:.4f}')
" 2>/dev/null || echo "N/A")

        RESULTS+=("$species: composite=$BEST dir=$OUTDIR")
        echo ">>> COMPLETED: $species (best composite: $BEST)"
    else
        FAILURES+=("$species")
        RESULTS+=("$species: FAILED")
        echo ">>> FAILED: $species"
    fi

    # Brief pause between species to respect rate limits
    sleep 10
done

echo ""
echo "============================================================"
echo "BATCH COMPLETE"
echo "============================================================"
echo "Finished: $(date)"
echo ""
echo "Results:"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done

if [ ${#FAILURES[@]} -gt 0 ]; then
    echo ""
    echo "Failures: ${FAILURES[*]}"
fi
