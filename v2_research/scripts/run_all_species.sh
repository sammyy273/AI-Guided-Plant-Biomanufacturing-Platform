#!/bin/bash
# Run auto-loop for all 9 species sequentially
# Usage: bash scripts/run_all_species.sh
#
# Runs 8 iterations per species, 3 variants per iteration
# Uses Evo2 + D3LM + mutational ensemble
# Estimated total time: ~90 minutes (10 min per species)
#
# Output: outputs/<species>/batch_<timestamp>/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_DIR/logs/batch_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

SPECIES_LIST="arabidopsis nbenthamiana rice maize tomato soybean wheat ntobacum by2_cells"

echo "============================================================"
echo "BATCH RUN: All Species"
echo "Timestamp: $TIMESTAMP"
echo "Iterations: 8 per species"
echo "Variants: 3 per iteration"
echo "Log dir: $LOG_DIR"
echo "============================================================"
echo ""

TOTAL=0
SUCCESS=0
FAIL=0

for SPECIES in $SPECIES_LIST; do
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "[$(date '+%H:%M:%S')] Starting: $SPECIES ($TOTAL/9)"
    echo "------------------------------------------------------------"

    LOG_FILE="$LOG_DIR/${SPECIES}.log"

    if python scripts/auto_loop_v2.py \
        --species "$SPECIES" \
        --iterations 8 \
        --variants 3 \
        > "$LOG_FILE" 2>&1; then

        # Extract summary from log
        BEST_SCORE=$(grep "Best composite:" "$LOG_FILE" | tail -1 | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
        CANDIDATES=$(grep "Total candidates generated:" "$LOG_FILE" | tail -1 | grep -oP '[0-9]+' || echo "0")
        PASSED=$(grep "Total passed hard filters:" "$LOG_FILE" | tail -1 | grep -oP '[0-9]+' || echo "0")
        PROGRESSION=$(grep "Composite score progression:" "$LOG_FILE" | tail -1 | sed 's/.*Composite score progression: //' || echo "N/A")

        echo "  DONE — Best: $BEST_SCORE | Candidates: $CANDIDATES | Passed: $PASSED"
        echo "  Progression: $PROGRESSION"
        SUCCESS=$((SUCCESS + 1))
    else
        EXIT_CODE=$?
        echo "  FAILED (exit code $EXIT_CODE)"
        echo "  See: $LOG_FILE"
        # Print last 10 lines of error
        tail -10 "$LOG_FILE" 2>/dev/null | sed 's/^/    /'
        FAIL=$((FAIL + 1))
    fi

    # Force GPU memory cleanup between species
    python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
    sleep 2

    echo ""
done

echo "============================================================"
echo "BATCH COMPLETE"
echo "============================================================"
echo "Total:  $TOTAL"
echo "Passed: $SUCCESS"
echo "Failed: $FAIL"
echo ""
echo "Logs: $LOG_DIR/"
echo ""

# Generate summary table from JSON (not grep — avoids timestamp corruption)
python3 -c "
import json, os, glob

log_dir = '$LOG_DIR'
output_base = '$PROJECT_DIR/outputs'
species_list = '$SPECIES_LIST'.split()

lines = []
lines.append(f\"{'Species':<16} | {'Best Score':>10} | {'Candidates':>10} | {'Passed':>6} | Progression\")
lines.append('-' * 110)

for sp in species_list:
    sp_dirs = sorted(glob.glob(f'{output_base}/{sp}/*'))
    batch_dir = None
    for d in reversed(sp_dirs):
        summary = os.path.join(d, 'loop_summary.json')
        if os.path.exists(summary):
            with open(summary) as f:
                s = json.load(f)
            if s['iterations_requested'] == 8 and s.get('iterations_completed', 0) == 8:
                batch_dir = d
                break
    if not batch_dir:
        for d in reversed(sp_dirs):
            summary = os.path.join(d, 'loop_summary.json')
            if os.path.exists(summary):
                batch_dir = d
                break

    if batch_dir:
        with open(os.path.join(batch_dir, 'loop_summary.json')) as f:
            s = json.load(f)
        total_cands = sum(r['n_candidates'] for r in s['results'])
        total_passed = sum(r['n_passed_filters'] for r in s['results'])
        prog = ' -> '.join(f\"{r['top_composite_score']:.3f}\" for r in s['results'] if r.get('top_composite_score'))
        best = s['best_composite_score']
        lines.append(f'{sp:<16} | {best:>10.3f} | {total_cands:>10} | {total_passed:>6} | {prog}')
    else:
        lines.append(f'{sp:<16} | NO DATA')

text = '\n'.join(lines)
print(text)
with open(os.path.join(log_dir, 'summary.txt'), 'w') as f:
    f.write(text + '\n')
"
done

echo "Summary:"
cat "$LOG_DIR/summary.txt"
