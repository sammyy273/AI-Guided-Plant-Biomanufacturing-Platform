# =============================================================================
# auto_loop.py  — Combined Auto-Research Loop
#
# WHAT THIS DOES:
#   Runs the promoter design pipeline in a continuous self-improving loop.
#   Each iteration:
#     1. Generates 60 synthetic promoter candidates (Evo2 API)
#     2. Filters by hard criteria (TATA, CAAT, GC 40-60%)
#     3. Scores with DNABERT-2 cosine similarity to 2×35S
#     4. Saves composite ranking
#     5. Feeds TOP candidate back as seed for next generation
#     6. Repeats until target score reached or max iterations hit
#
# WHY SUBPROCESS (not direct imports):
#   subprocess.run() isolates each step. If step3 crashes (e.g., API timeout),
#   the loop catches it and retries rather than crashing the entire process.
#   For overnight runs this is critical — direct Python imports would propagate
#   any exception and kill the whole loop.
#
# WHY 30-MINUTE SLEEP:
#   60 Evo2 API calls take ~20-30 minutes. The sleep adds buffer before the
#   next iteration to avoid cascading rate limit (429) errors.
#
# HOW THE FEEDBACK LOOP WORKS:
#   After each successful iteration, the top-ranked candidate sequence is saved
#   to data/current_seed.fasta. At the start of the next iteration, step3 checks
#   for this file and uses it as the primary seed (overriding the 35S core seed).
#   This is directed evolutionary search: each generation starts from the best
#   sequence discovered so far, not a fixed reference.
#
# TARGET:
#   composite_score > 0.85 is the convergence criterion.
#   If achieved, the loop saves the winning sequence and exits cleanly.
#   If not achieved after max_iterations, it exits with the best candidate found.
#
# HOW TO RUN OVERNIGHT ON A SERVER:
#   nohup python auto_loop.py > auto_loop.out 2>&1 &
#   Monitor: tail -f auto_loop.log
#
# HOW TO RUN LOCALLY (Windows):
#   python auto_loop.py
#   (Keep the terminal open or use Windows Task Scheduler)
# =============================================================================

import os
import sys
import time
import json
import shutil
import subprocess
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import load_fasta, save_fasta

# ── LOOP SETTINGS ────────────────────────────────────────────────────────────
MAX_ITERATIONS    = 10       # Stop after this many iterations regardless
TARGET_SCORE      = 0.85     # Stop early if top candidate reaches this composite score
SLEEP_BETWEEN     = 1800     # Seconds to wait between iterations (30 min)
RETRY_WAIT        = 300      # Seconds to wait after a failed step (5 min)
_SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
LOOP_LOG          = os.path.join(_SCRIPT_DIR, "..", "logs", "auto_loop.log")
METRICS_FILE      = os.path.join(_SCRIPT_DIR, "..", "logs", "auto_loop_metrics.json")
CUSTOM_SEED_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "current_seed.fasta")


# ── LOGGING ──────────────────────────────────────────────────────────────────
def log(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"
    with open(LOOP_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


# ── RUN ONE SCRIPT AS SUBPROCESS ─────────────────────────────────────────────
def run_step(script_name: str) -> bool:
    """
    Run a pipeline step as a subprocess.
    Returns True if successful (exit code 0), False otherwise.

    Using subprocess instead of direct imports means:
    - Each step runs in its own Python process
    - Crashes in step3 do not crash the loop
    - Memory is released between steps (important for overnight runs)
    """
    log(f"Running {script_name}...")
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        log(f"FAILED: {script_name}", level="ERROR")
        # Print last 500 chars of stderr for diagnosis
        stderr = result.stderr.strip()
        if stderr:
            log(f"Error output: {stderr[-500:]}", level="ERROR")
        return False
    log(f"Completed: {script_name}")
    return True


# ── GET BEST CANDIDATE FROM LAST RUN ─────────────────────────────────────────
def get_best_candidate() -> tuple[str | None, float]:
    """
    Reads the ranking CSV from the last step4 run.
    Returns (sequence_string, composite_score) of the top candidate.
    Returns (None, 0.0) if no results available.

    Note: ranking CSV has candidate ID as the DataFrame index (first column),
    not as a regular column. Use df.index[0] not df.iloc[0, 0].
    """
    if not os.path.exists(config.RANKING_CSV):
        return None, 0.0

    try:
        df = pd.read_csv(config.RANKING_CSV, index_col=0)
        if df.empty:
            return None, 0.0

        best_id    = df.index[0]              # candidate ID (e.g. seed_35S_core_v03)
        best_score = float(df.loc[best_id, "composite_score"])

        # Get full sequence from the all_candidates FASTA
        if not os.path.exists(config.CANDIDATES_FASTA):
            return None, 0.0
        candidates = load_fasta(config.CANDIDATES_FASTA)
        best_seq   = candidates.get(best_id, None)
        return best_seq, best_score

    except Exception as e:
        log(f"Error reading ranking CSV: {e}", level="WARN")
        return None, 0.0


# ── SAVE ITERATION METRICS ────────────────────────────────────────────────────
def save_metrics(iteration: int, score: float, candidates_passed: int,
                 best_id: str | None):
    """
    Appends one row of metrics to the JSON metrics file.
    Used to track progress across iterations for the meeting presentation.
    """
    record = {
        "iteration":        iteration,
        "timestamp":        datetime.now().isoformat(),
        "best_score":       score,
        "candidates_passed":candidates_passed,
        "best_candidate_id":best_id,
    }
    existing = []
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            try:
                existing = json.load(f)
            except Exception:
                existing = []
    existing.append(record)
    with open(METRICS_FILE, "w") as f:
        json.dump(existing, f, indent=2)


# ── COUNT PASSING CANDIDATES ─────────────────────────────────────────────────
def count_passing_candidates() -> int:
    """Count how many candidates passed the hard filter in the last run."""
    if not os.path.exists(config.ALL_SCORED_CSV):
        return 0
    try:
        df = pd.read_csv(config.ALL_SCORED_CSV, index_col=0)
        return len(df)
    except Exception:
        return 0


# ── BACKUP OUTPUTS ────────────────────────────────────────────────────────────
def backup_iteration_outputs(iteration: int):
    """Save a copy of this iteration's outputs before overwriting."""
    backup_dir = f"outputs/iteration_{iteration:02d}"
    os.makedirs(backup_dir, exist_ok=True)
    for fname in ["top3_candidates.fasta", "ranking_table.csv",
                  "all_candidates_scored.csv"]:
        src = os.path.join(config.OUTPUT_DIR, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(backup_dir, fname))
    if os.path.exists(config.CANDIDATES_FASTA):
        shutil.copy(config.CANDIDATES_FASTA,
                    os.path.join(backup_dir, "all_candidates.fasta"))
    log(f"Backed up iteration {iteration} outputs to {backup_dir}")


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    log("=" * 60)
    log("AUTO-RESEARCH LOOP STARTED")
    log(f"Max iterations:    {MAX_ITERATIONS}")
    log(f"Target score:      {TARGET_SCORE}")
    log(f"Sleep between:     {SLEEP_BETWEEN}s ({SLEEP_BETWEEN//60} min)")
    log(f"Model:             {getattr(config, 'EVO2_MODEL', 'evo2-40b')}")
    log("=" * 60)

    # ── One-time setup: fetch and score references ────────────────────────────
    if not os.path.exists(config.REF_FASTA):
        log("First run: fetching reference promoters...")
        if not run_step("fetch_references.py"):
            log("Cannot fetch references. Exiting.", level="ERROR")
            return
        run_step("score_references.py")
    else:
        log("References already fetched. Skipping step1/step2.")

    iteration  = 1
    best_score_ever = 0.0
    best_seq_ever   = None

    while iteration <= MAX_ITERATIONS:
        log("")
        log(f"{'─'*60}")
        log(f"ITERATION {iteration} of {MAX_ITERATIONS}  "
            f"(best score so far: {best_score_ever:.4f})")
        log(f"{'─'*60}")

        # ── Step 3: Generate candidates ───────────────────────────────────────
        generation_ok = run_step("generate_candidates.py")
        if not generation_ok:
            log(f"Generation failed. Waiting {RETRY_WAIT}s before retry.", level="WARN")
            time.sleep(RETRY_WAIT)
            continue   # retry same iteration

        # ── Step 4: Filter and rank ────────────────────────────────────────────
        ranking_ok = run_step("filter_and_rank.py")
        if not ranking_ok:
            log("Ranking step failed. Skipping to next iteration.", level="WARN")
            time.sleep(60)
            iteration += 1
            continue

        # ── Read results ───────────────────────────────────────────────────────
        best_seq, best_score = get_best_candidate()
        n_passing            = count_passing_candidates()

        log(f"Candidates passing filters: {n_passing}")
        log(f"Top composite score:        {best_score:.4f}")

        if best_seq:
            log(f"Best sequence preview:      {best_seq[:60]}...")

        # ── Save metrics for presentation ──────────────────────────────────────
        best_id = None
        if os.path.exists(config.RANKING_CSV):
            try:
                df = pd.read_csv(config.RANKING_CSV, index_col=0)
                if not df.empty:
                    best_id = df.index[0]
            except Exception:
                pass

        save_metrics(iteration, best_score, n_passing, best_id)

        # ── Backup this iteration's outputs ────────────────────────────────────
        backup_iteration_outputs(iteration)

        # ── Update best-ever tracker ───────────────────────────────────────────
        if best_score > best_score_ever and best_seq:
            best_score_ever = best_score
            best_seq_ever   = best_seq
            log(f"New best score: {best_score_ever:.4f} (iteration {iteration})")
            # Save the all-time best
            save_fasta({"all_time_best": best_seq_ever},
                       "outputs/all_time_best_candidate.fasta")

        # ── Check convergence ──────────────────────────────────────────────────
        if best_score >= TARGET_SCORE:
            log("")
            log("=" * 60)
            log(f"TARGET REACHED: composite score {best_score:.4f} >= {TARGET_SCORE}")
            log(f"Winning sequence saved to outputs/all_time_best_candidate.fasta")
            log("=" * 60)
            break

        # ── No candidates passed — handle gracefully ───────────────────────────
        if n_passing == 0 or best_seq is None:
            log("No candidates passed hard filter this iteration.", level="WARN")
            log("Possible causes: evo2-7b degenerate outputs, or API quality issue.")
            log("Continuing loop with original seed...")
            # Do not update seed — keep previous good seed or fall back to reference
            iteration += 1
            time.sleep(RETRY_WAIT)
            continue

        # ── Feed best candidate back as seed for next iteration ────────────────
        # Use last 150 bp as the seed (contains the core promoter / TATA region)
        seed_seq = best_seq[-150:]
        save_fasta({"seed_from_iteration": seed_seq}, CUSTOM_SEED_FILE)
        log(f"Saved best candidate (last 150 bp) as seed for iteration {iteration+1}")

        # ── Move to next iteration ─────────────────────────────────────────────
        iteration += 1

        if iteration <= MAX_ITERATIONS:
            log(f"Sleeping {SLEEP_BETWEEN}s ({SLEEP_BETWEEN//60} min) before next iteration...")
            time.sleep(SLEEP_BETWEEN)

    # ── Final summary ──────────────────────────────────────────────────────────
    log("")
    log("=" * 60)
    log("AUTO-RESEARCH LOOP COMPLETE")
    log(f"Iterations run:     {iteration - 1}")
    log(f"Best score achieved:{best_score_ever:.4f}")
    log(f"Target score:       {TARGET_SCORE}")
    log(f"Status:             {'TARGET REACHED' if best_score_ever >= TARGET_SCORE else 'MAX ITERATIONS REACHED'}")
    log(f"Best candidate:     outputs/all_time_best_candidate.fasta")
    log(f"Metrics log:        {METRICS_FILE}")
    log(f"All iterations:     outputs/iteration_XX/ folders")
    log("=" * 60)

    # Print metrics summary table
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            metrics = json.load(f)
        log("")
        log("ITERATION SUMMARY:")
        log(f"{'Iter':>4}  {'Score':>8}  {'Passing':>8}  {'Best ID'}")
        log("-" * 55)
        for m in metrics:
            log(f"{m['iteration']:>4}  "
                f"{m['best_score']:>8.4f}  "
                f"{m['candidates_passed']:>8}  "
                f"{m['best_candidate_id'] or 'none'}")


if __name__ == "__main__":
    main()
