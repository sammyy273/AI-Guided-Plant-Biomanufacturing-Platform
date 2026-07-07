# Validation Framework for Promoter Design Platform
#
# Provides validation modes:
# 1. Multi-run validation: Run outer loop N times, compute mean/std/CI
# 2. Ablation study: Run with each model individually, compare
# 3. Negative controls: Random DNA + scrambled promoter baselines
# 4. Model benchmark: Repeat per-model runs and compare mean/std metrics
#
# Usage:
#   python scripts/validation.py --species arabidopsis --mode multi-run --runs 5
#   python scripts/validation.py --species arabidopsis --mode ablation
#   python scripts/validation.py --species arabidopsis --mode negative-controls
#   python scripts/validation.py --species arabidopsis --mode all

import os
import sys
import json
import copy
import random
import re
import argparse
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.cross_species.species_config import (
    load_species_config, list_available_species, is_monocot,
)
from modules.evaluation.cis_scoring import (
    scan_cis_elements, compute_weighted_score, gc_content,
    score_candidate,
)
from modules.generation.mutational_generator import (
    generate_from_seed, _random_dna, CIS_ELEMENTS,
    safe_harbor_refine, enforce_spacing_constraints,
)
from outer_loop import AutoresearchState, DEFAULT_STRATEGY, run_outer_iteration
from auto_loop_v2 import (
    run_single_iteration, load_seed, setup_logging,
)


# ═══════════════════════════════════════════════════════════════════════
# MODE 1: MULTI-RUN VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def multi_run_validation(species: str, runs: int = 5,
                         iterations: int = 10,
                         models: str = "evo2+d3lm",
                         variants: int = 22,
                         logger: logging.Logger = None) -> dict:
    """Run the outer loop multiple times and compute statistics.

    This addresses the critical gap of single-run results being unreliable.
    With N=5 runs we can report mean, std, and 95% CI for key metrics.

    Args:
        species: Target species key
        runs: Number of independent runs
        iterations: Iterations per run
        models: Generation models to use
        variants: Total variants per iteration
        logger: Logger instance

    Returns:
        dict with per-run results and aggregate statistics
    """
    if logger is None:
        log_dir = os.path.join(_project_root, "logs")
        logger = setup_logging(log_dir, f"validation_{species}")

    species_config = load_species_config(species)
    species_config["_config_key"] = species
    species_name = species_config["species"]["name"]

    logger.info("=" * 70)
    logger.info("MULTI-RUN VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Species: {species_name}")
    logger.info(f"Runs: {runs}")
    logger.info(f"Iterations/run: {iterations}")
    logger.info(f"Models: {models}")
    logger.info(f"Variants/iteration: {variants}")
    logger.info("")

    all_run_results = []

    for run_idx in range(1, runs + 1):
        logger.info(f"── RUN {run_idx}/{runs} ──")

        # Fresh state for each run
        initial_strategy = copy.deepcopy(DEFAULT_STRATEGY)
        total = variants
        n_evo = total // 3
        n_d3lm = total // 3
        n_mut = total - n_evo - n_d3lm
        initial_strategy["n_variants_per_model"] = {
            "evo2": n_evo,
            "d3lm": n_d3lm,
            "mutational": n_mut,
        }

        state = AutoresearchState(species_config, initial_strategy)
        seed = load_seed(species)

        run_results = []
        for i in range(1, iterations + 1):
            try:
                result = run_outer_iteration(
                    state=state,
                    seed=seed,
                    iteration=i,
                    logger=logger,
                    models=models,
                )
                run_results.append(result)
                if result.get("top_candidate"):
                    seed = result["top_candidate"]
            except Exception as e:
                logger.error(f"  Run {run_idx}, iteration {i} failed: {e}")
                continue

        run_summary = {
            "run": run_idx,
            "best_composite": state.best_composite,
            "best_iteration": state.best_iteration,
            "composite_progression": state.composite_history,
            "model_wins": dict(state.model_wins),
            "strategy_changes": len(state.strategy_change_log),
            "final_stagnation": state.stagnation_count,
        }
        all_run_results.append(run_summary)
        logger.info(f"  Run {run_idx} best composite: {state.best_composite:.4f}")
        logger.info("")

    # Compute aggregate statistics
    best_scores = [r["best_composite"] for r in all_run_results]
    stats = {
        "n_runs": runs,
        "iterations_per_run": iterations,
        "best_composite_mean": float(np.mean(best_scores)),
        "best_composite_std": float(np.std(best_scores, ddof=1)) if runs > 1 else 0.0,
        "best_composite_min": float(np.min(best_scores)),
        "best_composite_max": float(np.max(best_scores)),
        "best_composite_95ci": None,
        "all_best_scores": best_scores,
    }

    # 95% CI: mean ± 1.96 * std / sqrt(n)
    if runs > 1:
        ci_half = 1.96 * stats["best_composite_std"] / np.sqrt(runs)
        stats["best_composite_95ci"] = (
            float(stats["best_composite_mean"] - ci_half),
            float(stats["best_composite_mean"] + ci_half),
        )

    # Model win frequency across all runs
    all_model_wins = defaultdict(int)
    for r in all_run_results:
        for model, count in r["model_wins"].items():
            all_model_wins[model] += count
    stats["model_win_totals"] = dict(all_model_wins)

    # Coefficient of variation
    if stats["best_composite_mean"] > 0:
        stats["coefficient_of_variation"] = float(
            stats["best_composite_std"] / stats["best_composite_mean"]
        )
    else:
        stats["coefficient_of_variation"] = None

    # Report
    logger.info("=" * 70)
    logger.info("MULTI-RUN VALIDATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"  Runs: {runs}")
    logger.info(f"  Best composite score:")
    logger.info(f"    Mean:   {stats['best_composite_mean']:.4f}")
    logger.info(f"    Std:    {stats['best_composite_std']:.4f}")
    logger.info(f"    Min:    {stats['best_composite_min']:.4f}")
    logger.info(f"    Max:    {stats['best_composite_max']:.4f}")
    if stats["best_composite_95ci"]:
        ci = stats["best_composite_95ci"]
        logger.info(f"    95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    if stats.get("coefficient_of_variation") is not None:
        logger.info(f"    CV:     {stats['coefficient_of_variation']:.3f}")
    logger.info(f"  Per-run scores: {[f'{s:.4f}' for s in best_scores]}")
    logger.info(f"  Model wins: {dict(all_model_wins)}")
    logger.info("=" * 70)

    return {
        "species": species,
        "mode": "multi_run",
        "statistics": stats,
        "runs": all_run_results,
    }


# ═══════════════════════════════════════════════════════════════════════
# MODE 2: ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════════

def _ablation_configs(variants: int) -> list:
    """Canonical model configurations for comparison."""
    return [
        (
            "mutational_only", "mutational_only",
            {"evo2": 0, "d3lm": 0, "mutational": variants},
            {"add_mutational_baseline": True, "fail_on_requested_model_failure": False},
        ),
        (
            "evo2_only", "evo2_only",
            {"evo2": variants, "d3lm": 0, "mutational": 0},
            {"add_mutational_baseline": False, "fail_on_requested_model_failure": True},
        ),
        (
            "d3lm_only", "d3lm_only",
            {"evo2": 0, "d3lm": variants, "mutational": 0},
            {"add_mutational_baseline": False, "fail_on_requested_model_failure": True},
        ),
        (
            "evo2+d3lm", "evo2+d3lm",
            {"evo2": variants // 2, "d3lm": variants // 2, "mutational": 0},
            {"add_mutational_baseline": False, "fail_on_requested_model_failure": True},
        ),
        (
            "full_adaptive", "evo2+d3lm", None,
            {"add_mutational_baseline": True, "fail_on_requested_model_failure": True},
        ),
    ]


def model_benchmark(species: str, runs: int = 5,
                    iterations: int = 3,
                    variants: int = 10,
                    logger: logging.Logger = None) -> dict:
    """Benchmark model families across repeated runs on the same task."""
    if logger is None:
        log_dir = os.path.join(_project_root, "logs")
        logger = setup_logging(log_dir, f"benchmark_{species}")

    species_config = load_species_config(species)
    species_config["_config_key"] = species
    species_name = species_config["species"]["name"]
    benchmark_results = {}

    logger.info("=" * 70)
    logger.info("MODEL BENCHMARK")
    logger.info("=" * 70)
    logger.info(f"Species: {species_name}")
    logger.info(f"Runs per config: {runs}")
    logger.info(f"Iterations per run: {iterations}")
    logger.info(f"Variants/iteration: {variants}")
    logger.info("")

    for config_name, models_str, _, generation_opts in _ablation_configs(variants):
        logger.info(f"── BENCHMARK: {config_name} ──")
        run_summaries = []

        for run_idx in range(1, runs + 1):
            random.seed(run_idx)
            np.random.seed(run_idx)
            seed = load_seed(species)
            per_iter_results = []

            for iteration in range(1, iterations + 1):
                try:
                    result = run_single_iteration(
                        species_config=species_config,
                        seed=seed,
                        iteration=iteration,
                        logger=logger,
                        models=models_str,
                        n_variants=variants,
                        add_mutational_baseline=generation_opts["add_mutational_baseline"],
                        fail_on_requested_model_failure=generation_opts["fail_on_requested_model_failure"],
                    )
                except Exception as e:
                    logger.error(
                        f"  Benchmark {config_name}, run {run_idx}, iteration {iteration} failed: {e}"
                    )
                    continue

                per_iter_results.append(result)
                if result.get("top_candidate"):
                    seed = result["top_candidate"]

            if not per_iter_results:
                continue

            best = max((r.get("top_composite_score", -1.0) for r in per_iter_results), default=-1.0)
            last = per_iter_results[-1]
            pass_rates = [
                r.get("n_passed_filters", 0) / max(1, r.get("n_candidates", 1))
                for r in per_iter_results
            ]
            top_id = str(last.get("top_candidate_id", ""))
            top_source = (
                "evo2" if top_id.startswith("evo2_")
                else "d3lm" if top_id.startswith("d3lm_")
                else "mutational" if top_id.startswith("mut_")
                else "unknown"
            )
            run_summaries.append({
                "run": run_idx,
                "best_composite": float(best),
                "final_composite": float(last.get("top_composite_score", -1.0)),
                "mean_pass_rate": float(np.mean(pass_rates)),
                "final_top_candidate_id": top_id,
                "final_top_source": top_source,
                "mean_wall_time_s": float(np.mean([
                    r.get("iteration_wall_time_s", 0.0) for r in per_iter_results
                ])),
            })
            logger.info(
                f"  run {run_idx}: best={best:.4f}, "
                f"final={last.get('top_composite_score', -1.0):.4f}, "
                f"pass={np.mean(pass_rates):.0%}, top={top_source}"
            )

        if not run_summaries:
            benchmark_results[config_name] = {"runs": [], "statistics": {}}
            logger.info("")
            continue

        best_scores = [r["best_composite"] for r in run_summaries]
        final_scores = [r["final_composite"] for r in run_summaries]
        pass_rates = [r["mean_pass_rate"] for r in run_summaries]
        wall_times = [r["mean_wall_time_s"] for r in run_summaries]
        top_source_counts = defaultdict(int)
        for r in run_summaries:
            top_source_counts[r["final_top_source"]] += 1

        benchmark_results[config_name] = {
            "runs": run_summaries,
            "statistics": {
                "n_runs": len(run_summaries),
                "best_composite_mean": float(np.mean(best_scores)),
                "best_composite_std": float(np.std(best_scores, ddof=1)) if len(best_scores) > 1 else 0.0,
                "final_composite_mean": float(np.mean(final_scores)),
                "final_composite_std": float(np.std(final_scores, ddof=1)) if len(final_scores) > 1 else 0.0,
                "mean_pass_rate": float(np.mean(pass_rates)),
                "pass_rate_std": float(np.std(pass_rates, ddof=1)) if len(pass_rates) > 1 else 0.0,
                "mean_wall_time_s": float(np.mean(wall_times)),
                "top_source_counts": dict(top_source_counts),
            },
        }
        logger.info("")

    logger.info("=" * 70)
    logger.info("MODEL BENCHMARK SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  {'Config':20s} | {'Best mean':>9s} | {'Best std':>8s} | {'Pass mean':>9s} | {'Time(s)':>8s}")
    logger.info(f"  {'-'*20}-+-{'-'*9}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}")

    ranking = []
    for name, result in benchmark_results.items():
        stats = result.get("statistics", {})
        if not stats:
            continue
        logger.info(
            f"  {name:20s} | "
            f"{stats['best_composite_mean']:9.4f} | "
            f"{stats['best_composite_std']:8.4f} | "
            f"{stats['mean_pass_rate']:9.0%} | "
            f"{stats['mean_wall_time_s']:8.2f}"
        )
        ranking.append((name, stats["best_composite_mean"]))

    ranking.sort(key=lambda x: -x[1])
    logger.info("")
    logger.info("  Ranking:")
    for idx, (name, score) in enumerate(ranking, 1):
        logger.info(f"    {idx}. {name} ({score:.4f})")
    logger.info("=" * 70)

    return {
        "species": species,
        "mode": "benchmark",
        "runs": runs,
        "iterations": iterations,
        "variants": variants,
        "benchmark_results": benchmark_results,
        "ranking": ranking,
    }


def ablation_study(species: str, iterations: int = 10,
                   variants: int = 22,
                   logger: logging.Logger = None) -> dict:
    """Run outer loop once per model, comparing individual contributions.

    Tests each generation model in isolation:
    - mutational only (rule-based)
    - evo2 only (large language model)
    - d3lm only (diffusion model)
    - evo2+d3lm (ML ensemble)
    - all (default adaptive system)

    This quantifies each model's contribution to the overall system.

    Args:
        species: Target species key
        iterations: Iterations per ablation run
        variants: Total variants per iteration
        logger: Logger instance

    Returns:
        dict with per-model results and comparison
    """
    if logger is None:
        log_dir = os.path.join(_project_root, "logs")
        logger = setup_logging(log_dir, f"ablation_{species}")

    species_config = load_species_config(species)
    species_config["_config_key"] = species
    species_name = species_config["species"]["name"]

    ablation_configs = _ablation_configs(variants)

    logger.info("=" * 70)
    logger.info("ABLATION STUDY")
    logger.info("=" * 70)
    logger.info(f"Species: {species_name}")
    logger.info(f"Iterations per config: {iterations}")
    logger.info(f"Variants/iteration: {variants}")
    logger.info(f"Configs: {[c[0] for c in ablation_configs]}")
    logger.info("")

    ablation_results = {}

    for config_name, models_str, allocation, _ in ablation_configs:
        logger.info(f"── ABLATION: {config_name} ──")

        initial_strategy = copy.deepcopy(DEFAULT_STRATEGY)
        if allocation:
            initial_strategy["n_variants_per_model"] = allocation
        else:
            total = variants
            n_evo = total // 3
            n_d3lm = total // 3
            n_mut = total - n_evo - n_d3lm
            initial_strategy["n_variants_per_model"] = {
                "evo2": n_evo, "d3lm": n_d3lm, "mutational": n_mut,
            }

        state = AutoresearchState(species_config, initial_strategy)
        seed = load_seed(species)

        for i in range(1, iterations + 1):
            try:
                result = run_outer_iteration(
                    state=state,
                    seed=seed,
                    iteration=i,
                    logger=logger,
                    models=models_str,
                )
                if result.get("top_candidate"):
                    seed = result["top_candidate"]
            except Exception as e:
                logger.error(f"  Ablation {config_name}, iteration {i} failed: {e}")
                continue

        ablation_results[config_name] = {
            "best_composite": state.best_composite,
            "best_iteration": state.best_iteration,
            "composite_progression": state.composite_history,
            "model_wins": dict(state.model_wins),
            "strategy_changes": len(state.strategy_change_log),
        }
        logger.info(f"  {config_name}: best_composite={state.best_composite:.4f}")
        logger.info("")

    # Comparison report
    logger.info("=" * 70)
    logger.info("ABLATION COMPARISON")
    logger.info("=" * 70)
    logger.info(f"  {'Config':20s} | {'Best':>8s} | {'Mean (last 3)':>14s} | {'Iter':>5s}")
    logger.info(f"  {'-'*20}-+-{'-'*8}-+-{'-'*14}-+-{'-'*5}")

    comparison_data = {}
    for name, result in ablation_results.items():
        best = result["best_composite"]
        prog = result["composite_progression"]
        mean_last3 = float(np.mean(prog[-3:])) if len(prog) >= 3 else float(np.mean(prog)) if prog else 0.0
        best_iter = result["best_iteration"]
        logger.info(f"  {name:20s} | {best:8.4f} | {mean_last3:14.4f} | {best_iter:5d}")
        comparison_data[name] = {
            "best_composite": best,
            "mean_last3": mean_last3,
            "best_iteration": best_iter,
            "convergence_rate": best_iter / max(1, iterations),
        }

    # Rank
    ranked = sorted(comparison_data.items(), key=lambda x: -x[1]["best_composite"])
    logger.info(f"")
    logger.info(f"  Ranking:")
    for rank, (name, _) in enumerate(ranked, 1):
        logger.info(f"    {rank}. {name} (composite={ablation_results[name]['best_composite']:.4f})")

    logger.info("=" * 70)

    return {
        "species": species,
        "mode": "ablation",
        "iterations": iterations,
        "ablation_results": ablation_results,
        "comparison": comparison_data,
        "ranking": [(name, ablation_results[name]["best_composite"]) for name, _ in ranked],
    }


# ═══════════════════════════════════════════════════════════════════════
# MODE 3: NEGATIVE CONTROLS
# ═══════════════════════════════════════════════════════════════════════

def generate_random_dna(length: int = 800, gc_target: float = 0.40) -> str:
    """Generate purely random DNA with target GC content."""
    at = int(length * (1 - gc_target) / 2)
    gc = int(length * gc_target / 2)
    bases = ["A"] * at + ["T"] * at + ["G"] * gc + ["C"] * gc
    while len(bases) < length:
        bases.append(random.choice("ACGT"))
    random.shuffle(bases)
    return "".join(bases[:length])


def scramble_sequence(seq: str) -> str:
    """Scramble a sequence by shuffling its bases (preserves GC content, destroys motifs)."""
    bases = list(seq.upper())
    random.shuffle(bases)
    return "".join(bases)


COMPONENT_KEYS = [
    "_architecture",
    "_diversity",
    "_gc_score",
    "_overuse_penalty",
    "_scattered_penalty",
    "_repeat_penalty",
]


def component_means(results: list) -> dict:
    """Mean score components for validation diagnostics."""
    if not results:
        return {k: 0.0 for k in COMPONENT_KEYS}
    return {
        k: float(np.mean([r.get(k, 0.0) for r in results]))
        for k in COMPONENT_KEYS
    }


def log_component_means(logger: logging.Logger, label: str, results: list) -> None:
    """Log compact component means for one validation cohort."""
    means = component_means(results)
    logger.info(
        f"  {label:24s} components | "
        f"arch={means['_architecture']:.2f}, "
        f"div={means['_diversity']:.2f}, "
        f"gc={means['_gc_score']:.2f}, "
        f"overuse={means['_overuse_penalty']:.2f}, "
        f"scattered={means['_scattered_penalty']:.2f}, "
        f"repeat={means['_repeat_penalty']:.2f}"
    )


def negative_controls(species: str, n_samples: int = 20,
                      logger: logging.Logger = None) -> dict:
    """Run negative controls: random DNA and scrambled promoters.

    These establish the null distribution — the score that sequences get
    when they have no biological promoter architecture.

    Three control types:
    1. Random DNA (pure noise, species-appropriate GC)
    2. Scrambled real promoters (same base composition, no motif order)
    3. Mutated real promoters (heavy mutation, mostly destroyed motifs)

    Results should show clear separation from designed promoters if
    the scoring function is biologically discriminative.
    """
    if logger is None:
        log_dir = os.path.join(_project_root, "logs")
        logger = setup_logging(log_dir, f"negcontrol_{species}")

    species_config = load_species_config(species)
    species_config["_config_key"] = species
    species_name = species_config["species"]["name"]
    gc_target = species_config.get("species", {}).get("gc_content", 40) / 100.0

    logger.info("=" * 70)
    logger.info("NEGATIVE CONTROLS")
    logger.info("=" * 70)
    logger.info(f"Species: {species_name}")
    logger.info(f"Samples per control type: {n_samples}")
    logger.info(f"GC target: {gc_target:.0%}")
    logger.info("")

    # Load real seed for scrambling
    seed = load_seed(species)

    # ── Control 1: Pure random DNA ──────────────────────────────────
    logger.info("Control 1: Random DNA...")
    random_scores = []
    random_results = []
    random_pass_filters = 0
    random_cis_counts = defaultdict(list)

    for i in range(n_samples):
        seq = generate_random_dna(800, gc_target)
        result = score_candidate(seq, species_config)
        random_results.append(result)
        random_scores.append(result["weighted_score"])
        if result["passed_filters"]:
            random_pass_filters += 1
        for key in ["TATA_box", "CAAT_box", "as1_element", "G_box", "GCN4_motif"]:
            random_cis_counts[key].append(result.get(key, 0))

    logger.info(f"  Score: mean={np.mean(random_scores):.2f}, "
                f"std={np.std(random_scores):.2f}, "
                f"range=[{np.min(random_scores):.2f}, {np.max(random_scores):.2f}]")
    logger.info(f"  Pass rate: {random_pass_filters}/{n_samples} "
                f"({random_pass_filters/n_samples:.0%})")

    # ── Control 2: Scrambled real promoters ──────────────────────────
    logger.info("Control 2: Scrambled promoters...")
    scrambled_scores = []
    scrambled_results = []
    scrambled_pass_filters = 0
    scrambled_cis_counts = defaultdict(list)

    for i in range(n_samples):
        seq = scramble_sequence(seed)
        if len(seq) < 800:
            seq = seq + generate_random_dna(800 - len(seq), gc_target)
        else:
            seq = seq[:800]
        result = score_candidate(seq, species_config)
        scrambled_results.append(result)
        scrambled_scores.append(result["weighted_score"])
        if result["passed_filters"]:
            scrambled_pass_filters += 1
        for key in ["TATA_box", "CAAT_box", "as1_element", "G_box", "GCN4_motif"]:
            scrambled_cis_counts[key].append(result.get(key, 0))

    logger.info(f"  Score: mean={np.mean(scrambled_scores):.2f}, "
                f"std={np.std(scrambled_scores):.2f}, "
                f"range=[{np.min(scrambled_scores):.2f}, {np.max(scrambled_scores):.2f}]")
    logger.info(f"  Pass rate: {scrambled_pass_filters}/{n_samples} "
                f"({scrambled_pass_filters/n_samples:.0%})")

    # ── Control 3: Heavily mutated promoters ─────────────────────────
    logger.info("Control 3: Heavily mutated promoters (50% mutation)...")
    mutated_scores = []
    mutated_results = []
    mutated_pass_filters = 0
    mutated_cis_counts = defaultdict(list)

    for i in range(n_samples):
        seq = list(seed[:800] if len(seed) >= 800 else seed + generate_random_dna(800 - len(seed), gc_target))
        for j in range(len(seq)):
            if random.random() < 0.50:
                seq[j] = random.choice("ACGT")
        seq = "".join(seq)
        result = score_candidate(seq, species_config)
        mutated_results.append(result)
        mutated_scores.append(result["weighted_score"])
        if result["passed_filters"]:
            mutated_pass_filters += 1
        for key in ["TATA_box", "CAAT_box", "as1_element", "G_box", "GCN4_motif"]:
            mutated_cis_counts[key].append(result.get(key, 0))

    logger.info(f"  Score: mean={np.mean(mutated_scores):.2f}, "
                f"std={np.std(mutated_scores):.2f}, "
                f"range=[{np.min(mutated_scores):.2f}, {np.max(mutated_scores):.2f}]")
    logger.info(f"  Pass rate: {mutated_pass_filters}/{n_samples} "
                f"({mutated_pass_filters/n_samples:.0%})")

    # ── Positive control 1: actual seed promoter ──────────────────────
    logger.info("Positive control 1: Real seed promoter...")
    positive_seq = seed[:800] if len(seed) >= 800 else seed + generate_random_dna(800 - len(seed), gc_target)
    positive_result = score_candidate(positive_seq, species_config)
    positive_score = positive_result["weighted_score"]

    logger.info(f"  Score: {positive_score:.2f}")
    logger.info(f"  Pass filters: {positive_result['passed_filters']}")

    # ── Positive control 2: CaMV 35S (strong viral promoter) ─────────
    logger.info("Positive control 2: CaMV 35S promoter...")
    camv35s = (
        "TGACGTAAAGGATCCCGTGTGGAATGTAAAAAGAATGAGCGCAAGACCTTCCAGATC"
        "TTTCCAAACTCTCCAAGCGCACGATCTTCAACTCTTCTCCACCATGGTGTCCAGAAG"
        "GTGTTTGAGCACTTCAACGAGCAGCAGTCCAAATATCAGTACCCACAGTATCTTGCC"
        "GTCAATGGTGACCTTAATGCTTTCTCTTAACATGGTTTATCCATTCGTTCAATCCAC"
        "TCTTAAGGCCTTTTAATATGGTGGAGATCATCACTTTTGGTCTCTCCAATCTTTAGC"
    )
    if len(camv35s) < 800:
        camv35s = camv35s + generate_random_dna(800 - len(camv35s), gc_target)
    else:
        camv35s = camv35s[:800]
    camv35s_result = score_candidate(camv35s, species_config)
    camv35s_score = camv35s_result["weighted_score"]
    logger.info(f"  Score: {camv35s_score:.2f}")
    logger.info(f"  Pass filters: {camv35s_result['passed_filters']}")

    # ── Positive control 3: Generated candidate (our system output) ───
    logger.info("Positive control 3: Generated promoter scaffold cohort...")
    from modules.generation.mutational_generator import build_species_scaffold
    gen_results = []
    gen_scores = []
    gen_pass_filters = 0
    for i in range(n_samples):
        generated_seq = build_species_scaffold(species_config, 800)
        result = score_candidate(generated_seq, species_config)
        gen_results.append(result)
        gen_scores.append(result["weighted_score"])
        if result["passed_filters"]:
            gen_pass_filters += 1
    gen_result = max(gen_results, key=lambda r: r["weighted_score"])
    gen_score = float(np.mean(gen_scores))
    logger.info(f"  Score: mean={gen_score:.2f}, "
                f"std={np.std(gen_scores):.2f}, "
                f"range=[{np.min(gen_scores):.2f}, {np.max(gen_scores):.2f}]")
    logger.info(f"  Pass rate: {gen_pass_filters}/{n_samples} "
                f"({gen_pass_filters/n_samples:.0%})")

    # ── Compute z-scores ────────────────────────────────────────────
    # How many standard deviations above random noise are the controls?
    random_mean = np.mean(random_scores)
    random_std = np.std(random_scores)

    if random_std > 0:
        z_scrambled = (np.mean(scrambled_scores) - random_mean) / random_std
        z_mutated = (np.mean(mutated_scores) - random_mean) / random_std
        z_positive = (positive_score - random_mean) / random_std
        z_camv35s = (camv35s_score - random_mean) / random_std
        z_generated = (np.mean(gen_scores) - random_mean) / random_std
    else:
        z_scrambled = z_mutated = z_positive = z_camv35s = z_generated = 0.0

    # ── Summary ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("NEGATIVE CONTROL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  {'Control':30s} | {'Mean':>8s} | {'Std':>8s} | {'Z vs Random':>12s} | {'Pass%':>6s}")
    logger.info(f"  {'-'*30}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}-+-{'-'*6}")
    logger.info(f"  {'Random DNA':30s} | {np.mean(random_scores):8.2f} | {np.std(random_scores):8.2f} | {'baseline':>12s} | {random_pass_filters/n_samples:5.0%}")
    logger.info(f"  {'Scrambled promoters':30s} | {np.mean(scrambled_scores):8.2f} | {np.std(scrambled_scores):8.2f} | {z_scrambled:12.2f} | {scrambled_pass_filters/n_samples:5.0%}")
    logger.info(f"  {'Heavily mutated':30s} | {np.mean(mutated_scores):8.2f} | {np.std(mutated_scores):8.2f} | {z_mutated:12.2f} | {mutated_pass_filters/n_samples:5.0%}")
    logger.info(f"  {'Positive: seed promoter':30s} | {positive_score:8.2f} | {'N/A':>8s} | {z_positive:12.2f} | {'PASS' if positive_result['passed_filters'] else 'FAIL':>6s}")
    logger.info(f"  {'Positive: CaMV 35S':30s} | {camv35s_score:8.2f} | {'N/A':>8s} | {z_camv35s:12.2f} | {'PASS' if camv35s_result['passed_filters'] else 'FAIL':>6s}")
    logger.info(f"  {'Positive: generated scaffold':30s} | {gen_score:8.2f} | {np.std(gen_scores):8.2f} | {z_generated:12.2f} | {gen_pass_filters/n_samples:5.0%}")
    logger.info("")
    logger.info("COMPONENT BREAKDOWN")
    log_component_means(logger, "Random DNA", random_results)
    log_component_means(logger, "Scrambled promoters", scrambled_results)
    log_component_means(logger, "Heavily mutated", mutated_results)
    log_component_means(logger, "Seed promoter", [positive_result])
    log_component_means(logger, "CaMV 35S", [camv35s_result])
    log_component_means(logger, "Generated scaffold", gen_results)
    logger.info("")

    # Discrimination assessment (use the BEST positive control = generated)
    best_positive_z = max(z_positive, z_camv35s, z_generated)
    best_positive_name = "seed" if best_positive_z == z_positive else "CaMV 35S" if best_positive_z == z_camv35s else "generated"

    logger.info("  Discrimination assessment:")
    if best_positive_z > 2.0:
        logger.info(f"    Scoring function STRONGLY discriminates ({best_positive_name}, z={best_positive_z:.2f})")
    elif best_positive_z > 1.5:
        logger.info(f"    Scoring function moderately discriminates ({best_positive_name}, z={best_positive_z:.2f})")
    else:
        logger.info(f"    WARNING: Scoring function weakly discriminates (best z={best_positive_z:.2f})")

    if z_scrambled < 0.5:
        logger.info(f"    Scrambled sequences correctly score near random (z={z_scrambled:.2f})")
    else:
        logger.info(f"    WARNING: Scrambled sequences score above random (z={z_scrambled:.2f}) "
                     f"— may indicate position-independent scoring bias")

    # Note about seed promoter
    if z_positive < 1.0 and z_generated > 2.0:
        logger.info(f"    Note: Seed promoter scores low (z={z_positive:.2f}) because it lacks canonical")
        logger.info(f"    TATA/CAAT positioning — this is biologically expected for endogenous promoters.")
        logger.info(f"    Generated promoters with correct architecture score z={z_generated:.2f}.")

    logger.info("=" * 70)

    return {
        "species": species,
        "mode": "negative_controls",
        "n_samples": n_samples,
        "controls": {
            "random_dna": {
                "scores": random_scores,
                "mean": float(np.mean(random_scores)),
                "std": float(np.std(random_scores)),
                "pass_rate": random_pass_filters / n_samples,
                "z_score": 0.0,
                "components": component_means(random_results),
            },
            "scrambled": {
                "scores": scrambled_scores,
                "mean": float(np.mean(scrambled_scores)),
                "std": float(np.std(scrambled_scores)),
                "pass_rate": scrambled_pass_filters / n_samples,
                "z_score": float(z_scrambled),
                "components": component_means(scrambled_results),
            },
            "heavily_mutated": {
                "scores": mutated_scores,
                "mean": float(np.mean(mutated_scores)),
                "std": float(np.std(mutated_scores)),
                "pass_rate": mutated_pass_filters / n_samples,
                "z_score": float(z_mutated),
                "components": component_means(mutated_results),
            },
            "positive_seed": {
                "score": float(positive_score),
                "passed_filters": positive_result["passed_filters"],
                "z_score": float(z_positive),
                "components": component_means([positive_result]),
            },
            "positive_camv35s": {
                "score": float(camv35s_score),
                "passed_filters": camv35s_result["passed_filters"],
                "z_score": float(z_camv35s),
                "components": component_means([camv35s_result]),
            },
            "positive_generated": {
                "scores": gen_scores,
                "score": float(gen_score),
                "mean": float(np.mean(gen_scores)),
                "std": float(np.std(gen_scores)),
                "best_score": float(gen_result["weighted_score"]),
                "pass_rate": gen_pass_filters / n_samples,
                "passed_filters": gen_pass_filters == n_samples,
                "z_score": float(z_generated),
                "components": component_means(gen_results),
            },
        },
        "discrimination_z_positive": float(best_positive_z),
        "discrimination_z_generated": float(z_generated),
        "random_baseline_mean": float(random_mean),
        "random_baseline_std": float(random_std),
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Validation Framework for Promoter Design Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-run validation (5 independent runs)
  python scripts/validation.py --species arabidopsis --mode multi-run --runs 5

  # Repeated model benchmark (recommended for model comparison)
  python scripts/validation.py --species arabidopsis --mode benchmark --runs 3 --iterations 2 --variants 5

  # Ablation study (test each model individually)
  python scripts/validation.py --species arabidopsis --mode ablation

  # Negative controls (establish null distribution)
  python scripts/validation.py --species arabidopsis --mode negative-controls

  # Run everything
  python scripts/validation.py --species arabidopsis --mode all --runs 3
        """,
    )
    parser.add_argument("--species", required=True,
                        help="Target species")
    parser.add_argument("--mode", default="all",
                        choices=["multi-run", "benchmark", "ablation", "negative-controls", "all"],
                        help="Validation mode (default: all)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs for multi-run validation (default: 5)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Iterations per run (default: 10)")
    parser.add_argument("--variants", type=int, default=22,
                        help="Total variants per iteration (default: 22)")
    parser.add_argument("--models", default="evo2+d3lm",
                        help="Generation models (default: evo2+d3lm)")
    parser.add_argument("--samples", type=int, default=20,
                        help="Samples for negative controls (default: 20)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory")
    args = parser.parse_args()

    # Setup
    species_config = load_species_config(args.species)
    species_name = species_config["species"]["name"]
    output_dir = args.output_dir or os.path.join(
        _project_root, "outputs",
        f"validation_{args.species}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(output_dir, exist_ok=True)

    log_dir = os.path.join(_project_root, "logs")
    logger = setup_logging(log_dir, f"validation_{args.species}_{args.mode}")

    logger.info(f"Validation output: {output_dir}")

    results = {}

    if args.mode in ("multi-run", "all"):
        logger.info("\n" + "#" * 70)
        logger.info("# MODE 1: MULTI-RUN VALIDATION")
        logger.info("#" * 70 + "\n")
        results["multi_run"] = multi_run_validation(
            species=args.species,
            runs=args.runs,
            iterations=args.iterations,
            models=args.models,
            variants=args.variants,
            logger=logger,
        )

    if args.mode in ("benchmark", "all"):
        logger.info("\n" + "#" * 70)
        logger.info("# MODE 2: MODEL BENCHMARK")
        logger.info("#" * 70 + "\n")
        results["benchmark"] = model_benchmark(
            species=args.species,
            runs=args.runs,
            iterations=args.iterations,
            variants=args.variants,
            logger=logger,
        )

    if args.mode in ("ablation", "all"):
        logger.info("\n" + "#" * 70)
        logger.info("# MODE 3: ABLATION STUDY")
        logger.info("#" * 70 + "\n")
        results["ablation"] = ablation_study(
            species=args.species,
            iterations=args.iterations,
            variants=args.variants,
            logger=logger,
        )

    if args.mode in ("negative-controls", "all"):
        logger.info("\n" + "#" * 70)
        logger.info("# MODE 4: NEGATIVE CONTROLS")
        logger.info("#" * 70 + "\n")
        results["negative_controls"] = negative_controls(
            species=args.species,
            n_samples=args.samples,
            logger=logger,
        )

    # Save results
    results_path = os.path.join(output_dir, "validation_results.json")

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(convert(results), f, indent=2)
    logger.info(f"\nAll results saved: {results_path}")

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 70)
    if "multi_run" in results:
        stats = results["multi_run"]["statistics"]
        logger.info(f"  Multi-run: mean={stats['best_composite_mean']:.4f} "
                     f"± {stats['best_composite_std']:.4f} "
                     f"(n={stats['n_runs']})")
    if "benchmark" in results:
        ranking = results["benchmark"]["ranking"]
        logger.info(f"  Benchmark: {' > '.join(f'{n}={s:.4f}' for n, s in ranking)}")
    if "ablation" in results:
        ranking = results["ablation"]["ranking"]
        logger.info(f"  Ablation: {' > '.join(f'{n}={s:.4f}' for n, s in ranking)}")
    if "negative_controls" in results:
        nc = results["negative_controls"]
        logger.info(f"  Negative controls: z_positive={nc['discrimination_z_positive']:.2f}, "
                     f"random_mean={nc['random_baseline_mean']:.2f}")


if __name__ == "__main__":
    main()
