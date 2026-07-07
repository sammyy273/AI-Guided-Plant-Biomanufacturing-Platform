# Outer Autoresearch Loop — Meta-Learner for Adaptive Promoter Design
#
# Wraps the inner loop (auto_loop_v2.py) and adds a strategy-adaptation layer
# that analyses failure patterns and stagnation to modify generation parameters
# across iterations.
#
# This is the core novelty: the system learns from its own failures and adjusts
# the generation strategy in real-time rather than using fixed parameters.
#
# Usage:
#   python scripts/outer_loop.py --species arabidopsis --iterations 10
#   python scripts/outer_loop.py --species maize --iterations 15 --stagnation-window 4
#   python scripts/outer_loop.py --species nbenthamiana --iterations 20 --models evo2+d3lm

import os
import sys
import json
import time
import copy
import random
import re
import argparse
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np

# ─── Project imports ────────────────────────────────────────────────────
_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _project_root)
# Also add scripts dir so we can import auto_loop_v2 as a module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.cross_species.species_config import (
    load_species_config, list_available_species, is_monocot,
)
from modules.evaluation.cis_scoring import (
    scan_cis_elements, score_candidate, gc_content,
)
from modules.generation.mutational_generator import (
    generate_from_seed,
    _apply_point_mutations,
    _insert_cis_element,
    _find_cis_positions,
    _random_dna,
    CIS_ELEMENTS,
    build_species_scaffold,
    safe_harbor_refine,
    enforce_spacing_constraints,
)

# Import the inner loop's iteration runner
import auto_loop_v2 as inner_loop_module
from auto_loop_v2 import (
    run_single_iteration,
    load_seed,
    setup_logging,
)


# ═══════════════════════════════════════════════════════════════════════
# DEFAULT STRATEGY
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_STRATEGY = {
    # Evo2 prompt suffix: appended to the seed before Evo2 generation
    # Empty means no additional constraint
    "evo2_prompt_suffix": "",

    # Evo2 sampling parameters
    "evo2_temperature_range": (0.6, 1.0),
    "evo2_top_k_choices": [3, 4, 5, 6],

    # Mutational generator parameters
    "mutation_rate_range": (0.03, 0.35),  # (refinement_min, exploration_max)
    "cis_insertion_prob": 1.0,            # Probability of forced cis-element insertion
    "gc_bias": None,                       # None = species default, "high" or "low"

    # Variant allocation across models
    "n_variants_per_model": {
        "evo2": 8,
        "d3lm": 8,
        "mutational": 6,
    },

    # Exploration vs refinement ratio (0.0 = all refinement, 1.0 = all exploration)
    "exploration_vs_refinement": 0.20,

    # Whether to force cis-element insertion on every variant
    "force_tata_insertion": True,
    "force_caat_insertion": True,
}


# ═══════════════════════════════════════════════════════════════════════
# AUTORESEARCH STATE — Accumulates knowledge across iterations
# ═══════════════════════════════════════════════════════════════════════

class AutoresearchState:
    """Meta-learner state that tracks failures, model performance, and
    strategy changes across outer-loop iterations.

    Attributes:
        iteration_history: List of per-iteration result dicts
        failure_history: List of failure_reasons dicts (one per iteration)
        composite_history: List of top composite scores (one per iteration)
        model_wins: Dict of {model_prefix: count} for which model produced
                    the top candidate
        strategy_history: List of (strategy_dict, iteration) tuples
        stagnation_count: Number of consecutive iterations with no improvement
        best_composite: Best composite score seen so far
        best_strategy: Strategy that produced the best score
        last_strategy_change_iter: Iteration at which strategy was last changed
    """

    def __init__(self, species_config: dict, initial_strategy: dict = None):
        self.species_config = species_config
        self.strategy = copy.deepcopy(initial_strategy or DEFAULT_STRATEGY)

        self.iteration_history = []
        self.failure_history = []
        self.composite_history = []
        self.model_wins = defaultdict(int)
        self.strategy_history = [(copy.deepcopy(self.strategy), 0)]
        self.strategy_change_log = []  # (iteration, what_changed, reason)

        self.best_composite = -1.0
        self.best_strategy = copy.deepcopy(self.strategy)
        self.best_seed = None
        self.best_iteration = 0
        self.stagnation_count = 0
        self.last_strategy_change_iter = 0

        # Track silencing risk across iterations for adaptive response
        self.silencing_history = []  # List of top candidate silencing risk per iteration

        # GC content range from species config
        hard_filter = species_config.get("hard_filter", {})
        self.min_gc = hard_filter.get("min_gc_pct", 30)
        self.max_gc = hard_filter.get("max_gc_pct", 60)

    # ─── Recording ───────────────────────────────────────────────────

    def record_iteration(self, result: dict):
        """Record results from one inner-loop iteration."""
        self.iteration_history.append(result)

        # Track failures (normalised by total candidates for fair comparison)
        failures = result.get("failure_reasons", {})
        n_candidates = result.get("n_candidates", 1)
        # Store as rates (failures / total_candidates) for normalised analysis
        normalised = {
            ftype: count / max(1, n_candidates)
            for ftype, count in failures.items()
        }
        normalised["_n_candidates"] = n_candidates  # Keep raw count for reference
        normalised["_raw"] = dict(failures)
        self.failure_history.append(normalised)

        # Track composite score
        composite = result.get("top_composite_score", -1.0)
        self.composite_history.append(composite)

        # Track which model produced the top candidate
        top_id = result.get("top_candidate_id", "")
        if top_id.startswith("evo2_"):
            self.model_wins["evo2"] += 1
        elif top_id.startswith("d3lm_"):
            self.model_wins["d3lm"] += 1
        elif top_id.startswith("mut_"):
            self.model_wins["mutational"] += 1

        # Track best
        if composite > self.best_composite:
            self.best_composite = composite
            self.best_strategy = copy.deepcopy(self.strategy)
            self.best_seed = result.get("top_candidate")
            self.best_iteration = result.get("iteration", 0)
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

        # Track silencing risk of top candidate (for adaptive response)
        silencing = result.get("top_silencing_risk", None)
        if silencing is not None:
            self.silencing_history.append(silencing)

    # ─── Stagnation Detection ────────────────────────────────────────

    def detect_stagnation(self, window: int = 3) -> bool:
        """Return True if no improvement in the last `window` iterations.

        Stagnation means the composite score has not improved (strictly
        increased) for `window` consecutive iterations.
        """
        if len(self.composite_history) < window:
            return False

        recent = self.composite_history[-window:]
        # Stagnation: no improvement means the max of recent <= best seen
        # before this window started
        return max(recent) <= self.best_composite and self.stagnation_count >= window

    # ─── Failure Analysis ────────────────────────────────────────────

    def analyse_failures(self, window: int = 5) -> dict:
        """Identify dominant failure mode from the last N iterations.

        Uses normalised failure rates (failures / total_candidates) so that
        iterations with more candidates don't dominate the analysis.

        Returns:
            dict with:
                dominant_failure: str or None (the most common failure type)
                failure_totals: dict of {failure_type: total_count} over window (raw)
                failure_rates: dict of {failure_type: avg_rate} over window (normalised)
                dominant_ratio: float (fraction of failures that are dominant)
        """
        if not self.failure_history:
            return {
                "dominant_failure": None,
                "failure_totals": {},
                "failure_rates": {},
                "dominant_ratio": 0.0,
            }

        # Aggregate normalised rates over the last `window` iterations
        recent_failures = self.failure_history[-window:]
        rate_totals = defaultdict(float)
        raw_totals = defaultdict(int)
        n_iters = 0

        for fdict in recent_failures:
            n_iters += 1
            raw = fdict.get("_raw", fdict)
            for ftype, count in raw.items():
                if ftype.startswith("_"):
                    continue
                raw_totals[ftype] += count
                rate_totals[ftype] += fdict.get(ftype, 0)

        if not rate_totals or sum(rate_totals.values()) == 0:
            return {
                "dominant_failure": None,
                "failure_totals": dict(raw_totals),
                "failure_rates": {},
                "dominant_ratio": 0.0,
            }

        # Dominant = highest average failure rate
        avg_rates = {k: v / n_iters for k, v in rate_totals.items()}
        dominant = max(avg_rates, key=avg_rates.get)
        total_rate = sum(avg_rates.values())

        return {
            "dominant_failure": dominant,
            "failure_totals": dict(raw_totals),
            "failure_rates": avg_rates,
            "dominant_ratio": avg_rates[dominant] / total_rate if total_rate > 0 else 0,
        }

    # ─── Model Performance Analysis ──────────────────────────────────

    def analyse_model_performance(self) -> dict:
        """Analyse which generation models consistently produce top candidates.

        Returns:
            dict with model allocation recommendations
        """
        total_wins = sum(self.model_wins.values())
        if total_wins == 0:
            return {"best_model": None, "allocation": None}

        # All models that should exist (from initial strategy)
        all_models = list(self.strategy["n_variants_per_model"].keys())

        # Sort by wins — models with 0 wins still included
        ranked = sorted(
            [(m, self.model_wins.get(m, 0)) for m in all_models],
            key=lambda x: -x[1],
        )
        best_model = ranked[0][0]

        # Allocation: best gets 45%, second gets 30%, third gets 25%
        # ALL models are preserved — none dropped
        total_variants = sum(self.strategy["n_variants_per_model"].values())
        fractions = [0.45, 0.30, 0.25]

        allocation = {}
        for i, (model, _wins) in enumerate(ranked):
            frac = fractions[i] if i < len(fractions) else 0.15
            allocation[model] = max(2, int(total_variants * frac))

        # Ensure all models present with minimum allocation
        for model in all_models:
            if model not in allocation:
                allocation[model] = 2

        # Adjust to match total exactly
        allocated = sum(allocation.values())
        if allocated != total_variants:
            # Add/subtract difference from best model
            allocation[best_model] += total_variants - allocated

        return {
            "best_model": best_model,
            "model_wins": dict(self.model_wins),
            "allocation": allocation,
        }

    # ─── Strategy Proposal ───────────────────────────────────────────

    def propose_strategy_change(self, failure_analysis: dict,
                                 model_analysis: dict,
                                 logger: logging.Logger) -> dict:
        """Propose a modified strategy based on failure and model analysis.

        Returns:
            dict of {parameter_name: new_value} for parameters to change.
            Only includes parameters that would actually change (no no-ops).
        """
        changes = {}
        dominant = failure_analysis["dominant_failure"]
        ratio = failure_analysis["dominant_ratio"]
        best_model = model_analysis["best_model"]

        # Hard limits on parameters (guardrails)
        MAX_EXPLORATION = 0.50      # Don't exceed 50% exploration
        MAX_MUTATION_RATE = 0.30    # Don't exceed 30% mutation rate
        MAX_TEMPERATURE = 1.2       # Don't exceed 1.2 temperature
        MIN_MUTATION_RATE = 0.03    # Don't go below 3%

        # ── Failure-driven strategy changes ──────────────────────────

        if dominant == "no_TATA" and ratio > 0.25:
            # Only propose if not already at max
            current_cis = self.strategy["cis_insertion_prob"]
            if current_cis < 1.0:
                changes["cis_insertion_prob"] = min(1.0, current_cis + 0.2)
            if not self.strategy.get("force_tata_insertion"):
                changes["force_tata_insertion"] = True
            logger.info(f"  Strategy: Dominant failure is no_TATA ({ratio:.0%}). "
                        f"Increasing TATA insertion rate.")

        elif dominant == "no_CAAT" and ratio > 0.25:
            current_cis = self.strategy["cis_insertion_prob"]
            if current_cis < 1.0:
                changes["cis_insertion_prob"] = min(1.0, current_cis + 0.2)
            if not self.strategy.get("force_caat_insertion"):
                changes["force_caat_insertion"] = True
            logger.info(f"  Strategy: Dominant failure is no_CAAT ({ratio:.0%}). "
                        f"Increasing CAAT insertion rate.")

        elif dominant == "gc_low" and ratio > 0.25:
            if self.strategy.get("gc_bias") != "high":
                changes["gc_bias"] = "high"
                logger.info(f"  Strategy: Dominant failure is gc_low ({ratio:.0%}). "
                            f"Biasing mutations toward G/C nucleotides.")

        elif dominant == "gc_high" and ratio > 0.25:
            if self.strategy.get("gc_bias") != "low":
                changes["gc_bias"] = "low"
                logger.info(f"  Strategy: Dominant failure is gc_high ({ratio:.0%}). "
                            f"Biasing mutations toward A/T nucleotides.")

        elif dominant == "too_many_TATA" and ratio > 0.25:
            current_cis = self.strategy["cis_insertion_prob"]
            if current_cis > 0.3:
                changes["cis_insertion_prob"] = max(0.3, current_cis - 0.2)
                logger.info(f"  Strategy: Dominant failure is too_many_TATA ({ratio:.0%}). "
                            f"Reducing cis-element insertion rate.")

        # ── Silencing-risk-driven strategy changes ────────────────────
        # If top candidates consistently have high silencing risk,
        # adjust generation to reduce CpG density and repeats.
        # This makes the outer loop genome-aware, not just score-aware.

        if len(self.silencing_history) >= 3:
            recent_silencing = self.silencing_history[-3:]
            mean_silencing = sum(recent_silencing) / len(recent_silencing)
            if mean_silencing > 0.5:
                # HIGH silencing risk: reduce GC, increase AT bias
                if self.strategy.get("gc_bias") != "low":
                    changes["gc_bias"] = "low"
                # Reduce mutation rate range to limit introducing CpG
                low, high = self.strategy["mutation_rate_range"]
                if high > 0.20:
                    changes["mutation_rate_range"] = (
                        max(MIN_MUTATION_RATE, low),
                        min(MAX_MUTATION_RATE, 0.20),
                    )
                logger.info(f"  Strategy: High silencing risk (mean={mean_silencing:.3f}). "
                            f"Reducing GC bias and mutation rate to lower CpG density.")

            elif mean_silencing > 0.3:
                # MODERATE silencing risk: mild GC adjustment
                if self.strategy.get("gc_bias") is None:
                    changes["gc_bias"] = "low"
                    logger.info(f"  Strategy: Moderate silencing risk (mean={mean_silencing:.3f}). "
                                f"Mildly biasing toward AT to reduce methylation risk.")

        # ── Stagnation-driven strategy changes ────────────────────────

        if self.stagnation_count >= 3:
            current_explore = self.strategy["exploration_vs_refinement"]
            if current_explore < MAX_EXPLORATION:
                changes["exploration_vs_refinement"] = min(
                    MAX_EXPLORATION, current_explore + 0.10
                )
            low, high = self.strategy["mutation_rate_range"]
            if high < MAX_MUTATION_RATE:
                changes["mutation_rate_range"] = (
                    max(MIN_MUTATION_RATE, low),
                    min(MAX_MUTATION_RATE, high + 0.03),
                )
            t_low, t_high = self.strategy["evo2_temperature_range"]
            if t_high < MAX_TEMPERATURE:
                changes["evo2_temperature_range"] = (
                    t_low, min(MAX_TEMPERATURE, t_high + 0.05)
                )
            if changes:
                logger.info(f"  Strategy: Stagnation detected ({self.stagnation_count} iters). "
                            f"Widening parameters within safe limits.")

        # ── Model-performance-driven allocation changes ───────────────

        if best_model and model_analysis.get("allocation"):
            changes["n_variants_per_model"] = model_analysis["allocation"]
            logger.info(f"  Strategy: Reallocating variants toward {best_model} "
                        f"(wins: {dict(self.model_wins)})")

        return changes

    # ─── Strategy Application ────────────────────────────────────────

    def apply_strategy(self, changes: dict, iteration: int,
                       logger: logging.Logger):
        """Apply proposed strategy changes to the current strategy.

        Records what was changed for later reporting.
        Skips no-op changes where old value equals new value.
        """
        if not changes:
            return

        actual_changes = 0
        for param, value in changes.items():
            old_value = self.strategy.get(param)
            # Skip no-op changes (same value)
            if old_value == value:
                continue

            self.strategy[param] = value
            reason = "failure_analysis" if param in (
                "cis_insertion_prob", "gc_bias",
                "force_tata_insertion", "force_caat_insertion",
                "evo2_prompt_suffix",
            ) else "stagnation" if param in (
                "exploration_vs_refinement", "mutation_rate_range",
                "evo2_temperature_range",
            ) else "model_reallocation"

            self.strategy_change_log.append({
                "iteration": iteration,
                "parameter": param,
                "old_value": old_value,
                "new_value": value,
                "reason": reason,
            })
            logger.info(f"    {param}: {old_value} -> {value} ({reason})")
            actual_changes += 1

        if actual_changes > 0:
            self.strategy_history.append((copy.deepcopy(self.strategy), iteration))
            self.last_strategy_change_iter = iteration

    # ─── Reporting ───────────────────────────────────────────────────

    def generate_improvement_report(self, logger: logging.Logger) -> str:
        """Generate a before/after report for the last strategy change.

        Returns:
            Formatted report string
        """
        if not self.strategy_change_log:
            return "No strategy changes made yet."

        # Find changes since last strategy shift
        recent_changes = [c for c in self.strategy_change_log
                          if c["iteration"] >= self.last_strategy_change_iter]

        if not recent_changes:
            return "No recent strategy changes."

        lines = []
        lines.append("=" * 60)
        lines.append("STRATEGY CHANGE REPORT")
        lines.append("=" * 60)

        for change in recent_changes:
            lines.append(
                f"  Iter {change['iteration']:3d} | "
                f"{change['parameter']:30s} | "
                f"{change['old_value']} -> {change['new_value']} "
                f"({change['reason']})"
            )

        # Score comparison: before vs after last strategy change
        change_iter = self.last_strategy_change_iter
        before_scores = [r.get("top_composite_score", 0)
                         for r in self.iteration_history
                         if r.get("iteration", 0) < change_iter]
        after_scores = [r.get("top_composite_score", 0)
                        for r in self.iteration_history
                        if r.get("iteration", 0) >= change_iter]

        if before_scores and after_scores:
            before_avg = np.mean(before_scores[-3:])  # Last 3 before change
            after_avg = np.mean(after_scores)          # All after change
            improvement = after_avg - before_avg

            lines.append("")
            lines.append(f"  Score before change (avg last 3): {before_avg:.4f}")
            lines.append(f"  Score after change  (avg all):    {after_avg:.4f}")
            lines.append(f"  Improvement: {improvement:+.4f} "
                         f"({'IMPROVED' if improvement > 0 else 'WORSENED'})")

        lines.append(f"  Best composite score overall: {self.best_composite:.4f} "
                     f"(iteration {self.best_iteration})")
        lines.append("=" * 60)

        report = "\n".join(lines)
        logger.info(report)
        return report

    def generate_final_summary(self) -> str:
        """Generate a final summary of the entire outer loop run."""
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("OUTER LOOP FINAL SUMMARY")
        lines.append("=" * 70)

        lines.append(f"  Total iterations: {len(self.iteration_history)}")
        completed = [r for r in self.iteration_history if r.get("status") == "complete"]
        lines.append(f"  Completed iterations: {len(completed)}")

        if self.composite_history:
            lines.append(f"  Composite score progression:")
            for i, score in enumerate(self.composite_history):
                marker = " <-- BEST" if score == self.best_composite else ""
                lines.append(f"    Iter {i+1:3d}: {score:.4f}{marker}")

        lines.append(f"  Best composite score: {self.best_composite:.4f} "
                     f"(iteration {self.best_iteration})")

        # Failure summary
        if self.failure_history:
            total_failures = defaultdict(int)
            for fdict in self.failure_history:
                for ftype, count in fdict.items():
                    total_failures[ftype] += count
            lines.append(f"  Failure summary (all iterations):")
            for ftype, count in sorted(total_failures.items(), key=lambda x: -x[1]):
                if count > 0:
                    lines.append(f"    {ftype}: {count}")

        # Model performance
        lines.append(f"  Model wins (top candidate came from):")
        for model, count in sorted(self.model_wins.items(), key=lambda x: -x[1]):
            lines.append(f"    {model}: {count}")

        # Strategy changes
        lines.append(f"  Strategy changes made: {len(self.strategy_change_log)}")
        for change in self.strategy_change_log:
            lines.append(f"    Iter {change['iteration']:3d}: "
                         f"{change['parameter']} ({change['reason']})")

        # Final strategy vs initial
        lines.append(f"  Strategy drift (initial -> final):")
        for key in DEFAULT_STRATEGY:
            initial = DEFAULT_STRATEGY[key]
            current = self.strategy.get(key)
            if initial != current:
                lines.append(f"    {key}: {initial} -> {current}")

        lines.append("=" * 70)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# ADAPTIVE GENERATION — Wraps generators with strategy-controlled params
# ═══════════════════════════════════════════════════════════════════════

def generate_adaptive_mutational(
    seed: str,
    species_config: dict,
    strategy: dict,
    n_variants: int = 6,
    target_length: int = 800,
) -> dict:
    """Generate mutational variants with strategy-adapted parameters.

    This replaces the default generate_from_seed with a version that
    respects the adaptive strategy parameters.
    """
    gc_target = species_config.get("species", {}).get("gc_content", 38) / 100.0

    # Apply GC bias from strategy
    gc_bias = strategy.get("gc_bias")
    if gc_bias == "high":
        gc_target = min(0.65, gc_target + 0.05)
    elif gc_bias == "low":
        gc_target = max(0.25, gc_target - 0.05)

    weights = species_config.get("cis_element_weights", {})
    is_mono = species_config.get("species", {}).get("type") == "monocot"
    exploration_ratio = strategy.get("exploration_vs_refinement", 0.20)
    mutation_low, mutation_high = strategy.get(
        "mutation_rate_range", (0.03, 0.35)
    )
    cis_prob = strategy.get("cis_insertion_prob", 1.0)

    # Pad/trim seed
    if len(seed) < target_length:
        seed = seed + _random_dna(target_length - len(seed), gc_target)
    else:
        seed = seed[:target_length]

    existing = _find_cis_positions(seed)
    candidates = {}

    for i in range(n_variants):
        variant_type = random.random()

        if variant_type < exploration_ratio:
            # EXPLORATION: Heavy mutation in DISTAL regions only,
            # preserving core promoter architecture (TATA zone + CAAT zone).
            # The core promoter (last ~130bp) is protected from heavy mutation
            # so that exploration doesn't destroy promoter architecture.
            rate = random.uniform(
                mutation_low + (mutation_high - mutation_low) * 0.5,
                mutation_high,
            )

            # Define core promoter protection zone (TATA + CAAT regions)
            core_start = max(0, target_length - 130)  # Covers CAAT zone to end
            distal_rate = rate
            core_rate = rate * 0.15  # Much lower mutation in core promoter

            # Apply higher mutation to distal region, lower to core
            seq_list = list(seed)
            for j in range(len(seq_list)):
                if core_start <= j:
                    # Core promoter: minimal mutation
                    if random.random() < core_rate:
                        if random.random() < gc_target:
                            seq_list[j] = random.choice("GC")
                        else:
                            seq_list[j] = random.choice("AT")
                else:
                    # Distal region: full exploration
                    if random.random() < distal_rate:
                        if random.random() < gc_target:
                            seq_list[j] = random.choice("GC")
                        else:
                            seq_list[j] = random.choice("AT")
            seq = "".join(seq_list)

            # Insert cis-elements only in distal regulatory region (NOT in core)
            available_elements = [
                k for k, v in weights.items()
                if v > 0 and k in CIS_ELEMENTS
            ]
            n_to_add = random.randint(2, min(4, len(available_elements)))
            elements_to_add = random.sample(
                available_elements,
                min(n_to_add, len(available_elements))
            )
            for elem_name in elements_to_add:
                if random.random() < cis_prob:
                    # Only insert in distal region (before core promoter)
                    pos = random.randint(50, core_start - 20)
                    elem_seq = random.choice(CIS_ELEMENTS[elem_name])
                    seq = _insert_cis_element(seq, elem_seq, pos)

        elif variant_type < exploration_ratio + 0.50:
            # REFINEMENT: Moderate mutation preserving core elements
            rate = random.uniform(
                mutation_low,
                mutation_low + (mutation_high - mutation_low) * 0.3,
            )
            seq = _apply_point_mutations(seed, rate, gc_target)

            # Restore known cis-elements
            for elem_name, positions in existing.items():
                for start, end in positions:
                    original = seed[start:end]
                    seq = seq[:start] + original + seq[end:]

        else:
            # CROSSOVER: Recombine seed with fresh scaffold
            scaffold = build_species_scaffold(species_config, target_length)
            crossover_point = random.randint(
                target_length // 4, 3 * target_length // 4
            )
            seq = seed[:crossover_point] + scaffold[crossover_point:]

        # Forced cis-element insertion based on strategy
        # Use canonical motifs at exact positions for reliable architecture
        if strategy.get("force_tata_insertion", True):
            if not re.search(r"TATA[AT]A[AT]", seq):
                tata_pos = target_length - 35
                seq = _insert_cis_element(seq, "TATAAAT", tata_pos)

        if strategy.get("force_caat_insertion", True):
            if not re.search(r"CCAAT", seq):
                caat_pos = target_length - 85
                seq = _insert_cis_element(seq, "CCAAT", caat_pos)

        # Too-many-TATA check: if strategy has reduced insertion rate,
        # cull excess TATA boxes by mutating them
        if strategy.get("cis_insertion_prob", 1.0) < 0.5:
            tata_matches = list(re.finditer(r"TATA[AT]A[AT]", seq))
            hard_filter = species_config.get("hard_filter", {})
            max_tata = hard_filter.get("max_tata_boxes", 5)
            if len(tata_matches) > max_tata:
                # Remove excess TATA boxes by replacing them
                for match in tata_matches[max_tata:]:
                    pos = match.start()
                    replacement = "".join(
                        random.choice("GC") for _ in range(len(match.group()))
                    )
                    seq = seq[:pos] + replacement + seq[pos + len(match.group()):]

        # Enforce TATA-CAAT spacing and order (biological constraint)
        seq = enforce_spacing_constraints(seq, target_length)

        # Safe harbor-aware refinement (reduce CpG, break repeats, open chromatin)
        seq = safe_harbor_refine(seq, gc_target)

        # Clean: only ACGT, pad/trim to target length
        seq = "".join(c for c in seq.upper() if c in "ACGT")
        if len(seq) < target_length:
            seq = seq + _random_dna(target_length - len(seq), gc_target)
        else:
            seq = seq[:target_length]

        candidates[f"mut_v{i+1:02d}"] = seq

    return candidates


def generate_adaptive_evo2(
    species_key: str,
    seed_sequence: str,
    strategy: dict,
    n_variants: int = 8,
    species_config: dict = None,
) -> dict:
    """Generate Evo2 candidates with strategy-adapted parameters.

    Applies temperature/top_k ranges from strategy, and prepends
    any prompt suffix for failure-driven constraints.
    """
    try:
        from modules.generation.evo2_generator import call_evo2, TAXONOMY_PROMPTS
    except ImportError:
        return {}

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        return {}

    # Modify seed with position-aware motif insertion if strategy dictates
    modified_seed = seed_sequence
    force_tata = strategy.get("force_tata_insertion", True)
    force_caat = strategy.get("force_caat_insertion", True)

    if len(modified_seed) >= 200:
        # Insert motifs at biologically meaningful positions:
        # TATA box: ~-25 to -35 from TSS (end of promoter)
        # CAAT box: ~-70 to -100 from TSS
        seed_len = len(modified_seed)
        if force_tata:
            tata_pos = seed_len - 35  # ~-35 from TSS
            tata_motif = random.choice(["TATAAA", "TATAAAT", "TATATAA"])
            if tata_pos >= 0 and tata_pos + len(tata_motif) <= seed_len:
                modified_seed = (
                    modified_seed[:tata_pos] + tata_motif
                    + modified_seed[tata_pos + len(tata_motif):]
                )
        if force_caat:
            caat_pos = seed_len - 85  # ~-85 from TSS
            caat_motif = random.choice(["CCAAT", "CCAAT"])
            if caat_pos >= 0 and caat_pos + len(caat_motif) <= seed_len:
                modified_seed = (
                    modified_seed[:caat_pos] + caat_motif
                    + modified_seed[caat_pos + len(caat_motif):]
                )

    # Temperature and top_k from strategy
    t_low, t_high = strategy.get("evo2_temperature_range", (0.6, 1.0))
    top_k_choices = strategy.get("evo2_top_k_choices", [3, 4, 5, 6])

    seed_len = len(modified_seed)
    tokens_to_gen = max(600, 800 - seed_len)

    candidates = {}
    for i in range(n_variants):
        temp = random.uniform(t_low, t_high)
        top_k = random.choice(top_k_choices)

        try:
            seq = call_evo2(
                seed_seq=modified_seed,
                api_key=api_key,
                num_tokens=tokens_to_gen,
                temperature=temp,
                top_k=top_k,
                species_key=species_key,
            )
            if seq and len(seq) >= 200:
                if len(seq) > 800:
                    seq = seq[-800:]
                candidates[f"evo2_v{i+1:02d}"] = seq
        except Exception:
            continue

    return candidates


def generate_adaptive_d3lm(
    species_key: str,
    seed_sequence: str,
    strategy: dict,
    n_variants: int = 8,
) -> dict:
    """Generate D3LM candidates with strategy-adapted parameters."""
    try:
        from modules.generation.d3lm_generator import generate_candidates as d3lm_gen
    except ImportError:
        return {}

    # D3LM generation doesn't have many tuneable params from outside,
    # but we can adjust variant count
    try:
        candidates = d3lm_gen(species_key, seed_sequence, n_variants=n_variants)
        return candidates
    except Exception:
        return {}


def generate_with_strategy(
    species_key: str,
    species_config: dict,
    seed: str,
    strategy: dict,
    models: str,
    logger: logging.Logger,
) -> dict:
    """Generate candidates using all active models with adaptive strategy.

    This replaces the inner loop's generate_candidates with strategy-aware
    generation that allocates variants based on model performance.
    """
    allocation = strategy.get("n_variants_per_model", {})
    candidates = {}

    # Evo2 generation
    if "evo2" in models:
        n_evo = allocation.get("evo2", 8)
        try:
            logger.info(f"  Generating {n_evo} candidates via Evo2 (adaptive)...")
            evo2_cands = generate_adaptive_evo2(
                species_key, seed, strategy,
                n_variants=n_evo,
                species_config=species_config,
            )
            candidates.update(evo2_cands)
            logger.info(f"  Evo2: {len(evo2_cands)} candidates generated")
        except Exception as e:
            logger.warning(f"  Evo2 generation failed: {e}")

    # D3LM generation
    if "d3lm" in models:
        n_d3lm = allocation.get("d3lm", 8)
        try:
            logger.info(f"  Generating {n_d3lm} candidates via D3LM (adaptive)...")
            d3lm_cands = generate_adaptive_d3lm(
                species_key, seed, strategy,
                n_variants=n_d3lm,
            )
            candidates.update(d3lm_cands)
            logger.info(f"  D3LM: {len(d3lm_cands)} candidates generated")
        except Exception as e:
            logger.info(f"  D3LM generation skipped: {e}")

    # Mutational generation — always included, always adapts
    n_mut = allocation.get("mutational", 6)
    try:
        logger.info(f"  Generating {n_mut} candidates via mutational (adaptive)...")
        mut_cands = generate_adaptive_mutational(
            seed=seed,
            species_config=species_config,
            strategy=strategy,
            n_variants=n_mut,
        )
        candidates.update(mut_cands)
        logger.info(f"  Mutational: {len(mut_cands)} candidates (strategy-adapted)")
    except Exception as e:
        logger.warning(f"  Mutational generation failed: {e}")

    if not candidates:
        logger.error("  No candidates generated from any model!")
    else:
        logger.info(f"  Total candidates: {len(candidates)}")

    return candidates


# ═══════════════════════════════════════════════════════════════════════
# SINGLE OUTER ITERATION — Runs inner loop with strategy injection
# ═══════════════════════════════════════════════════════════════════════

def run_outer_iteration(
    state: AutoresearchState,
    seed: str,
    iteration: int,
    logger: logging.Logger,
    models: str = "evo2+d3lm",
    protein_seq: str = None,
    output_dir: str = None,
    use_adaptive_generation: bool = True,
) -> dict:
    """Run one iteration of the outer loop.

    If use_adaptive_generation is True, replaces the inner loop's
    generation with strategy-adaptive generation. Otherwise, delegates
    entirely to run_single_iteration.

    Returns:
        dict with iteration results (same as inner loop, plus outer-loop metadata)
    """
    species_config = state.species_config
    species_key = species_config.get("_config_key", "unknown")

    logger.info(f"=== OUTER ITERATION {iteration} ===")
    logger.info(f"  Strategy: exploration={state.strategy['exploration_vs_refinement']:.0%}, "
                f"cis_prob={state.strategy['cis_insertion_prob']:.2f}, "
                f"gc_bias={state.strategy.get('gc_bias') or 'default'}")

    if use_adaptive_generation:
        # Run the inner loop but with our adaptive generation injected
        # We do this by running the inner loop's steps manually with our generator
        result = _run_with_adaptive_generation(
            state, seed, iteration, logger, models,
            protein_seq, output_dir,
        )
    else:
        # Delegate entirely to inner loop
        result = run_single_iteration(
            species_config=species_config,
            seed=seed,
            iteration=iteration,
            logger=logger,
            models=models,
            n_variants=sum(state.strategy["n_variants_per_model"].values()),
            protein_seq=protein_seq,
            output_dir=output_dir,
        )

    # Record results in state
    state.record_iteration(result)

    return result


def _run_with_adaptive_generation(
    state: AutoresearchState,
    seed: str,
    iteration: int,
    logger: logging.Logger,
    models: str,
    protein_seq: str,
    output_dir: str,
) -> dict:
    """Run one iteration with adaptive generation, then delegate scoring
    to the inner loop.

    Uses generate_fn parameter instead of monkey-patching for clean injection.
    """
    species_config = state.species_config
    species_key = species_config.get("_config_key", "unknown")
    strategy = state.strategy

    # Step 1: Generate with adaptive strategy
    logger.info("Step 1: Generating candidates (adaptive strategy)...")
    candidates = generate_with_strategy(
        species_key, species_config, seed, strategy, models, logger,
    )

    if not candidates:
        return {
            "iteration": iteration,
            "status": "no_candidates",
            "top_candidate": None,
            "n_candidates": 0,
            "failure_reasons": {
                "no_TATA": 0, "no_CAAT": 0,
                "gc_low": 0, "gc_high": 0, "too_many_TATA": 0,
            },
        }

    # Create a generate_fn closure that returns our pre-generated candidates
    def _adaptive_gen_fn(species_config_arg, seed_arg, models_arg,
                         n_variants_arg, logger_arg):
        """Return pre-generated adaptive candidates."""
        logger_arg.info(f"  (Using {len(candidates)} pre-generated adaptive candidates)")
        return candidates

    # Delegate to inner loop with generate_fn parameter (no monkey-patching)
    result = run_single_iteration(
            species_config=species_config,
            seed=seed,
            iteration=iteration,
            logger=logger,
            models=models,
            n_variants=sum(strategy["n_variants_per_model"].values()),
            protein_seq=protein_seq,
            output_dir=output_dir,
            generate_fn=_adaptive_gen_fn,  # Clean injection, no monkey-patching
        )

    return result


# ═══════════════════════════════════════════════════════════════════════
# MAIN OUTER LOOP
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Outer Autoresearch Loop — Adaptive Promoter Design Meta-Learner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/outer_loop.py --species arabidopsis --iterations 10
  python scripts/outer_loop.py --species maize --iterations 15 --stagnation-window 4
  python scripts/outer_loop.py --species nbenthamiana --iterations 20 --models evo2+d3lm

The outer loop wraps the inner auto_loop_v2 and adds:
  - Failure pattern analysis across iterations
  - Stagnation detection with automatic strategy adaptation
  - Model performance tracking with variant reallocation
  - Strategy change logging with before/after reporting
        """,
    )
    parser.add_argument("--species", required=True,
                        help="Target species (e.g., arabidopsis, maize, nbenthamiana)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of outer-loop iterations (default: 10)")
    parser.add_argument("--models", default="evo2+d3lm",
                        help="Generation models: evo2, d3lm, or evo2+d3lm (default: evo2+d3lm)")
    parser.add_argument("--variants", type=int, default=22,
                        help="Total variants per iteration across all models (default: 22)")
    parser.add_argument("--stagnation-window", type=int, default=3,
                        help="Iterations without improvement before strategy change (default: 3)")
    parser.add_argument("--failure-window", type=int, default=5,
                        help="Iterations to analyse for failure patterns (default: 5)")
    parser.add_argument("--min-dominant-ratio", type=float, default=0.25,
                        help="Minimum ratio of a failure type to trigger strategy change (default: 0.25)")
    parser.add_argument("--protein-seq", default=None,
                        help="Protein amino acid sequence for DeepLoc + yield prediction")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: outputs/outer_<species>_<timestamp>)")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable adaptive generation (use inner loop defaults)")
    args = parser.parse_args()

    # ── Load species config ───────────────────────────────────────────
    species_config = load_species_config(args.species)
    species_name = species_config["species"]["name"]
    species_config["_config_key"] = args.species

    # ── Setup output dirs and logging ─────────────────────────────────
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "outputs",
        f"outer_{args.species}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    logger = setup_logging(log_dir, f"outer_{args.species}")

    # ── Initialise state and strategy ─────────────────────────────────
    initial_strategy = copy.deepcopy(DEFAULT_STRATEGY)
    # Distribute variants across models
    total = args.variants
    n_evo = total // 3
    n_d3lm = total // 3
    n_mut = total - n_evo - n_d3lm
    initial_strategy["n_variants_per_model"] = {
        "evo2": n_evo,
        "d3lm": n_d3lm,
        "mutational": n_mut,
    }

    state = AutoresearchState(species_config, initial_strategy)

    # ── Banner ────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("OUTER AUTORESEARCH LOOP — Adaptive Promoter Design")
    logger.info("=" * 70)
    logger.info(f"Species: {species_name}")
    logger.info(f"Type: {'Monocot' if is_monocot(species_config) else 'Dicot'}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Iterations: {args.iterations}")
    logger.info(f"Total variants/iteration: {args.variants}")
    logger.info(f"Allocation: {initial_strategy['n_variants_per_model']}")
    logger.info(f"Stagnation window: {args.stagnation_window}")
    logger.info(f"Failure window: {args.failure_window}")
    logger.info(f"Adaptive generation: {not args.no_adaptive}")
    logger.info(f"Output: {output_dir}")
    if args.protein_seq:
        logger.info(f"Protein seq: {args.protein_seq[:30]}... ({len(args.protein_seq)} aa)")
    logger.info("")

    # ── Get initial seed ──────────────────────────────────────────────
    seed = load_seed(args.species)
    logger.info(f"Initial seed: {len(seed)} bp reference promoter")
    logger.info("")

    # ── Main outer loop ──────────────────────────────────────────────
    for i in range(1, args.iterations + 1):
        # Run one iteration
        try:
            result = run_outer_iteration(
                state=state,
                seed=seed,
                iteration=i,
                logger=logger,
                models=args.models,
                protein_seq=args.protein_seq,
                output_dir=output_dir,
                use_adaptive_generation=not args.no_adaptive,
            )
        except Exception as e:
            logger.error(f"Outer iteration {i} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Record a placeholder
            state.record_iteration({
                "iteration": i,
                "status": "error",
                "top_composite_score": -1,
                "failure_reasons": {
                    "no_TATA": 0, "no_CAAT": 0,
                    "gc_low": 0, "gc_high": 0, "too_many_TATA": 0,
                },
                "top_candidate_id": "",
            })
            logger.info("Continuing with previous seed...")
            continue

        # Update seed from top candidate
        if result.get("top_candidate"):
            seed = result["top_candidate"]
            logger.info(f"New seed from iteration {i} "
                        f"(composite={result.get('top_composite_score', 0):.3f})")

        # ── Strategy adaptation ───────────────────────────────────
        # Check for stagnation
        if state.detect_stagnation(window=args.stagnation_window):
            logger.info(f"  STAGNATION DETECTED: No improvement for "
                        f"{state.stagnation_count} iterations")

        # Analyse failures and model performance
        failure_analysis = state.analyse_failures(window=args.failure_window)
        model_analysis = state.analyse_model_performance()

        # Log current state
        if failure_analysis["dominant_failure"]:
            logger.info(f"  Dominant failure: {failure_analysis['dominant_failure']} "
                        f"({failure_analysis['dominant_ratio']:.0%} of failures)")
        if model_analysis["best_model"]:
            logger.info(f"  Best model: {model_analysis['best_model']} "
                        f"(wins: {model_analysis.get('model_wins', {})})")

        # Propose and apply strategy changes
        changes = state.propose_strategy_change(
            failure_analysis, model_analysis, logger,
        )
        if changes:
            logger.info(f"  Applying {len(changes)} strategy changes...")
            state.apply_strategy(changes, i, logger)

            # Generate improvement report
            state.generate_improvement_report(logger)

        logger.info("")
        logger.info(f"Completed outer iteration {i}/{args.iterations} "
                    f"| Best composite: {state.best_composite:.4f} "
                    f"| Stagnation: {state.stagnation_count}")
        logger.info("")

    # ── Final Summary ─────────────────────────────────────────────────
    summary = state.generate_final_summary()
    logger.info(summary)

    # Save final summary
    summary_path = os.path.join(output_dir, "outer_loop_summary.json")
    summary_data = {
        "species": args.species,
        "species_name": species_name,
        "iterations_requested": args.iterations,
        "iterations_completed": len(
            [r for r in state.iteration_history if r.get("status") == "complete"]
        ),
        "models": args.models,
        "best_composite_score": state.best_composite,
        "best_iteration": state.best_iteration,
        "best_seed_length": len(state.best_seed) if state.best_seed else 0,
        "composite_score_progression": state.composite_history,
        "model_wins": dict(state.model_wins),
        "strategy_changes": len(state.strategy_change_log),
        "strategy_change_details": state.strategy_change_log,
        "final_strategy": state.strategy,
        "results": [
            {
                "iteration": r.get("iteration"),
                "status": r.get("status"),
                "n_candidates": r.get("n_candidates", 0),
                "n_passed_filters": r.get("n_passed_filters", 0),
                "top_composite_score": r.get("top_composite_score"),
                "top_candidate_id": r.get("top_candidate_id"),
                "failure_reasons": r.get("failure_reasons", {}),
            }
            for r in state.iteration_history
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Summary saved: {summary_path}")

    # Save best seed as FASTA
    if state.best_seed:
        best_fasta_path = os.path.join(output_dir, "best_seed_outer_loop.fasta")
        with open(best_fasta_path, "w") as f:
            f.write(f">best_seed_{args.species}_composite{state.best_composite:.3f}"
                    f"_iter{state.best_iteration}\n")
            f.write(state.best_seed + "\n")
        logger.info(f"Best seed FASTA: {best_fasta_path}")

    # Save strategy history
    strategy_path = os.path.join(output_dir, "strategy_history.json")
    strat_history = [
        {"iteration": it, "strategy": copy.deepcopy(s)}
        for s, it in state.strategy_history
    ]
    with open(strategy_path, "w") as f:
        json.dump(strat_history, f, indent=2)
    logger.info(f"Strategy history: {strategy_path}")


if __name__ == "__main__":
    main()
