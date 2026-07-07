# Focused Ablation Study — Gap 12
#
# Runs the mutational-only ablation with 3× repeats per species to produce
# statistically rigorous results with mean ± std and 95% CI.
# Evo2-only and D3LM-only are attempted but gracefully skipped if unavailable.
#
# Usage:
#   python scripts/run_ablation_study.py --species nbenthamiana
#   python scripts/run_ablation_study.py --species nbenthamiana arabidopsis

import argparse
import json
import logging
import os
import sys
import time
import random
from datetime import datetime

import numpy as np

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.cross_species.species_config import load_species_config
from modules.evaluation.cis_scoring import score_candidate, gc_content
from modules.generation.mutational_generator import build_species_scaffold, generate_from_seed
from auto_loop_v2 import load_seed, setup_logging


N_REPEATS = 3
N_CANDIDATES = 10
MUTATION_RATES = [0.05, 0.10, 0.15, 0.20, 0.30]


def run_mutational_only(species_config, seed, n_candidates, logger):
    """Generate and score candidates using only mutational generator."""
    candidates = []
    try:
        # generate_from_seed returns dict of {variant_id: sequence}
        # It internally varies mutation rates across variants
        variants = generate_from_seed(seed, species_config, n_variants=n_candidates)
        for vid, seq in variants.items():
            result = score_candidate(seq, species_config)
            result["source"] = "mutational"
            result["variant_id"] = vid
            candidates.append(result)
    except Exception as e:
        logger.warning(f"  mutational generation failed: {e}")
    return candidates


def run_evo2_only(species_config, seed, n_candidates, logger):
    """Generate using Evo2 only. Returns empty list if unavailable."""
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        logger.info("  Evo2: SKIPPED (no API key)")
        return []
    try:
        from modules.generation.evo2_generator import generate_evo2_variants
        candidates = []
        results = generate_evo2_variants(
            seed_sequence=seed, n_variants=n_candidates, api_key=api_key,
            species_config=species_config,
        )
        for seq in results.get("sequences", []):
            result = score_candidate(seq, species_config)
            result["source"] = "evo2"
            candidates.append(result)
        return candidates
    except Exception as e:
        logger.info(f"  Evo2: FAILED ({e})")
        return []


def run_d3lm_only(species_config, seed, n_candidates, logger):
    """Generate using D3LM only. Returns empty list if unavailable."""
    try:
        from modules.generation.d3lm_generator import generate_candidates as d3lm_gen
        candidates = []
        species_key = species_config.get("_config_key", "nbenthamiana")
        results = d3lm_gen(
            species_key=species_key, seed_sequence=seed,
            n_variants=n_candidates,
        )
        # D3LM returns dict of {variant_id: sequence_string}
        if isinstance(results, dict):
            for vid, seq in results.items():
                if isinstance(seq, str) and len(seq) >= 100:
                    result = score_candidate(seq, species_config)
                    result["source"] = "d3lm"
                    result["variant_id"] = vid
                    candidates.append(result)
        return candidates
    except Exception as e:
        logger.info(f"  D3LM: SKIPPED ({e})")
        return []


def summarize(candidates, label):
    """Summarize scored candidates."""
    if not candidates:
        return {"n": 0, "label": label, "available": False}
    scores = [c["weighted_score"] for c in candidates]
    pass_count = sum(1 for c in candidates if c.get("passed_filters"))
    return {
        "label": label,
        "available": True,
        "n": len(candidates),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "pass_rate": pass_count / len(candidates),
        "pass_count": pass_count,
    }


def run_ablation_for_species(species_key, logger):
    """Run full ablation study for one species with 3× repeats."""
    species_config = load_species_config(species_key)
    species_config["_config_key"] = species_key
    name = species_config["species"]["name"]

    logger.info(f"\n{'='*70}")
    logger.info(f"ABLATION STUDY: {name} ({species_key})")
    logger.info(f"{'='*70}")

    seed = load_seed(species_key)
    if not seed:
        logger.error(f"No seed available for {species_key}")
        return None

    all_results = {}

    for config_name, runner in [
        ("mutational_only", run_mutational_only),
        ("evo2_only", run_evo2_only),
        ("d3lm_only", run_d3lm_only),
    ]:
        logger.info(f"\n--- {config_name} ---")
        repeat_summaries = []

        for repeat in range(1, N_REPEATS + 1):
            random.seed(repeat * 42)
            np.random.seed(repeat * 42)
            t0 = time.time()

            candidates = runner(species_config, seed, N_CANDIDATES, logger)
            elapsed = time.time() - t0
            summary = summarize(candidates, config_name)
            summary["repeat"] = repeat
            summary["wall_time_s"] = elapsed
            repeat_summaries.append(summary)

            if summary["available"]:
                logger.info(
                    f"  Repeat {repeat}: n={summary['n']}, "
                    f"mean={summary['mean_score']:.2f}, "
                    f"pass={summary['pass_rate']:.0%}, "
                    f"time={elapsed:.1f}s"
                )
            else:
                logger.info(f"  Repeat {repeat}: not available")

        # Aggregate across repeats
        available_runs = [s for s in repeat_summaries if s.get("available")]
        if available_runs:
            means = [s["mean_score"] for s in available_runs]
            passes = [s["pass_rate"] for s in available_runs]
            times = [s["wall_time_s"] for s in available_runs]
            aggregate = {
                "label": config_name,
                "available": True,
                "n_repeats": len(available_runs),
                "score_mean_of_means": float(np.mean(means)),
                "score_std_of_means": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
                "score_95ci": None,
                "pass_rate_mean": float(np.mean(passes)),
                "pass_rate_std": float(np.std(passes, ddof=1)) if len(passes) > 1 else 0.0,
                "wall_time_mean": float(np.mean(times)),
            }
            if len(means) > 1:
                ci_half = 1.96 * aggregate["score_std_of_means"] / np.sqrt(len(means))
                aggregate["score_95ci"] = (
                    float(aggregate["score_mean_of_means"] - ci_half),
                    float(aggregate["score_mean_of_means"] + ci_half),
                )
        else:
            aggregate = {"label": config_name, "available": False}

        all_results[config_name] = {
            "aggregate": aggregate,
            "repeats": repeat_summaries,
        }

    # Comparison table
    logger.info(f"\n{'='*70}")
    logger.info("ABLATION COMPARISON")
    logger.info(f"{'='*70}")
    logger.info(
        f"  {'Config':20s} | {'Mean':>8s} | {'Std':>8s} | {'95% CI':>20s} | "
        f"{'Pass%':>6s} | {'Time':>6s}"
    )
    logger.info(
        f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*20}-+-{'-'*6}-+-{'-'*6}"
    )

    ranking = []
    for name, data in all_results.items():
        agg = data["aggregate"]
        if not agg.get("available"):
            logger.info(f"  {name:20s} | {'N/A':>8s} | {'N/A':>8s} | {'N/A':>20s} | {'N/A':>6s} | {'N/A':>6s}")
            continue
        ci = agg.get("score_95ci", (0, 0))
        ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "N/A"
        logger.info(
            f"  {name:20s} | {agg['score_mean_of_means']:8.2f} | "
            f"{agg['score_std_of_means']:8.2f} | {ci_str:>20s} | "
            f"{agg['pass_rate_mean']:5.0%} | {agg['wall_time_mean']:5.1f}s"
        )
        ranking.append((name, agg["score_mean_of_means"]))

    ranking.sort(key=lambda x: -x[1])
    logger.info(f"\n  Ranking: {' > '.join(f'{n}={s:.2f}' for n, s in ranking)}")
    logger.info(f"{'='*70}")

    return all_results


def main():
    global N_REPEATS, N_CANDIDATES

    parser = argparse.ArgumentParser(description="Ablation Study (Gap 12)")
    parser.add_argument("--species", nargs="+", default=["nbenthamiana"])
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--candidates", type=int, default=N_CANDIDATES)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    N_REPEATS = args.repeats
    N_CANDIDATES = args.candidates

    output_dir = args.output_dir or os.path.join(
        _project_root, "outputs",
        f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(output_dir, exist_ok=True)

    log_dir = os.path.join(_project_root, "logs")
    logger = setup_logging(log_dir, "ablation_gap12")

    logger.info(f"Ablation study output: {output_dir}")
    logger.info(f"Species: {args.species}")
    logger.info(f"Repeats per config: {N_REPEATS}")
    logger.info(f"Candidates per repeat: {N_CANDIDATES}")

    all_species_results = {}
    for species_key in args.species:
        results = run_ablation_for_species(species_key, logger)
        if results:
            all_species_results[species_key] = results

    # Save
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

    out_path = os.path.join(output_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(convert(all_species_results), f, indent=2)
    logger.info(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
