# V2 Auto-Research Loop
# Species-agnostic autonomous promoter design loop
#
# Runs overnight: generate -> filter -> evaluate -> rank -> report -> repeat
# Supports any target species via species config
#
# Usage:
#   python auto_loop_v2.py --species nbenthamiana --iterations 5
#   python auto_loop_v2.py --species maize --iterations 10 --models evo2+d3lm
#   python auto_loop_v2.py --species arabidopsis --iterations 3 --protein-seq MSEKKIA...

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.cross_species.species_config import (
    load_species_config, list_available_species, is_monocot,
)
from modules.evaluation.cis_scoring import (
    scan_cis_elements, compute_weighted_score, gc_content,
    apply_hard_filters, score_candidate, compute_hybrid_expression,
)
from modules.evaluation.multi_objective import (
    pareto_rank_v2, compute_composite_v2, DEFAULT_OBJECTIVES,
)
from modules.evaluation.report_card import generate_report_card
from modules.silencing.silencing_risk import compute_silencing_risk
from modules.genomics.safe_harbor import SafeHarborPredictor


# ─────────────────────────────────────────────────────────────────────
# RUNTIME HELPERS
# ─────────────────────────────────────────────────────────────────────

def _candidate_source_counts(candidates: dict) -> dict:
    """Count generated candidates by source prefix."""
    counts = {"evo2": 0, "d3lm": 0, "mutational": 0, "other": 0}
    for cid in candidates:
        if cid.startswith("evo2_"):
            counts["evo2"] += 1
        elif cid.startswith("d3lm_"):
            counts["d3lm"] += 1
        elif cid.startswith("mut_"):
            counts["mutational"] += 1
        else:
            counts["other"] += 1
    return counts


def _requested_model_list(models: str) -> list:
    """Normalise requested model string into a list."""
    aliases = {
        "evo2_only": "evo2",
        "d3lm_only": "d3lm",
        "mutational_only": "mutational",
    }
    normalised = []
    for model in models.split("+"):
        model = model.strip()
        if not model:
            continue
        normalised.append(aliases.get(model, model))
    return normalised


BASELINES = {
    "dicot": 0.60,
    "monocot": 0.70,
}

SEED_FILE_ALIASES = {
    "ntobacum": "nbenthamiana",
    "by2_cells": "nbenthamiana",
    "soybean": "tomato",
    "wheat": "maize",
}

CASSSETTE_TEMPLATE = "Promoter -> 5'UTR -> Hyaluronidase -> Terminator"
CURATED_FALLBACK_SEEDS = {
    "arabidopsis": (
        "AAGTAAAAGAAGAATCTTTCTTTCTTTGTTGGTGAATTCAGGTGATGAAACGCAACGTTTCTGCTAAAAGATCGGAACTTTGTCTTGATTCTTCAAAGATTAGATTAGATCATCGTTGGAGCTTCATTGGAGGATCAAGAATCTCTGTTCAGTCCAATTCTTACACCGTAGTTCACAAGAAATTCTCCGGTGTACGAGCTTCATGTACTGTGTCTCATAACCATTTAAAGATTAAACTTTGTCACTCATTTTGTGACTTTGATCTAAGATTATATGTGTGATTAGAATCTCTTATATGTTCATGTTGTAGGGTTAACTACTACTCAGATTGCAAGCAGTGTATTTGCGGTTGGAACTACCGCGGTTTTACCCTTCTATACTCTAATGGTTGTAGCACCAA"
    ),
    "tomato": (
        "CCGATAAATCGAACGATAATATATAAAATCGAACCGAACCGATCGATGCACACCCTTGACCAAATCTGAAGCACATATTTATCGATCTAAATTTTATTAAAGAGATTAATATCGAATAATCATATACATATTTCATATGTATAACAAATTTCAAATACACGTATCTAATATATCGAGTGATGCGACAAATACATGTATCGGACGCACCAATTGATATAGAAAACGTAATATTGAAAACTAATGTAAAGAAAAGTAACTTGATCCTAAACTAATCAAGATAAGCCCAATAAATATACATTGTCATCTCCAAAGGCCCAAAAATGGCACAAGATGGCAGGCCCAATAACGAAGAAAAGGGCTTGTAAAACCCTAATAAAGTGGCACTGGCAGAGCTTACACT"
    ),
    "nbenthamiana": (
        "TTGTCTACTCCAAAAATATCAAAGATACAGTCTCAGAAGACCAAAGGGCAATTGAGACTTTTCAACAAAGGGTAATATCCGGAAACCTCCTCGGATTCCATTGCCCAGCTATCTGTCACTTTATTGTGAAGATAGTGGAAAAGGAAGGTGGCTCCTACAAATGCCATCATTGCGATAAAGGAAAGGCCATCGTTGAAGATGCCTCTGCCGACAGTGGTCCCAAAGATGGACCCCCACCCACGAGGAGCATCGTGGAAAAAGAAGACGTTCCAACCACGTCTTCAAAGCAAGTGGATTGATGTGATATCTCCACTGACGTAAGGGATGACGCACAATCCCACTATCCTTCGCAAGACCCTTCCTCTATATAAGGAAGTTCATTTCATTTGGAGAGAACACG"
    ),
    "rice": (
        "GGTAGTTTGGGTGGGCGAGAGGCGGCTTCGTGCGCGCCCAGATCGGTGCGCGGGAGGGGCGGGATCTCGCGGCTGGGGCTCTCGCCGGCGTGGATCCGGCCCGGATCTCGCGGGGAATGGGGCTCTCGGATGTAGATCTGCGATCCGCCGTTGTTGGGGGAGATGATGGGGGGTTTAAAATTTCCGCCATGCTAAACAAGATCAGGAAGAGGGGAAAAGGGCACTATGGTTTATATTTTTATATATTTCTGCTGCTTCGTCAGGCTTAGATGTGCTAGATCTTTCTTTCTTCTTTTTGTGGGTAGAATTTGAATCCCTCAGCATTGTTCATCGGTAGTTTTTCTTTTCATGATTTGTGACAAATGCAGCCTCGTGCGGAGCTTTTTTGTAGGTAGAAGCC"
    ),
    "maize": (
        "GTCGTTCATTCGTTCTAGATCGGAGTAGAATACTGTTTCAAACTACCTGGTGTATTTATTAATTTTGGAACTGTATGTGTGTGTCATACATCTTCATAGTTACGAGTTTAAGATGGATGGAAATATCGATCTAGGATAGGTATACATGTTGATGTGGGTTTTACTGATGCATATACATGATGGCATATGCAGCATCTATTCATATGCTCTAACCTTGAGTACCTATCTATTATAATAAACAAGTATGTTTTATAATTATTTTGATCTTGATATACTTGGATGATGGCATATGCAGCAGCTATATGTGGATTTTTTTAGCCCTGCCTTCATACGCTATTTATTTGCTTGGTACTGTTTCTTTTGTCGATGCTCACCCTGTTGTTTGGTGTTACTTCTGCAG"
    ),
}


def species_type(species_config: dict) -> str:
    return "monocot" if is_monocot(species_config) else "dicot"


def baseline_reference_name(species_config: dict) -> str:
    return "Maize_Ubiquitin" if species_type(species_config) == "monocot" else "CaMV_35S"


def predict_expression(weighted_score: float, silencing_risk: float, gc_pct: float) -> float:
    legacy = compute_hybrid_expression(
        {"weighted_score": weighted_score, "gc_pct": gc_pct},
        silencing_risk=silencing_risk,
        embedding_similarity=0.5,
        internal_div=0.5,
    )
    return legacy["expression_score"]


def expression_label(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.5:
        return "MEDIUM"
    return "LOW"


def apply_context(expr_score: float) -> dict:
    return {
        "safe_harbor": round(min(expr_score + 0.1, 1.0), 4),
        "random": round(expr_score, 4),
        "repressed": round(max(expr_score - 0.2, 0.0), 4),
    }


def apply_failure_feedback(seed: str, failure_counts: dict,
                           logger: logging.Logger = None) -> str:
    """Apply a minimal architecture repair when failures cluster."""
    if not failure_counts:
        return seed

    no_tata = failure_counts.get("no_TATA", 0) + failure_counts.get("no_in_zone_TATA", 0)
    no_caat = failure_counts.get("no_CAAT", 0) + failure_counts.get("no_in_zone_CAAT", 0)
    if no_tata <= 2 and no_caat <= 2:
        return seed

    try:
        from modules.generation.mutational_generator import restore_core_architecture
        repaired = restore_core_architecture(seed, target_length=len(seed))
        if logger is not None:
            repairs = []
            if no_tata > 2:
                repairs.append("TATA")
            if no_caat > 2:
                repairs.append("CAAT")
            logger.info(f"Applying failure feedback to seed: reinforcing {'/'.join(repairs)} architecture")
        return repaired
    except Exception as e:
        if logger is not None:
            logger.warning(f"Failure feedback skipped: {e}")
        return seed


# ─────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str, species: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"auto_loop_v2_{species}_{timestamp}.log")

    logger = logging.getLogger(f"auto_loop_v2_{species}")
    logger.setLevel(logging.INFO)
    # Prevent duplicate handlers on repeated calls
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(sh)

    return logger


# ─────────────────────────────────────────────────────────────────────
# NOVELTY & DIVERSITY (from V1 — Levenshtein-based)
# ─────────────────────────────────────────────────────────────────────

def levenshtein(s1: str, s2: str) -> int:
    """Pure-Python Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, prev[j] + (c1 != c2), curr[j] + 1))
        prev = curr
    return prev[-1]


def compute_novelty(candidates: dict, ref_seq: str, region_len: int = 300) -> dict:
    """Normalised Levenshtein distance from reference (last 300 bp)."""
    ref_region = ref_seq[-region_len:]
    max_dist = len(ref_region)
    scores = {}
    for cid, seq in candidates.items():
        cand_region = seq[-region_len:]
        dist = levenshtein(ref_region, cand_region)
        scores[cid] = round(min(dist, max_dist) / max_dist, 4)
    return scores


def compute_diversity(candidates: dict, region_len: int = 300) -> dict:
    """Mean pairwise Levenshtein distance (uniqueness within cohort)."""
    cids = list(candidates.keys())
    seqs = {cid: candidates[cid][-region_len:] for cid in cids}
    n = len(cids)
    scores = {}
    for i, ci in enumerate(cids):
        total = 0
        count = 0
        for j, cj in enumerate(cids):
            if i == j:
                continue
            total += levenshtein(seqs[ci], seqs[cj])
            count += 1
        scores[ci] = round(total / (count * region_len), 4) if count > 0 else 0.0
    return scores


def load_prior_top_sequences(output_dir: str, current_iteration: int) -> list:
    """Load prior top-ranked sequences from previous iteration CSVs."""
    if not output_dir or current_iteration <= 1:
        return []

    prior = []
    try:
        import pandas as pd
        for prev_iter in range(1, current_iteration):
            path = os.path.join(output_dir, f"iter{prev_iter}_scored.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, index_col=0)
            if not df.empty and "sequence" in df.columns:
                prior.append(str(df.iloc[0]["sequence"]))
    except Exception:
        return prior
    return prior


def compute_history_similarity_penalty(candidates: dict, prior_sequences: list,
                                       region_len: int = 300) -> dict:
    """Penalise near-duplicates of previous top candidates."""
    if not prior_sequences:
        return {cid: 0.0 for cid in candidates}

    penalties = {}
    prior_regions = [seq[-region_len:] for seq in prior_sequences if seq]
    for cid, seq in candidates.items():
        cand_region = seq[-region_len:]
        max_identity = 0.0
        for prev_region in prior_regions:
            dist = levenshtein(cand_region, prev_region)
            identity = 1.0 - min(dist, region_len) / region_len
            max_identity = max(max_identity, identity)
        penalties[cid] = round(max(0.0, (max_identity - 0.82) * 0.6), 4)
    return penalties


def _load_first_fasta_sequence(fasta_path: str) -> str:
    """Load the first FASTA entry from a seed file."""
    seq_parts = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_parts:
                    break
                continue
            seq_parts.append(line.upper())
    seq = "".join(seq_parts)
    return "".join(base for base in seq if base in "ACGTN")


# ─────────────────────────────────────────────────────────────────────
# EMBEDDING SIMILARITY (tries AgroNT -> PlantBERT -> kmer fallback)
# ─────────────────────────────────────────────────────────────────────

def compute_embedding_similarity(candidates: dict, ref_seq: str,
                                 logger: logging.Logger) -> dict:
    """Compute DNA embedding similarity to reference promoter.

    Tries AgroNT first (plant-specific, best quality), then PlantBERT,
    then falls back to 4-mer frequency cosine similarity.
    """
    # Try AgroNT
    try:
        from modules.embedding.agront import load_agront, embed_sequence_agront
        import torch
        import torch.nn.functional as F

        logger.info("  Using AgroNT (1B, plant-specific) for embedding similarity...")
        tokenizer, model = load_agront()
        ref_emb = embed_sequence_agront(ref_seq, tokenizer, model)

        sims = {}
        for cid, seq in candidates.items():
            emb = embed_sequence_agront(seq, tokenizer, model)
            sim = float(F.cosine_similarity(
                ref_emb.cpu().flatten().unsqueeze(0),
                emb.cpu().flatten().unsqueeze(0)
            ).item())
            sims[cid] = round(sim, 4)
        logger.info(f"  AgroNT similarity range: {min(sims.values()):.3f} - {max(sims.values()):.3f}")
        return sims
    except Exception as e:
        logger.info(f"  AgroNT unavailable ({e}), trying PlantBERT...")

    # Try PlantBERT
    try:
        from modules.embedding.plantbert import load_plantbert, embed_sequence_plant
        import torch
        import torch.nn.functional as F

        logger.info("  Using PlantBERT for embedding similarity...")
        tokenizer, model = load_plantbert()
        ref_emb = embed_sequence_plant(ref_seq, tokenizer, model)

        sims = {}
        for cid, seq in candidates.items():
            emb = embed_sequence_plant(seq, tokenizer, model)
            sim = float(F.cosine_similarity(
                ref_emb.cpu().flatten().unsqueeze(0),
                emb.cpu().flatten().unsqueeze(0)
            ).item())
            sims[cid] = round(sim, 4)
        logger.info(f"  PlantBERT similarity range: {min(sims.values()):.3f} - {max(sims.values()):.3f}")
        return sims
    except Exception as e:
        logger.info(f"  PlantBERT unavailable ({e}), using k-mer fallback...")

    # Fallback: 4-mer cosine similarity
    logger.info("  Using 4-mer frequency similarity (no ML model available)")
    return _kmer_similarity_batch(candidates, ref_seq)


def _kmer_similarity_batch(candidates: dict, ref_seq: str, k: int = 4) -> dict:
    """4-mer frequency cosine similarity fallback."""
    def kmer_vec(seq, k):
        seq = seq.upper()
        vec = np.zeros(4 ** k, dtype=np.float32)
        for i in range(len(seq) - k + 1):
            km = seq[i:i+k]
            if all(c in 'ACGT' for c in km):
                idx = sum({'A': 0, 'C': 1, 'G': 2, 'T': 3}[c] * (4 ** (k - 1 - j))
                          for j, c in enumerate(km))
                vec[idx] += 1
        total = vec.sum()
        return vec / total if total > 0 else vec

    ref_v = kmer_vec(ref_seq, k)
    sims = {}
    for cid, seq in candidates.items():
        v = kmer_vec(seq, k)
        dot = np.dot(ref_v, v)
        norm = np.linalg.norm(ref_v) * np.linalg.norm(v) + 1e-10
        sims[cid] = round(float(dot / norm), 4)
    return sims


# ─────────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────────

def generate_candidates(species_config: dict, seed: str,
                        models: str, n_variants: int,
                        logger: logging.Logger,
                        add_mutational_baseline: bool = True,
                        fail_on_requested_model_failure: bool = False) -> dict:
    """Generate candidate promoters using Evo2 and optionally D3LM.

    Args:
        species_config: Species configuration dict
        seed: Seed sequence from previous iteration
        models: "evo2", "d3lm", or "evo2+d3lm"
        n_variants: Number of variants per model
        logger: Logger

    Returns:
        dict of {candidate_id: sequence}
    """
    species_id = species_config["species"]["name"].lower().replace(" ", "_").replace(".", "")
    # Map back to config key
    species_key = None
    for s in list_available_species():
        cfg = load_species_config(s)
        if cfg["species"]["name"] == species_config["species"]["name"]:
            species_key = s
            break
    if not species_key:
        species_key = "nbenthamiana"  # fallback

    candidates = {}
    requested_models = _requested_model_list(models)
    generation_failures = {}

    # Evo2 generation
    if "evo2" in requested_models:
        try:
            from modules.generation.evo2_generator import generate_candidates as evo2_gen
            logger.info(f"  Generating {n_variants} candidates via Evo2-40B...")
            evo2_cands = evo2_gen(species_key, seed, n_variants=n_variants)
            candidates.update(evo2_cands)
            logger.info(f"  Evo2: {len(evo2_cands)} candidates generated")
        except Exception as e:
            generation_failures["evo2"] = str(e)
            logger.warning(f"  Evo2 generation failed: {e}")
            logger.warning("  (Is NVIDIA_API_KEY set? Continuing without Evo2)")

    # D3LM generation
    if "d3lm" in requested_models:
        try:
            from modules.generation.d3lm_generator import generate_candidates as d3lm_gen
            logger.info(f"  Generating {n_variants} candidates via D3LM...")
            d3lm_cands = d3lm_gen(species_key, seed, n_variants=n_variants)
            candidates.update(d3lm_cands)
            logger.info(f"  D3LM: {len(d3lm_cands)} candidates generated")
        except Exception as e:
            generation_failures["d3lm"] = str(e)
            logger.info(f"  D3LM generation skipped: {e}")

    # Explicit mutational mode or optional safety baseline
    need_mutational = ("mutational" in requested_models) or add_mutational_baseline
    if need_mutational:
        n_mut = n_variants if requested_models == ["mutational"] else max(2, n_variants // 2)
        try:
            from modules.generation.mutational_generator import generate_candidates as mut_gen
            mut_cands = mut_gen(species_key, seed, n_variants=n_mut,
                                species_config=species_config)
            candidates.update(mut_cands)
            if "mutational" in requested_models and not add_mutational_baseline:
                logger.info(f"  Mutational: {len(mut_cands)} candidates generated")
            else:
                logger.info(f"  Mutational: {len(mut_cands)} candidates (ensures filter-passing baseline)")
        except Exception as e:
            generation_failures["mutational"] = str(e)
            logger.warning(f"  Mutational generation failed: {e}")
    else:
        logger.info("  Mutational baseline disabled for strict benchmark run")

    if not candidates:
        logger.error("  No candidates generated from any model!")
    else:
        logger.info(f"  Total candidates: {len(candidates)}")
        source_counts = _candidate_source_counts(candidates)
        actual_sources = [name for name, count in source_counts.items() if count > 0 and name != "other"]
        logger.info(f"  Actual candidate sources: {', '.join(actual_sources) if actual_sources else 'none'}")
        if requested_models and set(actual_sources) - {"mutational"} != set(requested_models):
            logger.info(f"  Requested models: {', '.join(requested_models)}")
        if generation_failures:
            failure_summary = ", ".join(f"{k} failed" for k in generation_failures)
            logger.info(f"  Generation failures: {failure_summary}")

        if fail_on_requested_model_failure:
            source_counts = _candidate_source_counts(candidates)
            missing = [m for m in requested_models if source_counts.get(m, 0) == 0]
            if missing:
                raise RuntimeError(
                    "Requested models produced no candidates: " + ", ".join(missing)
                )

    return candidates


# ─────────────────────────────────────────────────────────────────────
# SINGLE ITERATION
# ─────────────────────────────────────────────────────────────────────

def run_single_iteration(species_config: dict, seed: str, iteration: int,
                         logger: logging.Logger, models: str = "evo2+d3lm",
                         n_variants: int = 20, protein_seq: str = None,
                         output_dir: str = None, generate_fn=None,
                         add_mutational_baseline: bool = True,
                         fail_on_requested_model_failure: bool = False) -> dict:
    """Run a single iteration of the auto-research loop.

    Steps:
    1. Generate candidates (Evo2 + D3LM ensemble)
    2. Apply species-specific hard filters
    3. Evaluate across all axes (strength, silencing, novelty, diversity, embedding)
    4. Pareto rank and compute composite scores
    5. Generate report cards for top candidates
    6. Return top candidate as seed for next iteration

    Args:
        species_config: Species configuration dict
        seed: Seed sequence from previous iteration (or reference)
        iteration: Iteration number
        logger: Logger instance
        models: Which generation models to use
        n_variants: Number of variants per model
        protein_seq: Optional protein sequence for DeepLoc/yield prediction
        output_dir: Directory to save results
        generate_fn: Optional callable to replace default generation.
                     Signature: (species_config, seed, models, n_variants, logger) -> dict
                     If provided, this is used instead of generate_candidates().
        models: Which generation models to use
        n_variants: Number of variants per model
        protein_seq: Optional protein sequence for DeepLoc/yield prediction
        output_dir: Directory to save results

    Returns:
        dict with iteration results including top candidate
    """
    species_name = species_config["species"]["name"]
    species_key = species_config.get("_config_key", "unknown")
    species_kind = species_type(species_config)
    iter_t0 = time.perf_counter()
    step_timings = {}
    requested_models = _requested_model_list(models)
    logger.info(f"=== ITERATION {iteration} ({species_name}) ===")

    # ── Step 1: Generate candidates ──────────────────────────────────
    logger.info("Step 1: Generating candidates...")
    t0 = time.perf_counter()
    if generate_fn is not None:
        candidates = generate_fn(species_config, seed, models, n_variants, logger)
    else:
        candidates = generate_candidates(
            species_config, seed, models, n_variants, logger,
            add_mutational_baseline=add_mutational_baseline,
            fail_on_requested_model_failure=fail_on_requested_model_failure,
        )
    step_timings["generate_candidates_s"] = round(time.perf_counter() - t0, 3)

    if not candidates:
        return {
            "iteration": iteration,
            "status": "no_candidates",
            "top_candidate": None,
            "n_candidates": 0,
            "requested_models": requested_models,
            "actual_model_counts": {},
            "step_timings_s": step_timings,
            "iteration_wall_time_s": round(time.perf_counter() - iter_t0, 3),
            "failure_reasons": {"no_TATA": 0, "no_CAAT": 0, "gc_low": 0, "gc_high": 0, "too_many_TATA": 0},
        }

    actual_model_counts = _candidate_source_counts(candidates)

    # ── Step 2: Cis-element scoring + hard filtering ─────────────────
    logger.info("Step 2: Cis-element scoring and hard filtering...")
    t0 = time.perf_counter()
    hard_filter = species_config.get("hard_filter", {})
    cis_weights = species_config.get("cis_element_weights", {})

    scored = {}
    filtered_out = {}
    failure_reasons = {
        "no_TATA": 0,
        "no_CAAT": 0,
        "no_in_zone_TATA": 0,
        "no_in_zone_CAAT": 0,
        "bad_TATA_CAAT_spacing": 0,
        "gc_low": 0,
        "gc_high": 0,
        "too_many_TATA": 0,
    }
    for cid, seq in candidates.items():
        result = score_candidate(seq, species_config)
        if result["passed_filters"]:
            scored[cid] = {"sequence": seq, **result}
        else:
            filtered_out[cid] = {"sequence": seq, **result}
            # Track WHY candidates fail (for feedback)
            # Note: score_candidate() spreads cis-counts as flat keys (TATA_box, CAAT_box, etc.)
            failures = result.get("filter_failures", [])
            if result.get("TATA_box", 0) == 0:
                failure_reasons["no_TATA"] += 1
            if "Missing in-zone TATA box" in failures:
                failure_reasons["no_in_zone_TATA"] += 1
            if result.get("CAAT_box", 0) == 0:
                failure_reasons["no_CAAT"] += 1
            if "Missing in-zone CAAT box" in failures:
                failure_reasons["no_in_zone_CAAT"] += 1
            if "TATA-CAAT spacing/order outside acceptable range" in failures:
                failure_reasons["bad_TATA_CAAT_spacing"] += 1
            gc = result.get("gc_pct", 0)
            if gc < hard_filter.get("min_gc_pct", 30):
                failure_reasons["gc_low"] += 1
            if gc > hard_filter.get("max_gc_pct", 60):
                failure_reasons["gc_high"] += 1
            if result.get("TATA_box", 0) > hard_filter.get("max_tata_boxes", 6):
                failure_reasons["too_many_TATA"] += 1

    logger.info(f"  Hard filter: {len(candidates)} -> {len(scored)} passed "
                f"({len(filtered_out)} removed)")
    step_timings["cis_scoring_filter_s"] = round(time.perf_counter() - t0, 3)

    # Log failure summary for feedback
    if filtered_out:
        top_failures = sorted(failure_reasons.items(), key=lambda x: -x[1])
        failure_str = ", ".join(f"{k}={v}" for k, v in top_failures if v > 0)
        logger.info(f"  Failure reasons: {failure_str}")

    if not scored:
        logger.warning("  No candidates passed hard filters! Using all candidates with penalty.")
        for cid, seq in candidates.items():
            result = score_candidate(seq, species_config)
            result["weighted_score"] *= 0.5  # Penalty for failing filters
            result["passed_filters"] = False
            scored[cid] = {"sequence": seq, **result}

    # ── Step 3: Novelty and diversity ────────────────────────────────
    logger.info("Step 3: Computing novelty and diversity...")
    t0 = time.perf_counter()
    novelty = compute_novelty({cid: d["sequence"] for cid, d in scored.items()}, seed)
    diversity = compute_diversity({cid: d["sequence"] for cid, d in scored.items()})
    prior_top_sequences = load_prior_top_sequences(output_dir, iteration)
    history_penalty = compute_history_similarity_penalty(
        {cid: d["sequence"] for cid, d in scored.items()},
        prior_top_sequences,
    )

    for cid in scored:
        scored[cid]["novelty_35s"] = novelty.get(cid, 0.5)
        scored[cid]["internal_div"] = diversity.get(cid, 0.5)
        scored[cid]["history_similarity_penalty"] = history_penalty.get(cid, 0.0)
    step_timings["novelty_diversity_s"] = round(time.perf_counter() - t0, 3)

    # ── Step 4: Silencing risk ───────────────────────────────────────
    logger.info("Step 4: Computing silencing risk...")
    t0 = time.perf_counter()
    for cid, data in scored.items():
        silencing = compute_silencing_risk(data["sequence"])
        scored[cid]["silencing_risk"] = silencing["overall_risk"]
        scored[cid]["silencing_detail"] = silencing

    silencing_vals = [d["silencing_risk"] for d in scored.values()]
    logger.info(f"  Silencing risk range: {min(silencing_vals):.3f} - {max(silencing_vals):.3f}")
    step_timings["silencing_s"] = round(time.perf_counter() - t0, 3)

    # ── Step 5: Embedding similarity ─────────────────────────────────
    logger.info("Step 5: Computing DNA embedding similarity...")
    t0 = time.perf_counter()
    try:
        emb_sims = compute_embedding_similarity(
            {cid: d["sequence"] for cid, d in scored.items()}, seed, logger
        )
        for cid in scored:
            scored[cid]["embedding_similarity"] = emb_sims.get(cid, 0.5)
    except Exception as e:
        logger.warning(f"  Embedding similarity failed: {e}")
        for cid in scored:
            scored[cid]["embedding_similarity"] = 0.5
    step_timings["embedding_similarity_s"] = round(time.perf_counter() - t0, 3)

    # ── Step 6: Safe harbor scoring ──────────────────────────────────
    logger.info("Step 6: Scoring safe harbor positions...")
    t0 = time.perf_counter()
    safe_harbors = species_config.get("safe_harbors", {})
    known_sites = safe_harbors.get("known_sites", [])
    genome_cfg = species_config.get("genome", {})
    fasta_path = genome_cfg.get("fasta", "")
    gff_path = genome_cfg.get("annotation", "")
    genome_available = bool(
        fasta_path and os.path.exists(fasta_path)
        and gff_path and os.path.exists(gff_path)
    )

    def _make_predictor():
        """Create SafeHarborPredictor with species key for centromere data."""
        return SafeHarborPredictor(
            genome_fasta=fasta_path if fasta_path and os.path.exists(fasta_path) else None,
            annotation_gff=gff_path if gff_path and os.path.exists(gff_path) else None,
            species_key=species_key,
        )

    if not genome_available:
        placement_label = "Safe harbor prediction not performed (genome data unavailable)"
        logger.info(f"  {placement_label}")
        sh_score = None
        best_sh = {
            "label": placement_label,
            "recommendation": placement_label,
        }
    elif known_sites:
        # Use known safe harbor sites from species config
        logger.info(f"  Using {len(known_sites)} known safe harbor sites")
        # If genome data is available, validate known sites with real scoring
        logger.info("  Validating known sites with genome data...")
        predictor = _make_predictor()
        best_sh = None
        sh_score = 0.0
        for site in known_sites:
            chr_name = str(site.get("chromosome", "1"))
            pos = int(str(site.get("position", "0")).replace(",", ""))
            result = predictor.score_position(chr_name, pos, insert_length=800)
            if result["overall_score"] > sh_score:
                sh_score = result["overall_score"]
                best_sh = result
        if best_sh is None:
            best_sh = known_sites[0]
            sh_score = 0.7
        placement_label = "Genome-aware heuristic"
    else:
        # Use computational prediction with real genome data if available
        logger.info("  No known safe harbor sites — using computational prediction")
        logger.info("  Loading genome data for computational prediction...")
        predictor = _make_predictor()
        sh_result = predictor.score_position("1", 1000000, insert_length=800)
        sh_score = sh_result["overall_score"]
        best_sh = sh_result
        placement_label = "Genome-aware heuristic"
    for cid in scored:
        scored[cid]["safe_harbor_score"] = sh_score
        scored[cid]["safe_harbor_detail"] = best_sh
        scored[cid]["safe_harbor_label"] = placement_label
    step_timings["safe_harbor_s"] = round(time.perf_counter() - t0, 3)

    # ── Step 7: Protein-level analysis (optional) ────────────────────
    localisation_results = {}
    yield_results = {}

    if protein_seq:
        logger.info("Step 7: Protein-level analysis...")
        t0 = time.perf_counter()

        # DeepLoc localisation
        try:
            from modules.protein.deeploc import predict_localisation, assess_yield_impact
            loc = predict_localisation(protein_seq)
            yield_impact = assess_yield_impact(loc)
            localisation_results["default"] = {**loc, "yield_impact": yield_impact}
            logger.info(f"  Predicted localisation: {loc['predicted_localisation']} "
                        f"(confidence: {loc['confidence']})")
        except Exception as e:
            logger.info(f"  DeepLoc unavailable: {e}")

        # FBA yield prediction
        try:
            from modules.protein.yield_predictor import load_metabolic_model, predict_yield
            model_key = species_config.get("species", {}).get("common_name", "").lower()
            # Map common names to yield_predictor keys
            model_map = {
                "lab tobacco": None,  # No metabolic model for N. benthamiana
                "arabidopsis": "arabidopsis",
                "thale cress": "arabidopsis",
                "soybean": "soybean",
                "maize": "maize",
                "rice": "rice",
            }
            yield_key = model_map.get(model_key)
            if yield_key:
                metabolic_model = load_metabolic_model(yield_key)
                yield_pred = predict_yield(metabolic_model, protein_seq, yield_key)
                yield_results["default"] = yield_pred
                logger.info(f"  FBA yield: {yield_pred['growth_rate']}x growth, "
                            f"{yield_pred['max_yield_tsp']}% TSP")
            else:
                logger.info(f"  No metabolic model for {species_name}")
        except Exception as e:
            logger.info(f"  FBA yield prediction unavailable: {e}")
        step_timings["protein_analysis_s"] = round(time.perf_counter() - t0, 3)

    # ── Step 8: Pareto ranking ───────────────────────────────────────
    logger.info("Step 8: Multi-objective Pareto ranking...")
    t0 = time.perf_counter()

    # Build DataFrame
    import pandas as pd
    rows = {}
    for cid, data in scored.items():
        rows[cid] = {
            "sequence": data["sequence"],
            "weighted_score": data["weighted_score"],
            "novelty_35s": data["novelty_35s"],
            "internal_div": data["internal_div"],
            "silencing_risk": data["silencing_risk"],
            "safe_harbor_score": data.get("safe_harbor_score"),
            "embedding_similarity": data.get("embedding_similarity", 0.5),
            "gc_pct": data["gc_pct"],
            "passed_filters": data["passed_filters"],
            "length_bp": len(data["sequence"]),
            "history_similarity_penalty": data.get("history_similarity_penalty", 0.0),
            "raw_weighted_score": data.get("_raw_weighted_score", data["weighted_score"]),
            "promoter_class": data.get("promoter_class", "unclassified"),
            "safe_harbor_label": data.get("safe_harbor_label", "Heuristic (no genome data)"),
        }

        expr_detail = compute_hybrid_expression(
            data,
            silencing_risk=data["silencing_risk"],
            embedding_similarity=data.get("embedding_similarity", 0.5),
            internal_div=data["internal_div"],
        )
        expr_score = expr_detail["expression_score"]
        expr_ctx = apply_context(expr_score)
        rel_strength = round(expr_score / BASELINES[species_kind], 4)

        data["expression_score"] = expr_score
        data["expression_label"] = expr_detail["expression_class"]
        data["expression_class"] = expr_detail["expression_class"]
        data["relative_strength"] = rel_strength
        data["expression_context"] = expr_ctx
        data["cassette"] = CASSSETTE_TEMPLATE
        data["expression_detail"] = expr_detail

        rows[cid]["expression_score"] = expr_score
        rows[cid]["expression_label"] = data["expression_label"]
        rows[cid]["expression_class"] = data["expression_class"]
        rows[cid]["relative_strength"] = rel_strength
        rows[cid]["expr_safe"] = expr_ctx["safe_harbor"]
        rows[cid]["expr_random"] = expr_ctx["random"]
        rows[cid]["expr_repressed"] = expr_ctx["repressed"]
        rows[cid]["cassette"] = CASSSETTE_TEMPLATE
        rows[cid]["baseline_reference"] = baseline_reference_name(species_config)
        rows[cid]["occupancy"] = expr_detail["occupancy"]
        rows[cid]["binding_score"] = expr_detail["binding_score"]
        rows[cid]["expression_penalty"] = expr_detail["penalty"]
        rows[cid]["spacing_penalty"] = expr_detail["spacing_penalty"]
        rows[cid]["expression_confidence"] = expr_detail["confidence"]
        rows[cid]["norm_cis"] = expr_detail["norm_cis"]
        rows[cid]["norm_gc"] = expr_detail["norm_gc"]
        rows[cid]["norm_silencing"] = expr_detail["norm_silencing"]
        rows[cid]["norm_embed"] = expr_detail["norm_embed"]
        rows[cid]["norm_div"] = expr_detail["norm_div"]
        rows[cid]["tata_position"] = expr_detail["tata_position"]
        rows[cid]["caat_position"] = expr_detail["caat_position"]

        # Yield TSP: use localisation-based estimate if available, else default
        if yield_results:
            rows[cid]["yield_tsp"] = yield_results["default"].get("max_yield_tsp", 5.0)
        else:
            rows[cid]["yield_tsp"] = 5.0  # Default estimate

        # Discovery bonus: ML-generated candidates that pass filters get a boost
        # This counters the mutational dominance bias (Gap 3).
        # Rationale: ML models explore sequence space beyond known motifs,
        # so candidates that pass filters despite not being explicitly
        # constructed are more likely to represent novel biology.
        is_ml = cid.startswith("evo2_") or cid.startswith("d3lm_")
        if is_ml and data["passed_filters"]:
            rows[cid]["weighted_score"] = data["weighted_score"] * 1.3  # 30% discovery bonus
            logger.info(f"    Discovery bonus: {cid} (+30% strength)")

    df = pd.DataFrame(rows).T

    if len(df) == 0:
        return {
            "iteration": iteration,
            "status": "no_scored",
            "top_candidate": None,
            "failure_reasons": failure_reasons,
        }

    # Pareto ranking across 7 objectives
    objectives = DEFAULT_OBJECTIVES["names"]
    higher_is_better = DEFAULT_OBJECTIVES["higher_is_better"]
    weights = DEFAULT_OBJECTIVES["weights"]

    # Only use objectives that exist in df
    available_objectives = [
        o for o in objectives
        if o in df.columns and (o != "safe_harbor_score" or genome_available)
    ]

    if len(df) >= 2:
        df["pareto_front"] = pareto_rank_v2(df, available_objectives, higher_is_better)
    else:
        df["pareto_front"] = 1

    df["base_composite_score"] = compute_composite_v2(
        df, available_objectives, higher_is_better, weights
    )
    df["realism_regularizer"] = (
        (df["raw_weighted_score"].astype(float) - df["weighted_score"].astype(float))
        .clip(lower=0.0) / 100.0
    )
    df["composite_score"] = (
        df["base_composite_score"]
        - df["history_similarity_penalty"].astype(float)
        - df["realism_regularizer"].astype(float)
    ).clip(lower=0.0)

    df = df.sort_values(["pareto_front", "composite_score"], ascending=[True, False])

    n_front1 = (df["pareto_front"] == 1).sum()
    logger.info(f"  Pareto front 1: {n_front1} candidates (of {len(df)})")
    logger.info(f"  Composite score range: {df['composite_score'].min():.3f} - "
                f"{df['composite_score'].max():.3f}")
    step_timings["pareto_ranking_s"] = round(time.perf_counter() - t0, 3)

    # ── Step 9: Report cards for top candidates ──────────────────────
    # Baseline comparison: score known promoters as references (Gap 8)
    logger.info("Step 9: Generating report cards + baseline comparison...")
    t0 = time.perf_counter()
    baselines = {}
    ref_promoters = {
        "CaMV_35S": (
            "TGACGTAAAGGATCCCGTGTGGAATGTAAAAAGAATGAGCGCAAGACCTTCCAGATC"
            "TTTCCAAACTCTCCAAGCGCACGATCTTCAACTCTTCTCCACCATGGTGTCCAGAAG"
            "GTGTTTGAGCACTTCAACGAGCAGCAGTCCAAATATCAGTACCCACAGTATCTTGCC"
            "GTCAATGGTGACCTTAATGCTTTCTCTTAACATGGTTTATCCATTCGTTCAATCCAC"
            "TCTTAAGGCCTTTTAATATGGTGGAGATCATCACTTTTGGTCTCTCCAATCTTTAGC"
        ),
        "Maize_Ubiquitin": (
            "GTCTTTACAGATCTCTCTCTCTCTCTCTCTCTCTCTCAACACTCTCTCTCTCTCTCTC"
            "ATATAAGGGGTGGGTTTGTTTGTTTGTTGGGGTGGGGATGTGGGGAGATGATGGGG"
            "AGGGGATGCTGCAGGCATGCCGCTGCAGGTACCCAAGCTGGGGATCCATGGATATGG"
        ),
    }
    for ref_name, ref_seq in ref_promoters.items():
        ref_result = score_candidate(ref_seq, species_config)
        baselines[ref_name] = {
            "weighted_score": ref_result["weighted_score"],
            "gc_pct": ref_result["gc_pct"],
            "passed_filters": ref_result["passed_filters"],
        }
    selected_baseline = baseline_reference_name(species_config)
    baseline_str = " | ".join(
        f"{name}: {b['weighted_score']:.0f} ({'PASS' if b['passed_filters'] else 'FAIL'})"
        for name, b in baselines.items()
    )
    top_score = df.iloc[0]["weighted_score"]
    logger.info(f"  Baselines: {baseline_str}")
    logger.info(f"  Top candidate ({df.index[0]}): {top_score:.0f} — "
                f"vs best baseline: {'BETTER' if top_score > max(b['weighted_score'] for b in baselines.values()) else 'WORSE'}")

    top_n = min(3, len(df))
    report_cards = {}

    for rank, (cid, row) in enumerate(df.head(top_n).iterrows(), 1):
        seq = row["sequence"]
        silencing_detail = scored[cid].get("silencing_detail", {})
        sh_detail = scored[cid].get("safe_harbor_detail", {})

        # Build scores dict for report card
        report_scores = {
            "weighted_score": row["weighted_score"],
            "TATA_box": scored[cid].get("TATA_box", 0),
            "CAAT_box": scored[cid].get("CAAT_box", 0),
            "as1_element": scored[cid].get("as1_element", 0),
            "GCN4_motif": scored[cid].get("GCN4_motif", 0),
            "ocs_like": scored[cid].get("ocs_like", 0),
            "gc_pct": row["gc_pct"],
            "composite_score": row["composite_score"],
            "pareto_front": int(row["pareto_front"]),
            "relative_strength": row["relative_strength"],
            "baseline_reference": selected_baseline,
            "cassette": row["cassette"],
            "promoter_class": row["promoter_class"],
            "baseline_scores": baselines,
            "baseline_verdict": "BETTER" if row["weighted_score"] > max(b["weighted_score"] for b in baselines.values()) else "NOT BETTER",
        }

        expression_dict = {
            "expression_class": row["expression_class"],
            "expression_score": row["expression_score"],
            "relative_strength": row["relative_strength"],
            "baseline_reference": selected_baseline,
            "expr_safe": row["expr_safe"],
            "expr_random": row["expr_random"],
            "expr_repressed": row["expr_repressed"],
            "occupancy": row["occupancy"],
            "binding_score": row["binding_score"],
            "penalty": row["expression_penalty"],
            "spacing_penalty": row["spacing_penalty"],
            "confidence": row["expression_confidence"],
            "norm_cis": row["norm_cis"],
            "norm_gc": row["norm_gc"],
            "norm_silencing": row["norm_silencing"],
            "norm_embed": row["norm_embed"],
            "norm_div": row["norm_div"],
            "tata_position": row["tata_position"],
            "caat_position": row["caat_position"],
        }

        # Novelty dict
        novelty_dict = {
            "novelty_35s": row["novelty_35s"],
            "internal_div": row["internal_div"],
        }

        # Safe harbor dict
        sh_dict = None
        if isinstance(sh_detail, dict) and "chromosome" in sh_detail:
            sh_dict = {**sh_detail, "label": scored[cid].get("safe_harbor_label", placement_label)}
        elif isinstance(sh_detail, dict) and "scores" in sh_detail:
            sh_dict = {**sh_detail, "label": scored[cid].get("safe_harbor_label", placement_label)}
        elif isinstance(sh_detail, dict):
            sh_dict = {**sh_detail, "label": scored[cid].get("safe_harbor_label", placement_label)}

        # Localisation dict
        loc_dict = localisation_results.get("default") if localisation_results else None
        yield_dict = yield_results.get("default") if yield_results else None

        card = generate_report_card(
            candidate_id=cid,
            sequence=seq,
            species=species_name,
            scores=report_scores,
            expression=expression_dict,
            localisation=loc_dict if loc_dict else None,
            silencing=silencing_detail,
            safe_harbor=sh_dict,
            yield_pred=yield_dict,
            novelty=novelty_dict,
        )

        report_cards[cid] = card

        # Save report card to file
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, f"iter{iteration}_rank{rank}_{cid}.txt")
            with open(report_path, "w") as f:
                f.write(card)
            logger.info(f"  Report card saved: {report_path}")
    step_timings["report_cards_s"] = round(time.perf_counter() - t0, 3)

    # ── Step 10: Select top candidate as next seed ───────────────────
    top_cid = df.index[0]
    top_seq = df.loc[top_cid, "sequence"]
    top_composite = df.loc[top_cid, "composite_score"]

    logger.info(f"  Top candidate: {top_cid} (composite={top_composite:.3f}, "
                f"pareto_front={int(df.loc[top_cid, 'pareto_front'])})")
    logger.info(f"  Strength={df.loc[top_cid, 'weighted_score']}, "
                f"Silencing={df.loc[top_cid, 'silencing_risk']:.3f}, "
                f"Novelty={df.loc[top_cid, 'novelty_35s']:.3f}")
    logger.info(f"  Expression={df.loc[top_cid, 'expression_label']} "
                f"(score={df.loc[top_cid, 'expression_score']:.3f}, "
                f"relative={df.loc[top_cid, 'relative_strength']:.3f})")

    # Save iteration results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, f"iter{iteration}_scored.csv")
        df.to_csv(results_path)
        logger.info(f"  Scored candidates saved: {results_path}")

    iteration_wall_time = round(time.perf_counter() - iter_t0, 3)
    logger.info("  Compute summary:")
    logger.info(f"    Requested models: {', '.join(requested_models) if requested_models else 'none'}")
    logger.info(
        "    Actual candidate sources: "
        f"evo2={actual_model_counts['evo2']}, "
        f"d3lm={actual_model_counts['d3lm']}, "
        f"mutational={actual_model_counts['mutational']}, "
        f"other={actual_model_counts['other']}"
    )
    logger.info(
        f"    Evaluations: generated={len(candidates)}, "
        f"passed_filters={len(scored)}, "
        f"embedded={len(scored)}, "
        f"silencing_scored={len(scored)}"
    )
    logger.info(f"    Wall time: {iteration_wall_time:.3f}s")
    logger.info("    Step timings (s): " + ", ".join(f"{k}={v:.3f}" for k, v in step_timings.items()))

    return {
        "iteration": iteration,
        "status": "complete",
        "top_candidate": top_seq,
        "top_candidate_id": top_cid,
        "requested_models": requested_models,
        "actual_model_counts": actual_model_counts,
        "step_timings_s": step_timings,
        "iteration_wall_time_s": iteration_wall_time,
        "top_composite_score": float(top_composite),
        "n_candidates": len(candidates),
        "n_passed_filters": len(scored),
        "n_pareto_front1": int(n_front1),
        "report_cards": report_cards,
        "failure_reasons": failure_reasons,
        "top_weighted_score": float(df.loc[top_cid, "weighted_score"]),
        "top_silencing_risk": float(df.loc[top_cid, "silencing_risk"]),
        "top_novelty": float(df.loc[top_cid, "novelty_35s"]),
        "top_expression_score": float(df.loc[top_cid, "expression_score"]),
        "top_expression_label": str(df.loc[top_cid, "expression_label"]),
        "top_expression_confidence": float(df.loc[top_cid, "expression_confidence"]),
        "top_relative_strength": float(df.loc[top_cid, "relative_strength"]),
        "top_expr_safe": float(df.loc[top_cid, "expr_safe"]),
        "top_expr_random": float(df.loc[top_cid, "expr_random"]),
        "top_expr_repressed": float(df.loc[top_cid, "expr_repressed"]),
        "top_cassette": str(df.loc[top_cid, "cassette"]),
        "top_baseline_reference": str(df.loc[top_cid, "baseline_reference"]),
    }


# ─────────────────────────────────────────────────────────────────────
# REFERENCE PROMOTER SEED LOADING
# ─────────────────────────────────────────────────────────────────────

def load_seed(species_key: str) -> str:
    """Load the initial seed promoter for a species.
    Uses species-specific FASTAs in configs/seeds/.

    Returns:
        DNA sequence string
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(script_dir, "..")
    seed_key = SEED_FILE_ALIASES.get(species_key, species_key)
    fasta_path = os.path.join(project_dir, "configs", "seeds", f"{seed_key}.fasta")

    if not os.path.exists(fasta_path):
        return CURATED_FALLBACK_SEEDS.get(seed_key, CURATED_FALLBACK_SEEDS["nbenthamiana"])

    seq = _load_first_fasta_sequence(fasta_path)
    if not seq:
        return CURATED_FALLBACK_SEEDS.get(seed_key, CURATED_FALLBACK_SEEDS["nbenthamiana"])

    return seq


# ─────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V2 Auto-Research Loop")
    parser.add_argument("--species", required=True,
                        help="Target species (e.g., nbenthamiana, maize, arabidopsis)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of iterations to run (default: 5)")
    parser.add_argument("--models", default="evo2+d3lm",
                        help="Generation models: evo2, d3lm, or evo2+d3lm (default: evo2+d3lm)")
    parser.add_argument("--variants", type=int, default=20,
                        help="Number of variants per model per iteration (default: 20)")
    parser.add_argument("--protein-seq", default=None,
                        help="Protein amino acid sequence for DeepLoc + yield prediction")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: v2_research/outputs/<species>)")
    args = parser.parse_args()

    # Load species config
    species_config = load_species_config(args.species)
    species_name = species_config["species"]["name"]
    species_config["_config_key"] = args.species  # Store key for lookups

    # Setup
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "outputs", args.species,
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir = os.path.abspath(output_dir)
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    logger = setup_logging(log_dir, args.species)

    logger.info("=" * 60)
    logger.info("V2 AUTO-RESEARCH LOOP")
    logger.info("=" * 60)
    logger.info(f"Species: {species_name}")
    logger.info(f"Type: {'Monocot' if is_monocot(species_config) else 'Dicot'}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Iterations: {args.iterations}")
    logger.info(f"Variants/model/iteration: {args.variants}")
    logger.info(f"Output: {output_dir}")
    if args.protein_seq:
        logger.info(f"Protein seq: {args.protein_seq[:30]}... ({len(args.protein_seq)} aa)")

    # Get initial seed
    seed = load_seed(args.species)
    logger.info(f"Initial seed: {len(seed)} bp reference promoter")
    logger.info("")

    # Run loop
    loop_t0 = time.perf_counter()
    results = []
    best_composite = -1
    best_seed = seed
    failure_feedback = {}

    for i in range(1, args.iterations + 1):
        try:
            seed_for_iteration = apply_failure_feedback(seed, failure_feedback, logger)
            result = run_single_iteration(
                species_config=species_config,
                seed=seed_for_iteration,
                iteration=i,
                logger=logger,
                models=args.models,
                n_variants=args.variants,
                protein_seq=args.protein_seq,
                output_dir=output_dir,
            )
            results.append(result)
            failure_feedback = result.get("failure_reasons", {})

            # Update seed from top candidate
            if result.get("top_candidate"):
                seed = result["top_candidate"]
                logger.info(f"New seed selected from iteration {i} "
                            f"(composite={result['top_composite_score']:.3f})")

                # Track best across all iterations
                if result["top_composite_score"] > best_composite:
                    best_composite = result["top_composite_score"]
                    best_seed = seed
                    logger.info(f"  New best composite score: {best_composite:.3f}")

        except Exception as e:
            logger.error(f"Iteration {i} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("Continuing with previous seed...")

        logger.info("")
        logger.info(f"Completed {i}/{args.iterations} iterations")
        logger.info("")

    # ── Summary ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Species: {species_name}")
    logger.info(f"Total iterations: {len(results)}")

    completed = [r for r in results if r["status"] == "complete"]
    if completed:
        composites = [r["top_composite_score"] for r in completed]
        logger.info(f"Composite score progression: "
                    + " -> ".join(f"{c:.3f}" for c in composites))
        logger.info(f"Best composite: {max(composites):.3f}")

        total_candidates = sum(r["n_candidates"] for r in completed)
        total_passed = sum(r["n_passed_filters"] for r in completed)
        total_wall = round(sum(r.get("iteration_wall_time_s", 0.0) for r in completed), 3)
        source_totals = {"evo2": 0, "d3lm": 0, "mutational": 0, "other": 0}
        for r in completed:
            for key, value in r.get("actual_model_counts", {}).items():
                source_totals[key] = source_totals.get(key, 0) + value
        logger.info(f"Total candidates generated: {total_candidates}")
        logger.info(f"Total passed hard filters: {total_passed}")
        logger.info(
            "Actual candidate sources total: "
            f"evo2={source_totals['evo2']}, "
            f"d3lm={source_totals['d3lm']}, "
            f"mutational={source_totals['mutational']}, "
            f"other={source_totals['other']}"
        )
        logger.info(f"Approx compute wall time: {total_wall:.3f}s")
    else:
        logger.info("No completed iterations")

    # Print top candidate info
    logger.info("")
    logger.info(f"Best seed sequence ({len(best_seed)} bp):")
    logger.info(f"  {best_seed[:80]}...")

    # Save summary
    summary_path = os.path.join(output_dir, "loop_summary.json")
    summary = {
        "species": args.species,
        "species_name": species_name,
        "iterations_requested": args.iterations,
        "iterations_completed": len(completed),
        "models": args.models,
        "requested_models": _requested_model_list(args.models),
        "best_composite_score": best_composite if completed else None,
        "best_seed_length": len(best_seed),
        "loop_wall_time_s": round(time.perf_counter() - loop_t0, 3),
        "results": [
            {
                "iteration": r["iteration"],
                "status": r["status"],
                "n_candidates": r.get("n_candidates", 0),
                "n_passed_filters": r.get("n_passed_filters", 0),
                "actual_model_counts": r.get("actual_model_counts", {}),
                "step_timings_s": r.get("step_timings_s", {}),
                "iteration_wall_time_s": r.get("iteration_wall_time_s"),
                "top_composite_score": r.get("top_composite_score"),
                "top_candidate_id": r.get("top_candidate_id"),
                "top_expression_score": r.get("top_expression_score"),
                "top_expression_label": r.get("top_expression_label"),
                "top_expression_confidence": r.get("top_expression_confidence"),
                "top_relative_strength": r.get("top_relative_strength"),
                "top_expr_safe": r.get("top_expr_safe"),
                "top_expr_random": r.get("top_expr_random"),
                "top_expr_repressed": r.get("top_expr_repressed"),
                "top_cassette": r.get("top_cassette"),
                "top_baseline_reference": r.get("top_baseline_reference"),
            }
            for r in results
        ],
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved: {summary_path}")

    # Save best seed as FASTA
    best_fasta_path = os.path.join(output_dir, "best_seed.fasta")
    with open(best_fasta_path, "w") as f:
        f.write(f">best_seed_{args.species}_composite{best_composite:.3f}\n")
        f.write(best_seed + "\n")
    logger.info(f"Best seed FASTA: {best_fasta_path}")


if __name__ == "__main__":
    main()
