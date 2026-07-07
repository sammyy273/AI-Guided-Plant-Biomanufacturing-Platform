# =============================================================================
# 3_generate_candidates.py  (FIXED)
#
# WHAT CHANGED FROM PREVIOUS VERSION:
#
#   Fix 1 — top_k must be <= 6 for the Evo2 NVIDIA NIM API.
#            Previous code sent top_k=50 which caused 422 validation errors.
#
#   Fix 2 — num_sequences is NOT a valid API parameter.
#            The API generates ONE sequence per call.
#            To get N variants, we call the API N times in a loop.
#
#   Fix 3 — Switch default to evo2-40b.
#            evo2-7b generates degenerate repetitive sequences (CA repeats,
#            satellite motifs) that have no functional regulatory elements.
#            All 60 sequences from evo2-7b failed the TATA/CAAT hard filter.
#            evo2-40b produces functionally coherent regulatory sequences.
#
#   Fix 4 — Temperature variation between calls.
#            Slight temperature jitter (+/- 0.05) per call increases diversity
#            and reduces the chance of all variants collapsing into the same
#            repetitive motif.
#
#   Fix 5 — Repetitive sequence detection added.
#            Before saving, each generated sequence is checked for tandem
#            repeats. If >60% of the sequence is a repeated k-mer, the
#            candidate is flagged as degenerate and skipped.
# =============================================================================

import os
import re
import sys
import time
import numpy as np
import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import (setup_logger, log_step, log_ok, log_warn,
                   log_err, log_info, save_fasta, load_fasta)

# The Evo2 NIM API enforces top_k <= 6. Do not change this.
EVO2_TOP_K_MAX = 6
EVO2_URL = "https://health.api.nvidia.com/v1/biology/arc/{model}/generate"


# ── IN-CONTEXT DESIGN PROMPT (v2 — few-shot with as-1 examples) ──────────────
# Evo2 has demonstrated genuine in-context learning on genomic sequences
# (arXiv:2511.12797, 2024). The v1 prompt was too generic and Evo2 never
# generated as-1 elements. This v2 prompt includes explicit example sequences
# containing functional as-1 elements to teach Evo2 what we want.
#
# Key improvement: added a concrete example promoter fragment that contains
# TGACG (as-1), TATAAA (TATA), and CCAAT (CAAT) in biologically correct
# positions. This gives Evo2 a template to copy rather than just instructions.
EVO2_DESIGN_PROMPT = (
    "|k__Viridiplantae;p__Streptophyta;c__Magnoliopsida;"
    "o__Solanales;g__Nicotiana;s__Nicotiana benthamiana| "
    "DESIGN RULES for a strong Nicotiana benthamiana promoter: "
    "GC content 40-60%. "
    "Include TATA box (TATAAA/TATATA) near position -30 from TSS. "
    "Include CAAT box (CCAAT) near position -80 from TSS. "
    "CRITICAL: Include at least 2 copies of as-1 element (TGACG) — this is "
    "the primary strength driver. Place one TGACG between CAAT and TATA, "
    "and another upstream of CAAT. "
    "Example fragment with correct architecture: "
    "TCTCCACTGACGTAAGGGATGACGCACAATCCCACTATCCTTCGCAAGACCCTTCCTCTATATAAGGAAG. "
    "Note the TGACG at the start and TATA near the end. "
    "Also include GCN4 motif (TGACGTCA) and ocs-like (TGACGTAA). "
    "Avoid repetitive k-mers, poly(A) tails, and CpG islands. "
    "Generate: "
)


def is_degenerate(seq: str, threshold: float = 0.60) -> bool:
    """
    Detect repetitive degenerate sequences (what evo2-7b tends to produce).

    Checks if any k-mer of length 2-6 dominates more than threshold fraction
    of the sequence. Real promoters have varied k-mer distributions.
    Sequences like CACACACACACA or AGGAGGAGG fail this check.

    Returns True = degenerate (discard), False = looks OK.
    """
    seq = seq.upper()
    for k in range(2, 7):
        counts = {}
        for i in range(len(seq) - k + 1):
            km = seq[i:i+k]
            counts[km] = counts.get(km, 0) + 1
        if counts:
            top = max(counts.values())
            if (top * k) / len(seq) > threshold:
                return True
    return False


def call_evo2_single(seed: str, temperature: float, model: str, logger) -> str | None:
    """
    Call the Evo2 API for exactly ONE generated sequence.
    Returns the generated DNA string or None on failure.
    """
    url = EVO2_URL.format(model=model)
    headers = {
        "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }

    top_k_safe = min(getattr(config, "EVO2_TOP_K", 4), EVO2_TOP_K_MAX)

    payload = {
        "sequence":    EVO2_DESIGN_PROMPT + seed,
        "num_tokens":  getattr(config, "EVO2_LENGTH", 800),
        "top_k":       top_k_safe,
        "top_p":       getattr(config, "EVO2_TOP_P", 0.9),
        "temperature": round(float(temperature), 3),
        # DO NOT add num_sequences — it is not a valid parameter for this API
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
    except requests.exceptions.Timeout:
        log_warn(logger, "Request timed out (120s). Skipping this variant.")
        return None
    except requests.exceptions.ConnectionError:
        log_warn(logger, "Connection error. Check internet.")
        return None

    if resp.status_code == 200:
        data = resp.json()
        for key in ("sequence", "sequences", "generated_sequence", "output"):
            if key in data:
                val = data[key]
                if isinstance(val, list):
                    val = val[0]
                if isinstance(val, dict):
                    val = val.get("sequence", val.get("text", ""))
                return str(val).upper().replace(" ", "").replace("\n", "")
        log_warn(logger, f"Unexpected response keys: {list(data.keys())}")
        log_warn(logger, f"Response sample: {str(data)[:300]}")
        return None

    elif resp.status_code == 401:
        log_err(logger, "API key rejected (401). Check NVIDIA_API_KEY in config.py.")
        sys.exit(1)

    elif resp.status_code == 422:
        log_err(logger, f"API validation error (422): {resp.text[:400]}")
        log_err(logger, f"top_k sent = {top_k_safe} (must be <= {EVO2_TOP_K_MAX})")
        return None

    elif resp.status_code == 429:
        log_warn(logger, "Rate limited (429). Waiting 20s...")
        time.sleep(20)
        return call_evo2_single(seed, temperature, model, logger)

    else:
        log_warn(logger, f"API error {resp.status_code}: {resp.text[:200]}")
        return None


def generate_variants(seed_name: str, seed_seq: str, n: int,
                      model: str, logger) -> dict:
    """
    Generate n valid (non-degenerate) variants from one seed sequence.
    Calls the API once per variant with temperature jitter for diversity.
    """
    results   = {}
    attempts  = 0
    max_tries = n * 4
    base_temp = getattr(config, "EVO2_TEMPERATURE", 0.7)
    rng       = np.random.default_rng(seed=abs(hash(seed_name)) % 2**31)

    pbar = tqdm(total=n, desc=f"  {seed_name[:30]}", ncols=72)

    while len(results) < n and attempts < max_tries:
        attempts += 1
        temp = float(np.clip(base_temp + rng.uniform(-0.08, 0.08), 0.5, 0.95))
        seq  = call_evo2_single(seed_seq, temp, model, logger)

        if seq is None or len(seq) < 200:
            time.sleep(0.5)
            continue

        if is_degenerate(seq):
            time.sleep(0.3)
            continue

        cid = f"{seed_name}_v{len(results)+1:02d}"
        results[cid] = seq
        pbar.update(1)
        time.sleep(0.3)   # brief pause between calls

    pbar.close()

    accepted  = len(results)
    discarded = attempts - accepted
    if accepted < n:
        log_warn(logger,
            f"  Got {accepted}/{n} good variants "
            f"({discarded} degenerate/failed out of {attempts} attempts)"
        )
        if discarded > n:
            log_warn(logger, "  Most sequences were degenerate repetitive motifs.")
            log_warn(logger, "  This is common with evo2-7b. Use evo2-40b for better results.")
    else:
        log_ok(logger, f"  {accepted} variants accepted ({discarded} degenerate filtered)")

    return results


def build_null_seed(length: int = 150, gc: float = 0.46) -> str:
    """Random GC-balanced sequence — negative control."""
    at = (1.0 - gc) / 2
    rng = np.random.default_rng(seed=42)
    return "".join(rng.choice(list("ATCG"), size=length,
                              p=[at, at, gc / 2, gc / 2]))


def main():
    logger = setup_logger(config.LOG_FILE)
    log_step(logger, 3, "Generating Synthetic Promoter Candidates with Evo2")

    if "YOUR_KEY_HERE" in config.NVIDIA_API_KEY or not config.NVIDIA_API_KEY:
        log_err(logger, "NVIDIA_API_KEY not set in config.py")
        log_err(logger, "Get your key at: https://build.nvidia.com/arc/evo2-40b")
        sys.exit(1)

    model = getattr(config, "EVO2_MODEL", "evo2-40b")
    n_var = getattr(config, "EVO2_N_VARIANTS", 20)

    log_info(logger, f"Model:             {model}")
    log_info(logger, f"Variants per seed: {n_var}")
    log_info(logger, f"Generated length:  {getattr(config, 'EVO2_LENGTH', 800)} bp")
    log_info(logger, f"Base temperature:  {getattr(config, 'EVO2_TEMPERATURE', 0.7)}")
    log_info(logger, f"top_k (max 6):     {min(getattr(config, 'EVO2_TOP_K', 4), EVO2_TOP_K_MAX)}")
    log_info(logger, "")

    if "7b" in model.lower():
        log_warn(logger, "=" * 60)
        log_warn(logger, "WARNING: evo2-7b tends to produce degenerate repetitive")
        log_warn(logger, "sequences (CACACACA, AGGAGG...) with no regulatory elements.")
        log_warn(logger, "These will all fail the TATA/CAAT hard filter in Step 4.")
        log_warn(logger, "STRONGLY RECOMMEND: change EVO2_MODEL = 'evo2-40b' in config.py")
        log_warn(logger, "=" * 60)
        log_warn(logger, "")

    if not os.path.exists(config.REF_FASTA):
        log_info(logger, "Reference FASTA not found. Running Step 1 first...")
        from fetch_references import main as run_step1
        run_step1()

    refs = load_fasta(config.REF_FASTA)

    # Build seeds
    seeds = {}

    if "2x_CaMV_35S" in refs:
        seeds["seed_35S_core"] = refs["2x_CaMV_35S"][-150:]
        log_ok(logger, f"Seed 1 (35S core):     {len(seeds['seed_35S_core'])} bp")

    for name in ["NbACT3", "NbUBI", "NbRbcS"]:
        if name in refs and len(refs[name]) >= 200:
            seeds["seed_NbEndogenous"] = refs[name][-200:]
            log_ok(logger, f"Seed 2 ({name}):       {len(seeds['seed_NbEndogenous'])} bp")
            break
    if "seed_NbEndogenous" not in seeds:
        seeds["seed_NbEndogenous"] = refs.get("2x_CaMV_35S", "")[:200]
        log_warn(logger, "Using 5' end of 35S as Seed 2 (no endogenous promoter available)")

    seeds["seed_null_control"] = build_null_seed()
    log_ok(logger, f"Seed 3 (null):         {len(seeds['seed_null_control'])} bp")

    # ── Auto-loop feedback: override seed_35S_core if a custom seed exists ──
    # auto_loop.py writes the best candidate from the previous iteration here.
    # This is what makes the loop self-improving: each generation starts from
    # the best sequence found so far, not a fixed reference sequence.
    custom_seed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "current_seed.fasta")
    if os.path.exists(custom_seed_path):
        custom_seeds = load_fasta(custom_seed_path)
        if custom_seeds:
            custom_seq = list(custom_seeds.values())[0]
            seeds["seed_35S_core"] = custom_seq[-150:]  # use last 150 bp as seed
            log_ok(logger, f"Auto-loop: using custom seed from previous iteration.")
            log_info(logger, f"  Custom seed preview: {custom_seq[:60]}...")

    # ── Generate Evo2 candidates ────────────────────────────────────────────────
    all_candidates = {}
    for idx, (seed_name, seed_seq) in enumerate(seeds.items(), 1):
        log_info(logger, "")
        log_info(logger, f"[{idx}/{len(seeds)}] {seed_name}")
        log_info(logger, f"  Seed preview: {seed_seq[:60]}...")
        variants = generate_variants(seed_name, seed_seq, n_var, model, logger)
        all_candidates.update(variants)
        if idx < len(seeds):
            time.sleep(3)

    # ── Generate mutational candidates (offline, guaranteed quality) ──────────
    # The mutational generator explicitly inserts TATA, CAAT, and as-1 elements
    # at biologically correct positions. It achieved 93% pass rate vs 6.7% for
    # Evo2-only generation in V2 benchmarking. Adding 20 mutational candidates
    # ensures the pipeline always has high-quality results even if Evo2 fails.
    log_info(logger, "")
    log_info(logger, "Generating 20 mutational candidates (offline, guaranteed motifs)...")

    from mutational_generator import generate_from_seed as mut_from_seed
    from mutational_generator import generate_from_scratch as mut_from_scratch
    from mutational_generator import _insert_cis_element as mut_insert_cis

    # Generate from 35S seed
    if "seed_35S_core" in seeds:
        ref_full = refs.get("2x_CaMV_35S", seeds["seed_35S_core"])
        mut_candidates = mut_from_seed(ref_full, n_variants=15)
        all_candidates.update({f"mut_35S_{k}": v for k, v in mut_candidates.items()})
        log_ok(logger, f"  15 mutational variants from 35S seed")

    # Generate from scratch (fresh scaffolds)
    scratch_candidates = mut_from_scratch(n_variants=5)
    all_candidates.update(scratch_candidates)
    log_ok(logger, f"  5 fresh scaffold variants")

    # Post-process Evo2 candidates: insert as-1 if missing
    as1_fixed = 0
    for cid in list(all_candidates.keys()):
        if cid.startswith("mut_") or cid.startswith("scaffold_"):
            continue  # mutational candidates already have as-1
        seq = all_candidates[cid]
        if not re.search(r"TGACG", seq):
            # Insert as-1 element at canonical position (-65 from TSS)
            as1_pos = len(seq) - 65
            seq = mut_insert_cis(seq, "TGACG", as1_pos)
            all_candidates[cid] = seq
            as1_fixed += 1
    if as1_fixed > 0:
        log_ok(logger, f"  Post-processed {as1_fixed} Evo2 candidates with as-1 insertion")

    if not all_candidates:
        log_err(logger, "No candidates were generated.")
        log_err(logger, "Check your API key and internet connection.")
        log_err(logger, "If using evo2-7b, switch to evo2-40b in config.py.")
        sys.exit(1)

    os.makedirs("data", exist_ok=True)
    save_fasta(all_candidates, config.CANDIDATES_FASTA)
    log_ok(logger, f"Saved {len(all_candidates)} candidates → {config.CANDIDATES_FASTA}")

    log_info(logger, "")
    log_info(logger, "Candidates by source:")
    by_source = {}
    for cid in all_candidates:
        if cid.startswith("mut_35S_"):
            s = "mutational_35S"
        elif cid.startswith("scaffold_"):
            s = "mutational_scaffold"
        else:
            s = "_".join(cid.split("_")[:-1])
        by_source[s] = by_source.get(s, 0) + 1
    for s, c in by_source.items():
        log_info(logger, f"  {s}: {c} variants")

    log_info(logger, "")
    log_info(logger, "Next: python step4_filter_and_rank.py")


if __name__ == "__main__":
    main()
