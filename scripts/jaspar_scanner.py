# =============================================================================
# jaspar_scanner.py
#
# WHAT THIS DOES:
#   Downloads plant-specific transcription factor binding profiles from the
#   JASPAR 2024 database and scans candidate promoter sequences against them
#   using Position Weight Matrix (PWM) scoring.
#
# WHY THIS REPLACES PLANTCARE:
#   PlantCARE was last substantially updated ~2015. It contains ~400 cis-element
#   patterns as simple consensus strings. JASPAR 2024 contains 1,745 plant TF
#   binding profiles as position frequency matrices (PFMs), which capture the
#   full binding specificity of each transcription factor — not just a single
#   consensus string. PWM scoring is far more sensitive and specific than
#   regex matching.
#
# HOW IT WORKS:
#   1. Downloads the top N most informative plant TF profiles from JASPAR API.
#   2. Converts PFMs to PWMs (log-odds scoring).
#   3. Slides each PWM across the candidate sequence.
#   4. Records hits above a score threshold (default: 80% of max PWM score).
#   5. Returns hit counts per TF, total hits, and a weighted strength score.
#
# KEY TF PROFILES FOR N. benthamiana PROMOTER STRENGTH:
#   TBP (TATA-binding protein) — TATA box recognition
#   NF-Y (CAAT-binding) — CAAT box recognition
#   TGA/bZIP — as-1 element binding (primary 35S strength driver)
#   bZIP — GCN4 motif binding
#   GBF — G-box binding
#   WRKY — W-box binding
#   MYB — MBS binding
#   AREB/ABF — ABRE binding
#
# REFERENCES:
#   JASPAR 2024: Khan et al., Nucleic Acids Research, 2024
#   FIMO: Grant et al., Bioinformatics, 2011 (scoring algorithm adapted)
# =============================================================================

import os
import math
import requests
import numpy as np
from collections import defaultdict


# JASPAR API base URL
JASPAR_API = "https://jaspar.genereg.net/api/v1"

# Key TF families for N. benthamiana promoter strength
# These get extra weight in scoring (matching the biology of 35S)
PRIORITY_TF_NAMES = [
    "TBP", "TGA", "bZIP", "GBF", "WRKY", "MYB", "ABF",
    "NF-Y", "GCN4", "OCS", "DOF", "bHLH",
]

# Cache directory for downloaded profiles
JASPAR_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "jaspar_cache")


def fetch_plant_profiles(max_profiles: int = 50,
                         min_information_content: float = 4.0) -> list:
    """
    Fetch plant TF binding profiles from JASPAR 2024 API.
    Returns list of dicts with 'matrix_id', 'name', 'pfm', 'length'.
    Only keeps profiles with sufficient information content.
    """
    os.makedirs(JASPAR_CACHE_DIR, exist_ok=True)

    # Check cache first
    cache_file = os.path.join(JASPAR_CACHE_DIR, "profiles.npz")
    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        profiles = list(data["profiles"])
        print(f"  [JASPAR] Loaded {len(profiles)} profiles from cache.")
        return profiles

    print(f"  [JASPAR] Fetching plant TF profiles from JASPAR 2024...")

    # Get list of plant profiles
    resp = requests.get(
        f"{JASPAR_API}/matrix/",
        params={
            "tax_group": "plants",
            "release": "2024",
            "format": "json",
            "page_size": 200,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    all_profiles = data.get("results", [])

    print(f"  [JASPAR] Found {data.get('count', 0)} plant profiles. "
          f"Downloading top {max_profiles}...")

    profiles = []
    for i, entry in enumerate(all_profiles[:max_profiles * 2]):  # fetch extra, filter later
        matrix_id = entry["matrix_id"]
        name = entry.get("name", "Unknown")

        try:
            # Fetch the full matrix with PFM
            mat_resp = requests.get(
                f"{JASPAR_API}/matrix/{matrix_id}/?format=json",
                timeout=15,
            )
            mat_resp.raise_for_status()
            mat_data = mat_resp.json()
            pfm = mat_data.get("pfm", {})

            if not pfm or len(pfm.get("A", [])) < 4:
                continue

            # Calculate information content
            ic = _information_content(pfm)
            if ic < min_information_content:
                continue

            profiles.append({
                "matrix_id": matrix_id,
                "name": name,
                "pfm": pfm,
                "length": len(pfm["A"]),
                "information_content": round(ic, 2),
            })

            if len(profiles) >= max_profiles:
                break

            # Small delay to not hammer the API
            if (i + 1) % 10 == 0:
                import time
                time.sleep(0.5)

        except Exception:
            continue

    # Sort by information content (most informative first)
    profiles.sort(key=lambda p: p["information_content"], reverse=True)

    # Cache
    np.savez(cache_file, profiles=np.array(profiles, dtype=object))
    print(f"  [JASPAR] Downloaded {len(profiles)} high-IC profiles. Cached.")

    return profiles


def _information_content(pfm: dict) -> float:
    """Calculate total information content of a PFM (bits)."""
    length = len(pfm["A"])
    total_ic = 0.0
    for i in range(length):
        counts = [pfm[b][i] for b in "ACGT"]
        total = sum(counts)
        if total == 0:
            continue
        ic_pos = 2.0  # max for DNA
        for c in counts:
            if c > 0:
                freq = c / total
                ic_pos -= freq * math.log2(freq)
        total_ic += ic_pos
    return total_ic


def pfm_to_pwm(pfm: dict, pseudocount: float = 0.5) -> dict:
    """
    Convert Position Frequency Matrix to Position Weight Matrix.
    Uses log-odds scoring with pseudocounts.
    Returns dict with 'A', 'C', 'G', 'T' arrays.
    """
    length = len(pfm["A"])
    pwm = {b: [] for b in "ACGT"}

    for i in range(length):
        total = sum(pfm[b][i] for b in "ACGT") + 4 * pseudocount
        for b in "ACGT":
            freq = (pfm[b][i] + pseudocount) / total
            # Background frequency = 0.25 (uniform)
            log_odds = math.log2(freq / 0.25) if freq > 0 else -10.0
            pwm[b].append(log_odds)

    return pwm


def score_pwm_on_sequence(pwm: dict, sequence: str) -> list:
    """
    Slide a PWM across a sequence and return all scores.
    Returns list of (position, score) tuples above threshold.
    """
    seq = sequence.upper()
    pwm_len = len(pwm["A"])
    scores = []

    # Calculate max possible score for thresholding
    max_score = sum(max(pwm[b][i] for b in "ACGT") for i in range(pwm_len))
    threshold = max_score * 0.75  # 75% of max

    for pos in range(len(seq) - pwm_len + 1):
        score = 0.0
        valid = True
        for i in range(pwm_len):
            base = seq[pos + i]
            if base not in "ACGT":
                valid = False
                break
            score += pwm[base][i]
        if valid and score >= threshold:
            scores.append((pos, round(score, 2)))

    return scores


def scan_sequence_jaspar(sequence: str, profiles: list) -> dict:
    """
    Scan a DNA sequence against all JASPAR profiles.
    Returns dict with hit counts per profile family, total hits,
    and a weighted strength score.
    """
    seq = sequence.upper()
    results = {}
    total_hits = 0
    priority_hits = 0

    for profile in profiles:
        pwm = pfm_to_pwm(profile["pfm"])
        hits = score_pwm_on_sequence(pwm, seq)
        hit_count = len(hits)
        results[profile["matrix_id"]] = {
            "name": profile["name"],
            "hits": hit_count,
            "positions": [h[0] for h in hits] if hit_count > 0 else [],
        }
        total_hits += hit_count

        # Check if this is a priority TF (strength driver)
        name_upper = profile["name"].upper()
        if any(p in name_upper for p in PRIORITY_TF_NAMES):
            priority_hits += hit_count

    # Weighted score: priority TFs count 5x, others 1x
    weighted = priority_hits * 5 + (total_hits - priority_hits) * 1

    return {
        "jaspar_total_hits": total_hits,
        "jaspar_priority_hits": priority_hits,
        "jaspar_weighted_score": weighted,
        "jaspar_profiles_matched": sum(1 for r in results.values() if r["hits"] > 0),
        "jaspar_details": results,
    }


def scan_all_candidates(candidates: dict, profiles: list = None,
                        max_profiles: int = 50) -> dict:
    """
    Scan all candidates against JASPAR plant profiles.
    Args:
        candidates: {id: sequence} dict
        profiles: pre-fetched profiles (or None to fetch fresh)
        max_profiles: how many JASPAR profiles to use
    Returns:
        {id: scan_results} dict
    """
    if profiles is None:
        profiles = fetch_plant_profiles(max_profiles=max_profiles)

    results = {}
    for cid, seq in candidates.items():
        results[cid] = scan_sequence_jaspar(seq, profiles)

    return results, profiles
