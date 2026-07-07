# =============================================================================
# 2_score_references.py
#
# WHAT THIS DOES:
#   Scans all reference promoters for known cis-regulatory elements.
#   Produces the "design target" profile — the element counts your synthetic
#   candidates must match or exceed to be considered strong promoters.
#
# WHAT IS A CIS-ELEMENT?
#   A short DNA sequence motif that a specific transcription factor (TF) binds
#   to. When the TF binds, it either activates or represses transcription.
#   A strong promoter has many activating cis-elements so many activating TFs
#   can bind simultaneously, recruiting RNA Polymerase II efficiently.
#
# KEY ELEMENTS SCANNED:
#   TATA_box    — Where RNA Pol II binds to start transcription. REQUIRED.
#   CAAT_box    — Efficiency enhancer. Present in ~80% of strong promoters.
#   G_box       — Light-responsive (CACGTG). Important for leaf expression.
#                 NOTE: 2×CaMV 35S has 0 G-boxes — it uses as-1 instead.
#   as1_element — Activation Sequence 1 (TGACG). Primary driver of CaMV 35S.
#   W_box       — WRKY TF binding. Defence/stress-responsive.
#   ABRE        — ABA-responsive. Drought stress.
#   MBS         — MYB binding. Drought.
#
# DESIGN RULE:
#   Your synthetic promoters MUST have TATA + CAAT (hard filters).
#   G-box absence is acceptable — as-1 presence compensates.
#   Total element count >= 2×35S is the target.
#
# OUTPUT:
#   data/reference_element_scores.csv
#   Printed comparison table
# =============================================================================

import os
import re
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import setup_logger, log_step, log_ok, log_info, load_fasta

# =============================================================================
# CIS-ELEMENT PATTERNS
# =============================================================================
#
# ELEMENT WEIGHTS (biological importance for promoter strength):
#   Primary drivers (5x)  — as-1 and GCN4 are the main strength amplifiers.
#                           35S promoter strength comes from tandem as-1 elements.
#                           GCN4 provides additional transcriptional activation.
#   Core elements (1x)    — TATA and CAAT are necessary but not sufficient alone.
#   Enhancer elements (2x) — Light-, stress-, and tissue-responsive elements
#                           that fine-tune expression when present.
#
# WEIGHT RATIONALE:
#   Literature: as-1 copy number correlates directly with 35S promoter strength.
#   Deleting as-1 elements from 35S reduces activity by ~60%.
#   GCN4 (TGACTCA) binds the GCN4 TF and synergizes with as-1.
#   TATA+CAAT alone produce minimal transcription (minimal promoter level).
#
# IUPAC CODES used:
#   R = A or G (purine)     Y = C or T (pyrimidine)
#   W = A or T              S = G or C
#   K = G or T              M = A or C
#   N = any nucleotide
#
CIS_ELEMENTS = {
    # Core (hard requirements) — weight 1x
    "TATA_box":    [r"TATAAA", r"TATATA", r"TATATAA"],
    "CAAT_box":    [r"CCAAT", r"CAAT(?!G)"],

    # Primary activating elements — weight 5x (MAIN DRIVERS of promoter strength)
    "as1_element": [r"TGACG", r"TGACG[CT]", r"TGACGT[AC]", r"TGACG[ATC]"],
    "GCN4_motif":  [r"TGACTCA", r"TGACT[CT][TC]"],   # GCN4 — amino acid starvation response, strong activator
    "ocs_like":    [r"TGACGTAA", r"TGACGTA[AC]"],    # OCS-like — ocs element (octopine synthase) — similar to as-1

    # Enhancer elements — weight 2x
    "G_box":       [r"CACGTG"],
    "Box_II":      [r"AAACCAATCT"],
    "GT1_motif":   [r"GGTTAAT", r"GGTTAAC"],
    "I_box":       [r"GATAAGAT"],
    "W_box":       [r"TTGACC", r"TTGACT"],
    "ABRE":        [r"ACGTGG", r"ACGTGC"],
    "MBS":         [r"CAACTG"],
    "ARE":         [r"AAACCA"],
    # Summary metrics (computed, not regex-scanned)
    "weighted_score": [],   # sum(count × weight) — filled in score_sequence()
}

# Weights per element class — key driver of strength score quality
ELEMENT_WEIGHTS = {
    # Primary drivers (5x)
    "as1_element": 5,
    "GCN4_motif":  5,
    "ocs_like":    3,
    # Core (1x)
    "TATA_box":    1,
    "CAAT_box":    1,
    # Enhancers (2x)
    "G_box":       2,
    "Box_II":      2,
    "GT1_motif":   2,
    "I_box":       2,
    "W_box":       2,
    "ABRE":        2,
    "MBS":         2,
    "ARE":         2,
}


def score_sequence(sequence: str) -> dict:
    """
    Count cis-element occurrences and compute weighted strength score.

    Returns:
        dict with raw counts, weighted_score, and gc_pct.
        weighted_score = sum(count[name] * ELEMENT_WEIGHTS[name]) for all elements.
    """
    seq = sequence.upper()
    results = {}
    weighted = 0
    for name, patterns in CIS_ELEMENTS.items():
        if not patterns:   # weighted_score has no regex patterns
            continue
        count = sum(len(re.findall(p, seq)) for p in patterns)
        results[name] = count
        weight = ELEMENT_WEIGHTS.get(name, 1)
        weighted += count * weight
    results["total_elements"] = sum(results.values())
    results["weighted_score"] = weighted
    gc = (seq.count("G") + seq.count("C")) / len(seq) * 100
    results["gc_pct"] = round(gc, 1)
    return results


def print_table(df: pd.DataFrame, logger):
    """Print a colour-coded element comparison table."""
    HARD_REQUIRED = {"TATA_box", "CAAT_box"}
    PRIMARY_DRIVERS = {"as1_element", "GCN4_motif", "ocs_like"}

    log_info(logger, "")
    header = f"  {'Element':<18}" + "".join(f"{col:>16}" for col in df.columns)
    log_info(logger, header)
    log_info(logger, "  " + "-" * (16 + 16 * len(df.columns)))

    for element in df.index:
        row_str = f"  {element:<18}"
        for col in df.columns:
            val = df.loc[element, col]
            # Colour: red if hard-required and missing, grey if 0, green if present
            if element in HARD_REQUIRED and val == 0:
                row_str += f"\033[91m{str(val):>16}\033[0m"
            elif val == 0:
                row_str += f"\033[90m{str(val):>16}\033[0m"
            else:
                row_str += f"\033[92m{str(val):>16}\033[0m"
        log_info(logger, row_str)

    log_info(logger, "  " + "-" * (16 + 16 * len(df.columns)))
    log_info(logger, "  " + f"{'weighted_score':<18}" + "".join(f"{col:>16}" for col in df.columns))
    log_info(logger, "  " + "-" * (16 + 16 * len(df.columns)))
    for col in df.columns:
        val = df.loc["weighted_score", col]
        log_info(logger, f"  {'(strength)':<18}" + f"\033[92m{str(val):>16}\033[0m")


def main():
    logger = setup_logger(config.LOG_FILE)
    log_step(logger, 2, "Scoring Reference Promoters for Cis-Elements")

    if not os.path.exists(config.REF_FASTA):
        log_info(logger, "Reference FASTA not found — running Step 1 first...")
        from fetch_references import main as run_step1
        run_step1()

    references = load_fasta(config.REF_FASTA)
    log_ok(logger, f"Loaded {len(references)} reference promoters")

    # Score each
    scores = {}
    for name, seq in references.items():
        scores[name] = score_sequence(seq)
        total = scores[name]["total_elements"]
        log_ok(logger, f"Scored {name}: {total} total elements")

    df = pd.DataFrame(scores)

    # Print table
    log_info(logger, "")
    log_info(logger, "CIS-ELEMENT PROFILE — REFERENCE PROMOTERS")
    print_table(df, logger)

    # Save CSV
    os.makedirs("data", exist_ok=True)
    df.to_csv(config.REF_SCORES_CSV)
    log_ok(logger, f"Saved → {config.REF_SCORES_CSV}")

    # Print design target
    if "2x_CaMV_35S" in scores:
        ref = scores["2x_CaMV_35S"]
        log_info(logger, "")
        log_info(logger, "DESIGN TARGET (your synthetic candidates must meet ALL of these):")
        log_info(logger, f"  TATA box:        >= 1     [2×35S = {ref['TATA_box']}]  HARD FILTER")
        log_info(logger, f"  CAAT box:        >= 1     [2×35S = {ref['CAAT_box']}]  HARD FILTER")
        log_info(logger, f"  as-1 elements:   >= {ref['as1_element']}     [2×35S = {ref['as1_element']}]  PRIMARY (5x weight)")
        log_info(logger, f"  GCN4 motif:      >= {ref.get('GCN4_motif', 0)}     [2×35S = {ref.get('GCN4_motif', 0)}]  PRIMARY (5x weight)")
        log_info(logger, f"  ocs-like:        >= {ref.get('ocs_like', 0)}     [2×35S = {ref.get('ocs_like', 0)}]  secondary (3x weight)")
        log_info(logger, f"  G-box:           >= 0     [2×35S = {ref['G_box']}]  soft preference only")
        log_info(logger, f"  Total elements:  >= {ref['total_elements']}     [match or beat 2×35S raw count]")
        log_info(logger, f"  Weighted score:  >= {ref['weighted_score']}     [2×35S baseline — higher is better]")
        log_info(logger, f"  GC content:      40-60%   [2×35S = {ref['gc_pct']}%]")
        log_info(logger, "")
        log_info(logger, "ELEMENT WEIGHTS: as-1=5x, GCN4=5x, ocs-like=3x, others=1-2x")
        log_info(logger, "NOTE: G-box = 0 in 2×CaMV 35S is expected and correct.")
        log_info(logger, "CaMV 35S transcription is driven by as-1/ocs elements, not G-boxes.")
        log_info(logger, "Do not discard candidates for lacking a G-box.")

    log_info(logger, "")
    log_info(logger, "Next: python step3_generate_candidates.py")
    return df, scores


if __name__ == "__main__":
    main()
