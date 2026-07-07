# =============================================================================
# 1_fetch_references.py
#
# WHAT THIS DOES:
#   Builds the reference promoter dataset used as your design target.
#
#   2×CaMV 35S  → hardcoded from Jameel et al., Plants 2020 (10:1700)
#                 Yellow (330 nt) + Brown (327 nt) + Red/core (89 nt) = 746 bp
#                 Validated: TATA box confirmed, GC 46.1%, as-1 elements present.
#                 NOTE: G-box count = 0 in this sequence — this is correct.
#                 CaMV 35S is driven by as-1 elements, not G-boxes.
#
#   NbACT3      → upstream region fetched from NCBI (AY007693)
#   NbUBI       → upstream region fetched from NCBI (AJ012639)
#   NbRbcS      → upstream region fetched from NCBI (X02591)
#
# WHY THESE FOUR?
#   They represent the spectrum of strong constitutive plant promoters:
#   - 2×35S: viral, constitutive, gold standard benchmark
#   - NbACT3: endogenous N. benthamiana, no silencing risk
#   - NbUBI: endogenous, ubiquitous expression in all tissues
#   - NbRbcS: endogenous, highest expression in leaf (our target tissue)
#   Together they define what a "strong plant promoter" looks like in
#   cis-element space — the target your Evo2 candidates must match.
#
# OUTPUT:
#   data/reference_promoters.fasta
# =============================================================================

import os
import sys
import time
from Bio import Entrez, SeqIO
from Bio.SeqUtils import gc_fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import setup_logger, log_step, log_ok, log_warn, log_err, log_info, save_fasta

Entrez.email = config.NCBI_EMAIL


# =============================================================================
# 2×CaMV 35S — FROM PAPER (Jameel et al., Plants 2020, 10:1700, Supplementary S1)
# =============================================================================
def get_camv_35s_from_paper() -> str:
    """
    2×CaMV 35S promoter sequence from the literature.

    Structure (5' to 3'):
      Yellow = duplicated enhancer (330 nt) — the upstream "volume boost"
      Brown  = original enhancer  (327 nt) — single copy of the enhancer
      Red    = core promoter       (89 nt) — contains TATA box, TSS

    Validated properties (confirmed by local scan):
      Total length:   746 bp
      GC content:     46.1%  (within 40–60% synthesis window)
      TATA box:       1 (TATATA in core/red region) — CONFIRMED
      CAAT box:       1 — CONFIRMED
      as-1 elements:  2 (TGACG) — these are the primary activating elements
      G-box:          0 — EXPECTED (CaMV 35S uses as-1, not G-box activation)

    Yellow vs Brown: 3 bp length difference (330 vs 327) due to a triplet
    variant in the enhancer. When aligned properly, identity is >98%.
    The 165 positional differences seen in character-by-character comparison
    are an alignment artefact from the 3-nt length offset — not real divergence.
    """
    yellow = (
        "ACATGGTGGAGCACGACACTCTCGTCTACTCCAAGAATATCAAAGATACAGTCTCAGAAGACCAGAGGGCTATTGAG"
        "ACTTTTCAACAAAGGGTAATATCGGGAAACCTCCTCGGATTCCATTGCCCAGCTATCTGTCACTTCATCGAAAGGAC"
        "AGTAGAAAAGGAAGATGGCTTCTACAAATGCCATCATTGCGATAAAGGAAAGGCTATCGTTCAAAGAATGCCTCTAC"
        "CGACAGTGGTCCCAAAGATGGACCCCCCACCCACGAGGAACATCGTGGAAAAAGAAGACGTTCCAACCACGTCTTCA"
        "AAGCAAGTGGATTGATGTGATA"
    )
    brown = (
        "ACATGGTGGAGCACGACACTCTCGTCTACTCCAAGAATATCAAAGATACAGTCTC"
        "AGAAGACCAGAGGGCTATTGAGACTTTCAACAAAGGGTAATATCGGGAAACCTCCTCGGATTCCATTGCCCAGCTAT"
        "CTGTCACTTCATCGAAAGGACAGTAGAAAAGGAAGATGGCTTCTACAAATGCCATCATTGCGATAAAGGAAAGGCTA"
        "TCGTTCAAGAATGCCTCTACCGACAGTGGTCCCAAAGATGGACCCCCACCCACGAGGAACATCGTGGAAAAAGAAGA"
        "CGTTCCAACCACGTCTTCAAAGCAAGTGGATTGATGTGATA"
    )
    red = (
        "TCTCCACTGACGTAAGGGATGACGCACAATCCCACT"
        "ATCCTTCGCAAGACCCTTCCTCTATATAAGGAAGTTCATTTCATTTGGAGAGG"
    )
    return yellow + brown + red


# =============================================================================
# NCBI FETCH UTILITIES
# =============================================================================
def fetch_ncbi_record(accession: str, retries: int = 3):
    """Fetch a GenBank record from NCBI with retry on failure."""
    for attempt in range(retries):
        try:
            handle = Entrez.efetch(
                db="nucleotide", id=accession,
                rettype="gb", retmode="text"
            )
            record = SeqIO.read(handle, "genbank")
            handle.close()
            time.sleep(0.5)   # Respect NCBI rate limit (max 3 requests/sec)
            return record
        except Exception:
            if attempt < retries - 1:
                time.sleep(3)
    return None


def extract_upstream(record, upstream_bp: int = 2000) -> str | None:
    """
    Extract the region upstream of the first CDS feature in a GenBank record.

    WHY 2000 bp?
    Most plant promoters are active within 1–2 kb upstream of the TSS.
    Taking 2 kb captures the full enhancer + core promoter region for all
    common plant promoters. Some have distal regulatory elements further
    upstream, but 2 kb covers >90% of the functional region.
    """
    for feature in record.features:
        if feature.type == "CDS":
            start = int(feature.location.start)
            if start >= upstream_bp:
                return str(record.seq[start - upstream_bp : start])
            elif start > 50:
                return str(record.seq[:start])   # Gene close to record start
    return None


# =============================================================================
# ENDOGENOUS N. benthamiana PROMOTERS
# =============================================================================
def get_nb_promoters(logger: object) -> dict:
    """
    Fetch upstream regions of three key N. benthamiana / Solanaceae genes.
    These serve as endogenous promoter benchmarks alongside the viral 2×35S.
    """
    targets = {
        "NbACT3": ("AY007693", "N. benthamiana actin 3 — constitutive housekeeping"),
        "NbUBI":  ("AJ012639", "Nicotiana ubiquitin — constitutive, all tissues"),
        "NbRbcS": ("X02591",   "Nicotiana RbcS — highest expression in leaf tissue"),
    }

    promoters = {}
    for name, (accession, description) in targets.items():
        log_info(logger, f"Fetching {name} ({accession}) — {description}")
        record = fetch_ncbi_record(accession)

        if not record:
            log_warn(logger, f"{name}: NCBI fetch failed after 3 attempts. Skipping.")
            log_warn(logger, f"  You can retry by re-running this step later.")
            continue

        upstream = extract_upstream(record)
        if upstream:
            gc = gc_fraction(upstream) * 100
            log_ok(logger, f"{name}: {len(upstream)} bp | GC {gc:.1f}%")
            promoters[name] = upstream
        else:
            log_warn(logger, f"{name}: No CDS feature found in record. Skipping.")

    return promoters


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    logger = setup_logger(config.LOG_FILE)
    log_step(logger, 1, "Building Reference Promoter Dataset")

    # ── 2×CaMV 35S (hardcoded from paper) ────────────────────────────────────
    camv = get_camv_35s_from_paper()
    gc   = (camv.count("G") + camv.count("C")) / len(camv) * 100
    log_ok(logger, f"2×CaMV 35S: {len(camv)} bp | GC {gc:.1f}%")
    log_info(logger, "  Source: Jameel et al., Plants 2020, 10(8):1700, Suppl. S1")
    log_info(logger, "  Structure: Yellow(330) + Brown(327) + Red/core(89) = 746 bp")

    # ── Endogenous N. benthamiana promoters (NCBI) ────────────────────────────
    log_info(logger, "")
    nb = get_nb_promoters(logger)

    if not nb:
        log_warn(logger, "No endogenous promoters fetched. Continuing with 2×35S only.")
        log_warn(logger, "NCBI may be temporarily unavailable. Re-run step1 later to add them.")

    # ── Combine all references ────────────────────────────────────────────────
    references = {"2x_CaMV_35S": camv, **nb}

    # ── Save ──────────────────────────────────────────────────────────────────
    save_fasta(references, config.REF_FASTA)
    log_ok(logger, f"Saved {len(references)} reference promoters → {config.REF_FASTA}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log_info(logger, "")
    log_info(logger, "Reference promoters collected:")
    for name, seq in references.items():
        gc = (seq.count("G") + seq.count("C")) / len(seq) * 100
        log_info(logger, f"  {name:<18} {len(seq):>6} bp   GC: {gc:.1f}%")

    log_info(logger, "")
    log_info(logger, "Next: python step2_score_references.py")
    return references


if __name__ == "__main__":
    main()
