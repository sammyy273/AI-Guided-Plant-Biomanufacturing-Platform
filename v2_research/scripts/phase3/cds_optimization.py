#!/usr/bin/env python3
"""
STEP 4: Advanced CDS Optimization (Translation-Aware).

Upgrades CDS from CAI-only to translation-aware with:
- Codon pair bias scoring
- 5' mRNA folding energy (first 50 nt)
- Ribosome binding site accessibility
- Secondary structure penalty near start codon

OUTPUTS:
  outputs/phase3/cds_optimization_advanced.csv
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "modules"))

OUTPUT_DIR = BASE_DIR / "outputs" / "phase3"

# Codon usage tables (simplified — relative adaptiveness)
CODON_TABLE = {
    "F": ["UUU", "UUC"], "L": ["UUA", "UUG", "CUU", "CUC", "CUA", "CUG"],
    "I": ["AUU", "AUC", "AUA"], "M": ["AUG"], "V": ["GUU", "GUC", "GUA", "GUG"],
    "S": ["UCU", "UCC", "UCA", "UCG", "AGU", "AGC"], "P": ["CCU", "CCC", "CCA", "CCG"],
    "T": ["ACU", "ACC", "ACA", "ACG"], "A": ["GCU", "GCC", "GCA", "GCG"],
    "Y": ["UAU", "UAC"], "H": ["CAU", "CAC"], "Q": ["CAA", "CAG"],
    "N": ["AAU", "AAC"], "K": ["AAA", "AAG"], "D": ["GAU", "GAC"], "E": ["GAA", "GAG"],
    "C": ["UGU", "UGC"], "W": ["UGG"], "R": ["CGU", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "G": ["GGU", "GGC", "GGA", "GGG"], "*": ["UAA", "UAG", "UGA"],
}

# Species-specific preferred codons (simplified)
SPECIES_PREFERRED = {
    "tomato": {"GCU": 0.35, "GCC": 0.25, "GCA": 0.25, "GCG": 0.15},
    "rice": {"GCU": 0.20, "GCC": 0.40, "GCA": 0.15, "GCG": 0.25},
    "nbenthamiana": {"GCU": 0.35, "GCC": 0.25, "GCA": 0.25, "GCG": 0.15},
}


def protein_to_mrna(protein_seq, species="tomato"):
    """Convert protein to mRNA using species-preferred codons."""
    # Simplified: use first codon for each amino acid
    # In reality, would use full codon usage tables
    mrna = ""
    for aa in protein_seq.upper():
        if aa in CODON_TABLE:
            mrna += CODON_TABLE[aa][0]  # simplified
        elif aa == "*":
            mrna += "UAA"
    return mrna


def compute_mrna_folding_energy(sequence, window=50):
    """
    Estimate 5' mRNA folding energy using simple base-pairing model.
    Real tools use ViennaRNA; this is a simplified proxy.
    """
    # Convert T to U if needed
    seq = sequence.upper().replace("T", "U")
    n = min(window, len(seq))

    if n < 10:
        return 0

    # Simple nearest-neighbor approximation
    # GC pairs: -3.0 kcal/mol, AU: -2.0, GU: -1.0
    energy = 0
    segment = seq[:n]

    # Count potential base pairs in first window
    pairs = 0
    for i in range(n // 2):
        j = n - 1 - i
        if j <= i:
            break
        b1, b2 = segment[i], segment[j]
        if (b1 == "G" and b2 == "C") or (b1 == "C" and b2 == "G"):
            energy -= 3.0
            pairs += 1
        elif (b1 == "A" and b2 == "U") or (b1 == "U" and b2 == "A"):
            energy -= 2.0
            pairs += 1
        elif (b1 == "G" and b2 == "U") or (b1 == "U" and b2 == "G"):
            energy -= 1.0
            pairs += 1

    return round(energy, 2)


def compute_codon_pair_bias(cds_seq, window=2):
    """Compute codon pair bias score."""
    codons = [cds_seq[i:i+3] for i in range(0, len(cds_seq) - 2, 3)]
    if len(codons) < window + 1:
        return 0.0

    # Simple metric: count preferred adjacent codon pairs
    # Higher = more favorable codon pairs
    preferred_pairs = 0
    total_pairs = 0

    for i in range(len(codons) - 1):
        pair = codons[i] + codons[i+1]
        total_pairs += 1

        # Simple heuristic: pairs with balanced GC are preferred
        gc = sum(1 for c in pair if c in "GCgc") / len(pair)
        if 0.35 <= gc <= 0.55:
            preferred_pairs += 1

    return round(preferred_pairs / max(total_pairs, 1), 4)


def compute_ribosome_accessibility(cds_seq, start_region=30):
    """Estimate ribosome binding site accessibility (low structure near start)."""
    # Count runs of same base (reduces accessibility)
    seq = cds_seq.upper()[:start_region]
    if len(seq) < 10:
        return 1.0

    # Count homopolymer runs
    max_run = 1
    current_run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    # Penalty for long runs
    accessibility = max(0.0, 1.0 - max_run * 0.05)

    # GC content near start: moderate is best
    gc = sum(1 for c in seq if c in "GC") / len(seq)
    gc_penalty = abs(gc - 0.40) * 2

    return round(max(0.0, accessibility - gc_penalty), 4)


def check_forbidden_motifs(cds_seq):
    """Check for forbidden motifs that should be avoided."""
    seq = cds_seq.upper().replace("T", "U")
    forbidden = {
        "premature_polyA": ["AAAAAA", "UUUUUU"],
        "internal_ATG": [],  # Would need frame tracking
        "cryptic_splice_donor": ["GUAGU", "AUGAGU"],
        "cryptic_splice_acceptor": ["CAGGU", "UCCAG"],
        "restriction_sites": {
            "EcoRI": "GAATTC", "XbaI": "TCTAGA", "BamHI": "GGATCC",
            "SalI": "GTCGAC", "PstI": "CTGCAG", "HindIII": "AAGCTT",
        },
    }

    issues = []

    # Check poly-A/T
    for motif in forbidden["premature_polyA"]:
        if motif.replace("U", "T") in cds_seq.upper():
            issues.append(f"premature_polyA: {motif}")

    # Check cryptic splice sites
    dna_seq = cds_seq.upper()
    for motif in forbidden["cryptic_splice_donor"]:
        if motif.replace("U", "T") in dna_seq:
            issues.append(f"cryptic_splice_donor: {motif}")
    for motif in forbidden["cryptic_splice_acceptor"]:
        if motif.replace("U", "T") in dna_seq:
            issues.append(f"cryptic_splice_acceptor: {motif}")

    # Check restriction sites
    for enzyme, site in forbidden["restriction_sites"].items():
        if site in dna_seq:
            idx = dna_seq.index(site)
            issues.append(f"restriction_site: {enzyme} ({site}) at position {idx}")

    return issues


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 4: Advanced CDS Optimization")
    print("=" * 60)

    # Load existing CDS data from final reports
    species_data = {}
    for sp in ["tomato", "rice", "nbenthamiana"]:
        report_path = BASE_DIR / "outputs" / f"final_report_{sp}.json"
        if report_path.exists():
            with open(report_path) as fh:
                report = json.load(fh)
            cds = report.get("optimized_cds", {})
            if cds.get("sequence"):
                species_data[sp] = cds

    if not species_data:
        print("  ERROR: No CDS data found in final reports")
        return

    rows = []
    for sp, cds in species_data.items():
        cds_seq = cds["sequence"].upper()
        print(f"\n  {sp}: CDS length={cds['length_bp']}bp, CAI={cds.get('cai', '?')}, GC={cds.get('gc_pct', '?')}%")

        # 1. Codon pair bias
        cpb = compute_codon_pair_bias(cds_seq)
        print(f"    Codon pair bias: {cpb:.4f}")

        # 2. 5' mRNA folding energy
        folding_energy = compute_mrna_folding_energy(cds_seq)
        print(f"    5' folding energy: {folding_energy:.2f} kcal/mol")
        folding_penalty = 0 if folding_energy > -10 else min(0.3, abs(folding_energy + 10) * 0.03)
        print(f"    Folding penalty: {folding_penalty:.4f}")

        # 3. Ribosome accessibility
        ribo_access = compute_ribosome_accessibility(cds_seq)
        print(f"    Ribosome accessibility: {ribo_access:.4f}")

        # 4. Forbidden motifs
        forbidden = check_forbidden_motifs(cds_seq)
        print(f"    Forbidden motifs: {len(forbidden)}")
        for issue in forbidden:
            print(f"      - {issue}")

        # 5. GC extremes
        gc = cds.get("gc_pct", 0)
        gc_penalty = 0
        if gc > 65 or gc < 25:
            gc_penalty = 0.2
            print(f"    WARNING: Extreme GC ({gc}%)")

        # 6. Repeat analysis
        # Check for tandem repeats > 20bp
        tandem_repeats = 0
        for length in range(20, min(50, len(cds_seq) // 2)):
            for start in range(len(cds_seq) - length):
                motif = cds_seq[start:start + length]
                if cds_seq.count(motif) > 1:
                    tandem_repeats += 1
                    break
            if tandem_repeats > 0:
                break

        # 7. Overall CDS quality score
        base_cai = cds.get("cai", 0.8)
        quality_score = (
            0.35 * base_cai +
            0.20 * cpb +
            0.20 * ribo_access +
            0.10 * (1 - folding_penalty) +
            0.10 * (1 - gc_penalty) +
            0.05 * (1 - min(1, len(forbidden) / 5))
        )
        quality_score = round(max(0, min(1, quality_score)), 4)

        print(f"    CDS quality score: {quality_score:.4f}")

        rows.append({
            "species": sp,
            "cds_length_bp": cds["length_bp"],
            "gc_pct": gc,
            "cai": cds.get("cai", 0),
            "codon_pair_bias": cpb,
            "mrna_5prime_folding_energy": folding_energy,
            "folding_penalty": round(folding_penalty, 4),
            "ribosome_accessibility": ribo_access,
            "forbidden_motifs": len(forbidden),
            "forbidden_motif_details": "; ".join(forbidden) if forbidden else "none",
            "tandem_repeats": tandem_repeats,
            "gc_extreme_penalty": gc_penalty,
            "cds_quality_score": quality_score,
            "optimization_notes": cds.get("warnings", ""),
        })

    out_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "cds_optimization_advanced.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
