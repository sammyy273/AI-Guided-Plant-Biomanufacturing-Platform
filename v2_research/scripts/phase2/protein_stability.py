#!/usr/bin/env python3
"""
STEP 3: Protein Stability — Mechanistic Degradation Analysis.

Strengthens degradation analysis using real sequence signals:
- Signal peptide detection → secretion status
- Transmembrane region detection
- PEST sequences (degradation motifs)
- Low complexity regions
- Protease cleavage site mapping
- Ubiquitination site quantification

Maps to protease exposure risk based on compartment routing.

OUTPUTS:
  outputs/phase2/protein_stability_enhanced.csv
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

OUTPUT_DIR = BASE_DIR / "outputs" / "phase2"
PROTEIN_DIR = BASE_DIR / "data" / "protein"


# ── Amino acid properties ──────────────────────────────────────────────────

HYDROPHOBIC = set("AILMFWV")
POSITIVELY_CHARGED = set("KR")
NEGATIVELY_CHARGED = set("DE")
PEST_AAS = set("PESTDNGQ")  # PEST-favoring amino acids
AROMATIC = set("FWY")
POLAR = set("STCNQ")


def load_target_protein():
    """Load the target protein sequence."""
    from Bio import SeqIO
    for fasta_file in PROTEIN_DIR.glob("*.fasta"):
        for record in SeqIO.parse(str(fasta_file), "fasta"):
            return str(record.seq), record.id
    return None, None


def detect_signal_peptide(seq):
    """Detect N-terminal signal peptide (SignalP-like heuristic)."""
    seq = seq.upper()
    n_term = seq[:50]

    # N-region: positive charges in first 5 residues
    n_pos = sum(1 for aa in n_term[:5] if aa in POSITIVELY_CHARGED)

    # H-region: hydrophobic core in residues 5-25
    h_region = n_term[5:25] if len(n_term) >= 25 else n_term[5:]
    h_hydro = sum(1 for aa in h_region if aa in HYDROPHOBIC)

    # Cleavage site (AXA motif in positions 15-35)
    cleavage = None
    for i in range(15, min(35, len(seq) - 2)):
        if seq[i] in "ASG" and seq[i+2] in "ASG":
            cleavage = i + 1
            break

    detected = n_pos >= 1 and h_hydro >= 8

    return {
        "detected": detected,
        "n_region_positive_charges": n_pos,
        "h_region_hydrophobic_count": h_hydro,
        "h_region_hydrophobic_fraction": round(h_hydro / max(len(h_region), 1), 3),
        "predicted_cleavage_site": cleavage,
        "signal_peptide_sequence": seq[:cleavage] if cleavage and detected else seq[:25],
    }


def detect_transmembrane_regions(seq, window=19, threshold=0.65):
    """Detect transmembrane helices."""
    seq = seq.upper()
    regions = []

    for i in range(0, len(seq) - window + 1, 3):
        segment = seq[i:i+window]
        hydro_frac = sum(1 for aa in segment if aa in HYDROPHOBIC) / window
        if hydro_frac >= threshold:
            regions.append({
                "start": i + 1,
                "end": i + window,
                "hydrophobic_fraction": round(hydro_frac, 3),
                "sequence": segment,
            })

    # Merge overlapping regions
    merged = []
    for r in regions:
        if merged and r["start"] <= merged[-1]["end"] + 3:
            merged[-1]["end"] = max(merged[-1]["end"], r["end"])
            merged[-1]["hydrophobic_fraction"] = max(
                merged[-1]["hydrophobic_fraction"], r["hydrophobic_fraction"]
            )
        else:
            merged.append(dict(r))

    return merged


def detect_pest_sequences(seq, window=12, pest_fraction_threshold=0.58):
    """
    Detect PEST sequences — regions enriched in Pro/Glu/Ser/Thr/Asp/Asn/Gln
    that target proteins for rapid degradation.
    """
    seq = seq.upper()
    pest_regions = []

    for i in range(0, len(seq) - window + 1):
        segment = seq[i:i+window]
        pest_count = sum(1 for aa in segment if aa in PEST_AAS)
        pest_frac = pest_count / window

        if pest_frac >= pest_fraction_threshold:
            pest_regions.append({
                "start": i + 1,
                "end": i + window,
                "pest_fraction": round(pest_frac, 3),
                "enrichment_score": round(pest_frac / 0.4375, 3),  # normalize by expected random
            })

    # Merge overlapping PEST regions
    merged = []
    for r in pest_regions:
        if merged and r["start"] <= merged[-1]["end"] + 1:
            merged[-1]["end"] = max(merged[-1]["end"], r["end"])
            merged[-1]["pest_fraction"] = max(merged[-1]["pest_fraction"], r["pest_fraction"])
        else:
            merged.append(dict(r))

    return merged


def detect_low_complexity_regions(seq, window=20, entropy_threshold=2.0):
    """Detect low complexity regions by Shannon entropy."""
    import math
    seq = seq.upper()
    regions = []

    for i in range(0, len(seq) - window + 1, 5):
        segment = seq[i:i+window]
        counts = {}
        for aa in segment:
            counts[aa] = counts.get(aa, 0) + 1

        entropy = 0
        for count in counts.values():
            p = count / window
            if p > 0:
                entropy -= p * math.log2(p)

        if entropy < entropy_threshold:
            dominant = max(counts, key=counts.get)
            regions.append({
                "start": i + 1,
                "end": i + window,
                "entropy": round(entropy, 3),
                "dominant_aa": dominant,
                "dominant_fraction": round(counts[dominant] / window, 3),
            })

    # Merge
    merged = []
    for r in regions:
        if merged and r["start"] <= merged[-1]["end"] + 5:
            merged[-1]["end"] = max(merged[-1]["end"], r["end"])
        else:
            merged.append(dict(r))

    return merged


def detect_protease_sites(seq):
    """Map protease cleavage motif occurrences."""
    seq = seq.upper()

    # Subtilase (SBT1) motifs
    sbt1_motifs = ["RR", "KR", "LK", "IR", "LR"]
    sbt1_hits = sum(seq.count(m) for m in sbt1_motifs)

    # C1A cysteine protease motifs
    c1a_motifs = ["VG", "FA", "WA", "GVA", "GFA"]
    c1a_hits = sum(seq.count(m) for m in c1a_motifs)

    # A1 aspartic protease motifs
    a1_motifs = ["DTG", "DSG", "DTA", "DSG", "DTS", "DVG"]
    a1_hits = sum(seq.count(m) for m in a1_motifs)

    # Additional specific motifs
    trypsin_like = sum(1 for i in range(len(seq)-1) if seq[i] in "KR" and seq[i+1] not in "P")

    return {
        "SBT1_subtilase_motifs": sbt1_hits,
        "C1A_cysteine_protease_motifs": c1a_hits,
        "A1_aspartic_protease_motifs": a1_hits,
        "trypsin_like_sites": trypsin_like,
        "total_protease_motifs": sbt1_hits + c1a_hits + a1_hits,
    }


def compute_ubiquitination_sites(seq):
    """Quantify lysine residues (ubiquitination acceptor sites)."""
    seq = seq.upper()
    lysine_count = seq.count("K")
    lysine_density = lysine_count / len(seq)

    return {
        "lysine_count": lysine_count,
        "lysine_density": round(lysine_density, 4),
        "ubiquitination_risk": round(min(1.0, lysine_density * 6.0), 4),
    }


def compute_intrinsic_instability(seq):
    """
    Compute intrinsic instability score based on sequence composition.
    Based on N-end rule and PEST-based instability.
    """
    seq = seq.upper()

    # N-end rule: N-terminal residue
    n_terminal = seq[0] if seq else "X"
    destabilizing_n_term = {"R", "K", "F", "L", "W", "Y"}  # Type 1+2 destabilizing
    n_term_score = 0.3 if n_terminal in destabilizing_n_term else 0.0

    # PEST content
    pest_aas = sum(1 for aa in seq if aa in PEST_AAS)
    pest_fraction = pest_aas / len(seq)

    # Lysine content (ubiquitination)
    lys_fraction = seq.count("K") / len(seq)

    # Proline content (affects stability)
    pro_fraction = seq.count("P") / len(seq)

    # Glycine content (flexibility, can destabilize)
    gly_fraction = seq.count("G") / len(seq)

    # Instability index components
    instability = (
        0.25 * min(1.0, pest_fraction / 0.50) +  # PEST-driven
        0.25 * min(1.0, lys_fraction * 5.0) +     # ubiquitination-driven
        0.20 * n_term_score +                       # N-end rule
        0.15 * min(1.0, pro_fraction * 3.0) +     # proline-driven
        0.15 * min(1.0, gly_fraction * 3.0)        # flexibility-driven
    )

    return round(instability, 4)


def compute_exposure_adjusted_degradation(localization, protease_sites, ubiquitination):
    """
    Adjust degradation risk based on subcellular compartment routing.
    Different compartments expose proteins to different protease families.
    """
    # Compartment-specific protease exposure multipliers
    compartment_risk = {
        "extracellular": {"A1": 1.5, "SBT1": 1.3, "C1A": 0.8, "ubiquitin": 0.3},
        "secreted": {"A1": 1.5, "SBT1": 1.3, "C1A": 0.8, "ubiquitin": 0.3},
        "apoplast": {"A1": 1.5, "SBT1": 1.3, "C1A": 0.8, "ubiquitin": 0.3},
        "ER": {"A1": 0.5, "SBT1": 0.3, "C1A": 0.2, "ubiquitin": 0.5},
        "membrane": {"A1": 0.8, "SBT1": 0.6, "C1A": 0.4, "ubiquitin": 0.6},
        "cell membrane": {"A1": 0.8, "SBT1": 0.6, "C1A": 0.4, "ubiquitin": 0.6},
        "cytoplasm": {"A1": 0.3, "SBT1": 0.2, "C1A": 0.3, "ubiquitin": 1.5},
        "nucleus": {"A1": 0.2, "SBT1": 0.1, "C1A": 0.2, "ubiquitin": 1.0},
        "vacuole": {"A1": 0.6, "SBT1": 0.4, "C1A": 1.5, "ubiquitin": 0.5},
        "chloroplast": {"A1": 0.3, "SBT1": 0.2, "C1A": 0.3, "ubiquitin": 0.8},
    }

    # Default if unknown
    loc = localization.lower()
    for key in compartment_risk:
        if key in loc:
            risk_mult = compartment_risk[key]
            break
    else:
        risk_mult = {"A1": 1.0, "SBT1": 1.0, "C1A": 1.0, "ubiquitin": 1.0}

    # Weighted protease exposure
    total_motifs = protease_sites["total_protease_motifs"]
    if total_motifs == 0:
        total_motifs = 1

    a1_exposure = (protease_sites["A1_aspartic_protease_motifs"] / total_motifs) * risk_mult["A1"]
    sbt1_exposure = (protease_sites["SBT1_subtilase_motifs"] / total_motifs) * risk_mult["SBT1"]
    c1a_exposure = (protease_sites["C1A_cysteine_protease_motifs"] / total_motifs) * risk_mult["C1A"]
    ubiquitin_exposure = ubiquitination["ubiquitination_risk"] * risk_mult["ubiquitin"]

    degradation_score = min(1.0, (
        0.35 * a1_exposure +
        0.25 * sbt1_exposure +
        0.20 * c1a_exposure +
        0.20 * ubiquitin_exposure
    ))

    # Normalize by total protease motif count
    motif_density = min(1.0, total_motifs / 100)
    degradation_score = min(1.0, degradation_score + motif_density * 0.2)

    return {
        "degradation_score": round(degradation_score, 4),
        "risk_class": "HIGH" if degradation_score >= 0.66 else ("MEDIUM" if degradation_score >= 0.33 else "LOW"),
        "a1_aspartic_exposure": round(a1_exposure, 4),
        "sbt1_subtilase_exposure": round(sbt1_exposure, 4),
        "c1a_cysteine_exposure": round(c1a_exposure, 4),
        "ubiquitin_exposure": round(ubiquitin_exposure, 4),
        "compartment_risk_multipliers": risk_mult,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 3: Protein Stability — Mechanistic Degradation Analysis")
    print("=" * 60)

    sequence, seq_id = load_target_protein()
    if sequence is None:
        print("  ERROR: No target protein found")
        return

    print(f"  Target protein: {seq_id}")
    print(f"  Sequence length: {len(sequence)} aa")
    print()

    # 1. Signal peptide
    print("  Detecting signal peptide...")
    sp = detect_signal_peptide(sequence)
    print(f"    Detected: {sp['detected']}, cleavage@{sp['predicted_cleavage_site']}")

    # 2. Transmembrane regions
    print("  Detecting transmembrane regions...")
    tm = detect_transmembrane_regions(sequence)
    print(f"    Found: {len(tm)} TM regions")
    for t in tm:
        print(f"      {t['start']}-{t['end']}: hydro={t['hydrophobic_fraction']:.2f}")

    # 3. PEST sequences
    print("  Detecting PEST degradation motifs...")
    pest = detect_pest_sequences(sequence)
    print(f"    Found: {len(pest)} PEST regions")
    for p in pest[:5]:
        print(f"      {p['start']}-{p['end']}: PEST_frac={p['pest_fraction']:.2f}")
    if len(pest) > 5:
        print(f"      ... and {len(pest) - 5} more")

    # 4. Low complexity regions
    print("  Detecting low complexity regions...")
    lcr = detect_low_complexity_regions(sequence)
    print(f"    Found: {len(lcr)} low complexity regions")

    # 5. Protease cleavage sites
    print("  Mapping protease cleavage sites...")
    protease = detect_protease_sites(sequence)
    print(f"    SBT1 (subtilase) motifs: {protease['SBT1_subtilase_motifs']}")
    print(f"    C1A (cysteine protease) motifs: {protease['C1A_cysteine_protease_motifs']}")
    print(f"    A1 (aspartic protease) motifs: {protease['A1_aspartic_protease_motifs']}")
    print(f"    Trypsin-like sites: {protease['trypsin_like_sites']}")

    # 6. Ubiquitination sites
    print("  Quantifying ubiquitination sites...")
    ubiq = compute_ubiquitination_sites(sequence)
    print(f"    Lysine count: {ubiq['lysine_count']}")
    print(f"    Lysine density: {ubiq['lysine_density']:.4f}")
    print(f"    Ubiquitination risk: {ubiq['ubiquitination_risk']:.4f}")

    # 7. Intrinsic instability
    print("  Computing intrinsic instability score...")
    instability = compute_intrinsic_instability(sequence)
    print(f"    Intrinsic instability: {instability:.4f}")

    # 8. Exposure-adjusted degradation for different compartments
    print("\n  Computing exposure-adjusted degradation by compartment...")
    compartments_to_test = ["extracellular", "ER", "cytoplasm", "vacuole", "membrane"]
    degradation_by_compartment = {}

    for compartment in compartments_to_test:
        result = compute_exposure_adjusted_degradation(compartment, protease, ubiq)
        degradation_by_compartment[compartment] = result
        print(f"    {compartment:15s}: score={result['degradation_score']:.4f} ({result['risk_class']})")

    # 9. Determine likely routing and degradation
    # If signal peptide detected → secretory pathway → extracellular default
    if sp["detected"]:
        routing = "secretory_pathway"
        primary_compartment = "extracellular"
    else:
        routing = "cytosolic"
        primary_compartment = "cytoplasm"

    primary_degradation = degradation_by_compartment[primary_compartment]
    print(f"\n  Primary routing: {routing} → {primary_compartment}")
    print(f"  Primary degradation risk: {primary_degradation['degradation_score']:.4f} ({primary_degradation['risk_class']})")

    # A1 dominance check
    a1_exp = primary_degradation["a1_aspartic_exposure"]
    sbt1_exp = primary_degradation["sbt1_subtilase_exposure"]
    c1a_exp = primary_degradation["c1a_cysteine_exposure"]

    dominant_protease = max(
        [("A1_aspartic", a1_exp), ("SBT1_subtilase", sbt1_exp), ("C1A_cysteine", c1a_exp)],
        key=lambda x: x[1]
    )
    print(f"  Dominant protease family: {dominant_protease[0]} (exposure={dominant_protease[1]:.4f})")

    # 10. Build output CSV
    rows = []

    # Summary row
    rows.append({
        "protein_id": seq_id,
        "sequence_length": len(sequence),
        "signal_peptide": sp["detected"],
        "signal_peptide_cleavage": sp["predicted_cleavage_site"],
        "tm_region_count": len(tm),
        "pest_region_count": len(pest),
        "low_complexity_region_count": len(lcr),
        "lysine_count": ubiq["lysine_count"],
        "lysine_density": ubiq["lysine_density"],
        "ubiquitination_risk": ubiq["ubiquitination_risk"],
        "intrinsic_instability": instability,
        "routing": routing,
        "primary_compartment": primary_compartment,
        "primary_degradation_score": primary_degradation["degradation_score"],
        "primary_degradation_class": primary_degradation["risk_class"],
        "dominant_protease": dominant_protease[0],
        "dominant_protease_exposure": dominant_protease[1],
        "A1_motifs": protease["A1_aspartic_protease_motifs"],
        "SBT1_motifs": protease["SBT1_subtilase_motifs"],
        "C1A_motifs": protease["C1A_cysteine_protease_motifs"],
        "trypsin_like_sites": protease["trypsin_like_sites"],
    })

    # Per-compartment rows
    for compartment, result in degradation_by_compartment.items():
        rows.append({
            "protein_id": seq_id,
            "sequence_length": len(sequence),
            "signal_peptide": sp["detected"],
            "signal_peptide_cleavage": sp["predicted_cleavage_site"],
            "tm_region_count": len(tm),
            "pest_region_count": len(pest),
            "low_complexity_region_count": len(lcr),
            "lysine_count": ubiq["lysine_count"],
            "lysine_density": ubiq["lysine_density"],
            "ubiquitination_risk": ubiq["ubiquitination_risk"],
            "intrinsic_instability": instability,
            "routing": routing,
            "primary_compartment": compartment,
            "primary_degradation_score": result["degradation_score"],
            "primary_degradation_class": result["risk_class"],
            "dominant_protease": max(
                [("A1", result["a1_aspartic_exposure"]),
                 ("SBT1", result["sbt1_subtilase_exposure"]),
                 ("C1A", result["c1a_cysteine_exposure"])],
                key=lambda x: x[1]
            )[0],
            "dominant_protease_exposure": max(
                [result["a1_aspartic_exposure"],
                 result["sbt1_subtilase_exposure"],
                 result["c1a_cysteine_exposure"]]
            ),
            "A1_motifs": protease["A1_aspartic_protease_motifs"],
            "SBT1_motifs": protease["SBT1_subtilase_motifs"],
            "C1A_motifs": protease["C1A_cysteine_protease_motifs"],
            "trypsin_like_sites": protease["trypsin_like_sites"],
        })

    out_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "protein_stability_enhanced.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(out_df)} rows)")

    # Save detailed JSON
    detail = {
        "protein_id": seq_id,
        "signal_peptide": sp,
        "transmembrane_regions": tm,
        "pest_regions": {"count": len(pest), "regions": pest[:20]},
        "low_complexity_regions": {"count": len(lcr), "regions": lcr[:10]},
        "protease_sites": protease,
        "ubiquitination": ubiq,
        "intrinsic_instability": instability,
        "degradation_by_compartment": degradation_by_compartment,
        "routing": routing,
        "primary_compartment": primary_compartment,
        "dominant_protease": dominant_protease[0],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    detail_path = OUTPUT_DIR / "protein_stability_detail.json"
    with open(detail_path, "w") as fh:
        json.dump(detail, fh, indent=2, default=str)
    print(f"  Saved: {detail_path}")


if __name__ == "__main__":
    main()
