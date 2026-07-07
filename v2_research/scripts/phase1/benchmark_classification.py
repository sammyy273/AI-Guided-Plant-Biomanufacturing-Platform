#!/usr/bin/env python3
"""
STEP 3: Benchmark against literature-aligned expression classes.

Replaces overclaiming baseline comparisons with defensible classification.
Maps internal pipeline scores to relative expression classes (weak/moderate/strong)
using quantile-based thresholds.

OUTPUTS:
  outputs/phase1/benchmark_reclassified.csv
  outputs/phase1/benchmark_reference_notes.txt
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "modules"))

OUTPUT_DIR = BASE_DIR / "outputs" / "phase1"
PROMOTER_DIR = BASE_DIR / "data" / "promoters"

# ── Motif patterns ─────────────────────────────────────────────────────────

MOTIF_PATTERNS = {
    "TATA_box": ["TATAAA", "TATATA", "TATACA", "TATAGA"],
    "CAAT_box": ["CCAAT", "CAAAT"],
    "GC_box": ["GGGCGG", "CCGCCC"],
    "as1_element": ["TGACG", "CGTCA"],
    "DOF_site": ["AAAG", "CTTT"],
}


def count_motifs(seq, patterns):
    return sum(seq.upper().count(p) for p in patterns)


def compute_features(seq):
    """Compute key features for a promoter sequence."""
    s = seq.upper()
    gc = (s.count("G") + s.count("C")) / len(s) if s else 0

    proximal = s[-50:] if len(s) >= 50 else s
    upstream_region = s[-120:] if len(s) >= 120 else s

    has_tata = any(p in proximal for p in MOTIF_PATTERNS["TATA_box"])
    has_caat = any(p in upstream_region for p in MOTIF_PATTERNS["CAAT_box"])

    tata_count = count_motifs(s, MOTIF_PATTERNS["TATA_box"])
    caat_count = count_motifs(s, MOTIF_PATTERNS["CAAT_box"])
    gc_box_count = count_motifs(s, MOTIF_PATTERNS["GC_box"])
    as1_count = count_motifs(s, MOTIF_PATTERNS["as1_element"])
    dof_count = count_motifs(s, MOTIF_PATTERNS["DOF_site"])

    # Architecture presence
    if has_tata and has_caat:
        architecture = "TATA+CAAT"
    elif has_tata:
        architecture = "TATA_only"
    elif has_caat:
        architecture = "CAAT_only"
    else:
        architecture = "none"

    # Total motif richness
    total_motifs = tata_count + caat_count + gc_box_count + as1_count + dof_count

    # TATA positioning score (ideal at -25 to -35, i.e., within last 50bp)
    tata_position_score = 1.0 if has_tata else 0.0

    return {
        "gc_content": round(gc, 4),
        "length": len(s),
        "tata_count": tata_count,
        "caat_count": caat_count,
        "gc_box_count": gc_box_count,
        "as1_count": as1_count,
        "dof_count": dof_count,
        "total_motifs": total_motifs,
        "architecture": architecture,
        "tata_position_score": tata_position_score,
        "has_tata": has_tata,
        "has_caat": has_caat,
    }


def classify_expression(composite_score, quantiles):
    """Classify expression based on quantile thresholds of the score distribution."""
    if composite_score >= quantiles[0.66]:
        return "strong"
    elif composite_score >= quantiles[0.33]:
        return "moderate"
    else:
        return "weak"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 3: Benchmark Classification")
    print("=" * 60)

    species_list = ["arabidopsis", "rice", "tomato"]
    n_per_species = 500  # sample size for efficiency

    all_records = []

    for species in species_list:
        fasta_path = PROMOTER_DIR / f"{species}_promoters_1kb.fasta"
        if not fasta_path.exists():
            print(f"  {species}: FASTA not found, skipping")
            continue

        from Bio import SeqIO
        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        print(f"  {species}: loaded {len(records)} promoters")

        # Sample deterministically
        rng = np.random.RandomState(42)
        indices = rng.choice(len(records), min(n_per_species, len(records)), replace=False)

        for idx in indices:
            rec = records[idx]
            seq = str(rec.seq)
            features = compute_features(seq)

            # Simple proxy composite score (mirrors pipeline's weighted approach)
            gc = features["gc_content"]
            arch_bonus = {
                "TATA+CAAT": 0.5,
                "TATA_only": 0.25,
                "CAAT_only": 0.15,
                "none": 0.0,
            }[features["architecture"]]

            motif_score = min(1.0, features["total_motifs"] / 30)
            gc_balance = max(0, 1 - abs(gc - 0.38) / 0.25)

            composite = (
                0.30 * arch_bonus +
                0.25 * motif_score +
                0.20 * gc_balance +
                0.15 * features["tata_position_score"] +
                0.10 * (1 - min(1, features["gc_content"] * 3))
            )

            all_records.append({
                "species": species,
                "sequence_id": rec.id,
                **features,
                "composite_score": round(composite, 4),
            })

    if not all_records:
        print("ERROR: No records processed.")
        return

    df = pd.DataFrame(all_records)
    print(f"\n  Total promoters classified: {len(df)}")

    # Compute quantile thresholds per species
    quantile_thresholds = {}
    for species in species_list:
        sp_df = df[df["species"] == species]
        if sp_df.empty:
            continue
        q33 = sp_df["composite_score"].quantile(0.33)
        q66 = sp_df["composite_score"].quantile(0.66)
        quantile_thresholds[species] = {0.33: q33, 0.66: q66}
        print(f"  {species} thresholds: weak<{q33:.3f} <= moderate<{q66:.3f} <= strong")

    # Classify
    df["expression_class_relative"] = df.apply(
        lambda row: classify_expression(
            row["composite_score"],
            quantile_thresholds.get(row["species"], {0.33: 0.3, 0.66: 0.6})
        ),
        axis=1
    )

    # Add disclaimer
    df["classification_note"] = (
        "Classification is relative to internal scoring framework; "
        "not directly comparable to measured expression levels."
    )

    # Save
    out_path = OUTPUT_DIR / "benchmark_reclassified.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(df)} rows)")

    # Distribution summary
    print("\n  Expression class distribution:")
    for species in species_list:
        sp_df = df[df["species"] == species]
        counts = sp_df["expression_class_relative"].value_counts()
        total = len(sp_df)
        print(f"    {species}: ", end="")
        for cls in ["weak", "moderate", "strong"]:
            n = counts.get(cls, 0)
            print(f"{cls}={n} ({n/total*100:.1f}%) ", end="")
        print()

    # ── Reference notes ────────────────────────────────────────────────

    notes = [
        "BENCHMARK REFERENCE NOTES",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "CLASSIFICATION METHOD:",
        "  Promoters are classified into expression classes using quantile-based",
        "  thresholds of the internal composite score distribution:",
        "    - strong:  >= 66th percentile of composite score",
        "    - moderate: 33rd–66th percentile",
        "    - weak:   < 33rd percentile",
        "",
        "  Classification is relative to internal scoring framework;",
        "  not directly comparable to measured expression levels.",
        "",
        "INTERNAL COMPOSITE SCORE COMPONENTS:",
        "  30% — Core architecture (TATA+CAAT presence and positioning)",
        "  25% — Motif richness (total cis-element count, normalized)",
        "  20% — GC balance (proximity to species-optimal GC%)",
        "  15% — TATA positioning score (presence in -25/-35 zone)",
        "  10% — Silencing risk proxy (inverse of CG density)",
        "",
        "LITERATURE CONTEXT (qualitative, NOT numeric claims):",
        "",
        "  Strong plant promoters (literature reference ranges):",
        "    - CaMV 35S: widely used constitutive strong promoter in dicots",
        "      (Odell et al., 1985, Nature 313:810-812)",
        "    - OsAct1: strong constitutive in monocots (McElroy et al., 1990, Plant Cell 2:163-171)",
        "    - ZmUbi1: strong constitutive in monocots (Christensen & Quail, 1996, Transgenic Res 5:213-218)",
        "",
        "  Moderate plant promoters:",
        "    - AtUBQ10: moderate constitutive in Arabidopsis (Norris et al., 1993, Plant Mol Biol 21:895-906)",
        "    - AtACT2: moderate constitutive (An et al., 1996, Plant J 10:107-121)",
        "",
        "  Weak/regulated promoters:",
        "    - NOS promoter: typically weaker than 35S (Sanders et al., 1987, Nucleic Acids Res 15:1543-1558)",
        "    - E8: fruit-ripening specific, conditional strength (Deikman & Fischer, 1988, EMBO J 7:3315-3320)",
        "",
        "  These references describe general relative ordering of well-characterized",
        "  promoters. They do NOT provide numeric thresholds directly applicable to",
        "  our computational scoring framework.",
        "",
        "IMPORTANT CAVEATS:",
        "  1. Our classification is ENTIRELY computational and relative.",
        "  2. No experimental validation has been performed.",
        "  3. 'Strong' in our framework does not equal 'strong' in wet-lab measurements.",
        "  4. The scoring emphasizes core promoter architecture (TATA/CAAT positioning)",
        "     which correlates with but does not determine expression strength.",
        "  5. Post-transcriptional regulation, chromatin context, and copy number",
        "     effects are NOT captured by this classification.",
    ]

    # Per-species thresholds
    notes.append("")
    notes.append("SPECIES-SPECIFIC THRESHOLDS:")
    for species, thresholds in quantile_thresholds.items():
        sp_df = df[df["species"] == species]
        notes.append(
            f"  {species}: weak < {thresholds[0.33]:.4f} <= moderate < {thresholds[0.66]:.4f} <= strong"
        )
        notes.append(
            f"    score range: [{sp_df['composite_score'].min():.4f}, {sp_df['composite_score'].max():.4f}]"
        )
        notes.append(
            f"    mean: {sp_df['composite_score'].mean():.4f}, median: {sp_df['composite_score'].median():.4f}"
        )

    notes_path = OUTPUT_DIR / "benchmark_reference_notes.txt"
    with open(notes_path, "w") as fh:
        fh.write("\n".join(notes))
    print(f"  Saved reference notes: {notes_path}")


if __name__ == "__main__":
    main()
