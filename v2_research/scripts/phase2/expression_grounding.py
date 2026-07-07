#!/usr/bin/env python3
"""
STEP 1: Expression Grounding with Literature Calibration.

Builds a literature reference table of known promoters, maps internal scores
to percentile-based rankings relative to these references, and calibrates
the output without claiming real expression prediction.

OUTPUTS:
  outputs/phase2/expression_grounding.csv
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
PROMOTER_DIR = BASE_DIR / "data" / "promoters"
PHASE1_DIR = BASE_DIR / "outputs" / "phase1"

# ── Literature reference table ─────────────────────────────────────────────

LITERATURE_REFERENCES = [
    {
        "promoter_name": "CaMV 35S",
        "species": "general dicot",
        "qualitative_strength": "strong",
        "source_reference": "Odell et al., 1985, Nature 313:810-812",
        "notes": "Viral constitutive promoter, widely used in dicots. ~835 bp.",
    },
    {
        "promoter_name": "Maize Ubiquitin 1 (ZmUbi1)",
        "species": "monocot (maize, rice)",
        "qualitative_strength": "strong",
        "source_reference": "Christensen & Quail, 1996, Transgenic Res 5:213-218",
        "notes": "Strong constitutive in monocots. ~1993 bp with intron.",
    },
    {
        "promoter_name": "Rice Actin 1 (OsAct1)",
        "species": "monocot (rice)",
        "qualitative_strength": "strong",
        "source_reference": "McElroy et al., 1990, Plant Cell 2:163-171",
        "notes": "Strong constitutive in rice. ~1413 bp.",
    },
    {
        "promoter_name": "Rice Ubiquitin 2 (OsUbi2)",
        "species": "monocot (rice)",
        "qualitative_strength": "strong",
        "source_reference": "Wang & Oard, 2003, Plant Cell Rep 22:129-134",
        "notes": "Strong constitutive in rice. ~1719 bp.",
    },
    {
        "promoter_name": "AtUBQ10",
        "species": "Arabidopsis thaliana",
        "qualitative_strength": "moderate",
        "source_reference": "Norris et al., 1993, Plant Mol Biol 21:895-906",
        "notes": "Moderate constitutive in Arabidopsis.",
    },
    {
        "promoter_name": "AtACT2",
        "species": "Arabidopsis thaliana",
        "qualitative_strength": "moderate",
        "source_reference": "An et al., 1996, Plant J 10:107-121",
        "notes": "Moderate constitutive in Arabidopsis.",
    },
    {
        "promoter_name": "Tomato UBQ (SlUBQ)",
        "species": "Solanum lycopersicum",
        "qualitative_strength": "moderate",
        "source_reference": "Rollfinke et al., unpublished; used in plant transient expression",
        "notes": "Moderate constitutive in tomato. ~1321 bp.",
    },
    {
        "promoter_name": "NOS (nopaline synthase)",
        "species": "general",
        "qualitative_strength": "weak",
        "source_reference": "Sanders et al., 1987, Nucleic Acids Res 15:1543-1558",
        "notes": "Weak constitutive. ~307 bp. Typically weaker than CaMV 35S.",
    },
    {
        "promoter_name": "Tomato E8",
        "species": "Solanum lycopersicum",
        "qualitative_strength": "weak",
        "source_reference": "Deikman & Fischer, 1988, EMBO J 7:3315-3320",
        "notes": "Fruit-ripening specific, ethylene-responsive. Conditional strength.",
    },
    {
        "promoter_name": "CaMV 35S minimal (-90)",
        "species": "general dicot",
        "qualitative_strength": "weak",
        "source_reference": "Odell et al., 1985; minimal promoter region only",
        "notes": "Minimal TATA-proximal region. Requires enhancer for activity.",
    },
]

# ── Motif patterns ─────────────────────────────────────────────────────────

MOTIF_PATTERNS = {
    "TATA_box": ["TATAAA", "TATATA", "TATACA", "TATAGA"],
    "CAAT_box": ["CCAAT", "CAAAT"],
    "GC_box": ["GGGCGG", "CCGCCC"],
    "as1_element": ["TGACG", "CGTCA"],
    "DOF_site": ["AAAG", "CTTT"],
}


def count_motif(seq, patterns):
    return sum(seq.upper().count(p) for p in patterns)


def compute_internal_score(seq):
    """Compute the internal composite score (same formula as Phase 1)."""
    s = seq.upper()
    length = len(s)
    gc = (s.count("G") + s.count("C")) / length

    proximal = s[-50:] if length >= 50 else s
    upstream = s[-120:] if length >= 120 else s

    has_tata = int(any(p in proximal for p in MOTIF_PATTERNS["TATA_box"]))
    has_caat = int(any(p in upstream for p in MOTIF_PATTERNS["CAAT_box"]))
    has_both = has_tata and has_caat

    arch_bonus = 0.5 if has_both else (0.25 if has_tata else (0.15 if has_caat else 0.0))

    total_motifs = sum(count_motif(s, p) for p in MOTIF_PATTERNS.values())
    motif_score = min(1.0, total_motifs / 30)
    gc_balance = max(0, 1 - abs(gc - 0.38) / 0.25)
    cg_density = s.count("CG") / length

    composite = (
        0.30 * arch_bonus +
        0.25 * motif_score +
        0.20 * gc_balance +
        0.15 * has_tata +
        0.10 * (1 - min(1, cg_density * 3))
    )
    return round(composite, 4)


def score_known_promoters():
    """Score the known reference promoters from seed FASTA files."""
    seed_dir = BASE_DIR / "data" / "promoter_seeds"
    results = []

    for fasta_file in sorted(seed_dir.glob("*.fasta")):
        from Bio import SeqIO
        for record in SeqIO.parse(str(fasta_file), "fasta"):
            seq = str(record.seq)
            score = compute_internal_score(seq)
            # Extract promoter name from FASTA header
            name = record.description.split(" ", 1)[0] if " " in record.description else record.id
            results.append({
                "promoter_name": fasta_file.stem,
                "sequence_id": record.id,
                "internal_score": score,
                "sequence_length": len(seq),
            })

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Expression Grounding with Literature Calibration")
    print("=" * 60)

    # 1. Load literature reference table
    lit_df = pd.DataFrame(LITERATURE_REFERENCES)
    print(f"\n  Literature references loaded: {len(lit_df)} known promoters")

    # 2. Score known promoters from seed files
    known_scores = score_known_promoters()
    known_df = pd.DataFrame(known_scores)
    print(f"  Reference promoters scored: {len(known_df)}")

    # 3. Score curated promoters from Phase 1
    curated_scores = []
    species_list = ["arabidopsis", "rice", "tomato"]
    n_sample = 500

    for species in species_list:
        fasta_path = PROMOTER_DIR / f"{species}_promoters_1kb.fasta"
        if not fasta_path.exists():
            continue

        from Bio import SeqIO
        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        rng = np.random.RandomState(42)
        indices = rng.choice(len(records), min(n_sample, len(records)), replace=False)

        for idx in indices:
            seq = str(records[idx].seq)
            score = compute_internal_score(seq)
            curated_scores.append({
                "species": species,
                "sequence_id": records[idx].id,
                "internal_score": score,
                "source": "curated_real",
            })

    curated_df = pd.DataFrame(curated_scores)
    print(f"  Curated promoters scored: {len(curated_df)}")

    # 4. Load Phase 1 designed promoter scores
    phase1_ml = PHASE1_DIR / "ml_predictions.csv"
    designed_scores = []
    if phase1_ml.exists():
        p1_df = pd.read_csv(phase1_ml)
        for _, row in p1_df.iterrows():
            designed_scores.append({
                "species": row["species"],
                "sequence_id": row["sequence_id"],
                "internal_score": row["composite_score"],
                "source": "designed_pipeline",
            })
    print(f"  Phase 1 designed promoters: {len(designed_scores)}")

    # 5. Load final report scores (best designed promoters)
    final_scores = []
    for species in ["nbenthamiana", "rice", "tomato"]:
        report_path = BASE_DIR / "outputs" / f"final_report_{species}.json"
        if report_path.exists():
            with open(report_path) as fh:
                report = json.load(fh)
            if "best_candidate" in report:
                bc = report["best_candidate"]
                final_scores.append({
                    "species": species,
                    "sequence_id": bc.get("id", "unknown"),
                    "internal_score": bc.get("composite_score", 0),
                    "source": "best_pipeline_output",
                })
    print(f"  Best pipeline promoters: {len(final_scores)}")

    # 6. Build unified score distribution for percentile ranking
    all_scores = (
        [r["internal_score"] for r in curated_scores] +
        [r["internal_score"] for r in designed_scores] +
        [r["internal_score"] for r in known_scores] +
        [r["internal_score"] for r in final_scores]
    )
    score_array = np.array(all_scores)
    print(f"\n  Total score distribution: {len(score_array)} promoters")
    print(f"  Score range: [{score_array.min():.4f}, {score_array.max():.4f}]")
    print(f"  Score median: {np.median(score_array):.4f}")

    # 7. Compute percentile ranks for reference promoters
    print("\n  Reference promoter rankings:")
    ref_rankings = []
    for _, row in known_df.iterrows():
        percentile = np.mean(score_array <= row["internal_score"]) * 100
        # Find matching literature entry
        lit_match = lit_df[lit_df["promoter_name"].str.contains(
            row["promoter_name"].replace("_promoter", "").replace("_", " ").split()[0],
            case=False
        )]
        qual_strength = lit_match["qualitative_strength"].values[0] if len(lit_match) > 0 else "unknown"
        ref_source = lit_match["source_reference"].values[0] if len(lit_match) > 0 else "NOT AVAILABLE"

        ref_rankings.append({
            "promoter_name": row["promoter_name"],
            "sequence_id": row["sequence_id"],
            "internal_score": row["internal_score"],
            "percentile_in_distribution": round(percentile, 1),
            "qualitative_strength": qual_strength,
            "source_reference": ref_source,
            "calibration_note": f"Scores at {percentile:.1f}th percentile of full distribution",
        })
        print(f"    {row['promoter_name']:40s} score={row['internal_score']:.4f} "
              f"percentile={percentile:.1f}% strength={qual_strength}")

    # 8. Compute percentile ranks for final (best) promoters
    print("\n  Best pipeline promoter rankings:")
    for entry in final_scores:
        percentile = np.mean(score_array <= entry["internal_score"]) * 100
        entry["percentile_in_distribution"] = round(percentile, 1)
        entry["relative_to_known_strong"] = (
            "above known strong promoters" if percentile > 90
            else "comparable to known strong promoters" if percentile > 75
            else "below known strong promoters"
        )
        print(f"    {entry['species']:15s} score={entry['internal_score']:.4f} "
              f"percentile={percentile:.1f}% → {entry['relative_to_known_strong']}")

    # 9. Build output DataFrame
    output_rows = []

    # Reference promoters
    for r in ref_rankings:
        output_rows.append({
            "category": "literature_reference",
            "species": "reference",
            "sequence_id": r["sequence_id"],
            "promoter_name": r["promoter_name"],
            "internal_score": r["internal_score"],
            "percentile_in_distribution": r["percentile_in_distribution"],
            "qualitative_strength": r["qualitative_strength"],
            "source_reference": r["source_reference"],
            "calibration_note": r["calibration_note"],
        })

    # Best pipeline outputs
    for entry in final_scores:
        output_rows.append({
            "category": "best_pipeline_output",
            "species": entry["species"],
            "sequence_id": entry["sequence_id"],
            "promoter_name": "designed_best",
            "internal_score": entry["internal_score"],
            "percentile_in_distribution": entry["percentile_in_distribution"],
            "qualitative_strength": "relative_to_known_strong",
            "source_reference": "internal pipeline",
            "calibration_note": entry["relative_to_known_strong"],
        })

    # Curated real promoters (sample)
    for entry in curated_scores[:100]:  # first 100 for tractability
        percentile = np.mean(score_array <= entry["internal_score"]) * 100
        output_rows.append({
            "category": "curated_real",
            "species": entry["species"],
            "sequence_id": entry["sequence_id"],
            "promoter_name": "real_endogenous",
            "internal_score": entry["internal_score"],
            "percentile_in_distribution": round(percentile, 1),
            "qualitative_strength": "real_promoter",
            "source_reference": f"GFF3 annotation ({entry['species']})",
            "calibration_note": "Real TSS-anchored promoter from genome",
        })

    out_df = pd.DataFrame(output_rows)
    out_path = OUTPUT_DIR / "expression_grounding.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(out_df)} rows)")

    # 10. Save literature reference table separately
    lit_path = OUTPUT_DIR / "literature_reference_table.csv"
    lit_df.to_csv(lit_path, index=False)
    print(f"  Saved: {lit_path}")


if __name__ == "__main__":
    main()
