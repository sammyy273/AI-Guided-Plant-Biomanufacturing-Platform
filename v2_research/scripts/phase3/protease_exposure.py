#!/usr/bin/env python3
"""
STEP 2: Structure-Aware Protease Exposure Mapping.

Uses structure analysis from Step 1 to recompute degradation scores
with exposure weighting — buried motifs contribute less than exposed ones.

Replaces naive motif counting with spatial realism.

OUTPUTS:
  outputs/phase3/degradation_structure_adjusted.csv
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

OUTPUT_DIR = BASE_DIR / "outputs" / "phase3"
PHASE2_DIR = BASE_DIR / "outputs" / "phase2"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 2: Structure-Aware Protease Exposure")
    print("=" * 60)

    # Load structure analysis from Step 1
    struct_path = OUTPUT_DIR / "structure_analysis.json"
    if not struct_path.exists():
        print("  ERROR: structure_analysis.json not found. Run Step 1 first.")
        return

    with open(struct_path) as fh:
        structure = json.load(fh)

    # Load Phase 2 stability data
    with open(PHASE2_DIR / "protein_stability_detail.json") as fh:
        stability = json.load(fh)

    rasa = structure["solvent_accessibility"]["per_residue"]
    protease_motifs = structure["protease_accessible_motifs"]["details"]
    seq_len = structure["sequence_length"]

    print(f"  Protein: {structure['protein_id']} ({seq_len} aa)")
    print(f"  Structure source: {structure['structure_source']}")
    print(f"  Protease motifs total: {structure['protease_accessible_motifs']['total']}")
    print(f"  Surface-accessible: {structure['protease_accessible_motifs']['accessible']}")
    print(f"  Buried: {structure['protease_accessible_motifs']['buried']}")

    # ── Recompute exposure-weighted degradation ─────────────────────────

    # Group motifs by family and exposure status
    family_stats = {}
    for motif in protease_motifs:
        family = motif["family"]
        if family not in family_stats:
            family_stats[family] = {"total": 0, "exposed": 0, "buried": 0, "exposure_scores": []}
        family_stats[family]["total"] += 1
        if motif["accessible"]:
            family_stats[family]["exposed"] += 1
        else:
            family_stats[family]["buried"] += 1
        family_stats[family]["exposure_scores"].append(motif["exposure"])

    print("\n  Motif exposure by family:")
    for family, stats in family_stats.items():
        mean_exp = np.mean(stats["exposure_scores"])
        print(f"    {family:25s}: {stats['total']} total, {stats['exposed']} exposed, "
              f"mean exposure={mean_exp:.3f}")

    # ── Exposure-weighted degradation scoring ───────────────────────────

    # Only exposed motifs count toward degradation risk
    # Weight by exposure score (0-1)
    exposure_weighted = {}
    for family, stats in family_stats.items():
        exposed_weighted_count = sum(stats["exposure_scores"])
        exposure_weighted[family] = round(exposed_weighted_count, 3)
        print(f"    {family:25s}: exposure-weighted count = {exposed_weighted_count:.3f}")

    # Compartment-specific protease risk multipliers
    compartment_multipliers = {
        "extracellular": {"SBT1_subtilase": 1.3, "C1A_cysteine": 0.8, "A1_aspartic": 1.5},
        "ER": {"SBT1_subtilase": 0.3, "C1A_cysteine": 0.2, "A1_aspartic": 0.5},
        "cytoplasm": {"SBT1_subtilase": 0.2, "C1A_cysteine": 0.3, "A1_aspartic": 0.3},
        "vacuole": {"SBT1_subtilase": 0.4, "C1A_cysteine": 1.5, "A1_aspartic": 0.6},
        "membrane": {"SBT1_subtilase": 0.6, "C1A_cysteine": 0.4, "A1_aspartic": 0.8},
    }

    # Lysine ubiquitination: weight by surface exposure
    ubiquitination = stability["ubiquitination"]
    # Approximate: lysines that are surface-exposed contribute more
    sequence_heuristic = structure.get("confidence_note", "")
    # For lysine exposure, use mean RASA of all residues (proxy)
    mean_rasa = np.mean(rasa)
    lys_exposure_factor = min(1.0, mean_rasa * 1.5)

    # Compute per-compartment degradation
    rows = []
    for compartment, mults in compartment_multipliers.items():
        # Protease family contributions (exposure-weighted)
        sbt1_risk = exposure_weighted.get("SBT1_subtilase", 0) * mults["SBT1_subtilase"]
        c1a_risk = exposure_weighted.get("C1A_cysteine", 0) * mults["C1A_cysteine"]
        a1_risk = exposure_weighted.get("A1_aspartic", 0) * mults["A1_aspartic"]

        # Normalize by sequence length
        sbt1_norm = min(1.0, sbt1_risk / (seq_len * 0.02))
        c1a_norm = min(1.0, c1a_risk / (seq_len * 0.02))
        a1_norm = min(1.0, a1_risk / (seq_len * 0.02))

        # Ubiquitination contribution (compartment-dependent)
        ubiquitin_mult = {
            "extracellular": 0.3, "ER": 0.5, "cytoplasm": 1.5,
            "vacuole": 0.5, "membrane": 0.6,
        }.get(compartment, 1.0)
        ubiquitin_risk = ubiquitination["ubiquitination_risk"] * lys_exposure_factor * ubiquitin_mult

        # PEST contribution (modulated by disorder)
        pest_count = stability["pest_regions"]["count"]
        pest_factor = min(1.0, pest_count / 20)
        # Weight PEST by exposure (disordered regions are more exposed)
        disorder_frac = structure["disorder"]["disordered_fraction"]
        pest_risk = pest_factor * (0.5 + 0.5 * disorder_frac)

        # Combined score
        total_risk = (
            0.30 * sbt1_norm +
            0.20 * c1a_norm +
            0.10 * a1_norm +
            0.20 * ubiquitin_risk +
            0.20 * pest_risk
        )
        total_risk = min(1.0, total_risk)

        risk_class = "HIGH" if total_risk >= 0.66 else ("MEDIUM" if total_risk >= 0.33 else "LOW")

        dominant = max(
            [("SBT1", sbt1_norm), ("C1A", c1a_norm), ("A1", a1_norm), ("ubiquitin", ubiquitin_risk)],
            key=lambda x: x[1]
        )

        rows.append({
            "compartment": compartment,
            "sbt1_exposure_weighted": round(sbt1_norm, 4),
            "c1a_exposure_weighted": round(c1a_norm, 4),
            "a1_exposure_weighted": round(a1_norm, 4),
            "ubiquitin_exposure_adjusted": round(ubiquitin_risk, 4),
            "pest_disorder_adjusted": round(pest_risk, 4),
            "total_degradation_risk": round(total_risk, 4),
            "risk_class": risk_class,
            "dominant_factor": dominant[0],
        })

        print(f"\n  {compartment}:")
        print(f"    SBT1: {sbt1_norm:.4f}, C1A: {c1a_norm:.4f}, A1: {a1_norm:.4f}")
        print(f"    Ubiquitin: {ubiquitin_risk:.4f}, PEST: {pest_risk:.4f}")
        print(f"    Total: {total_risk:.4f} ({risk_class}), dominant: {dominant[0]}")

    # Add summary row for primary routing (extracellular)
    primary = "extracellular"
    primary_row = [r for r in rows if r["compartment"] == primary][0]

    rows.append({
        "compartment": "SUMMARY (extracellular routing)",
        "sbt1_exposure_weighted": primary_row["sbt1_exposure_weighted"],
        "c1a_exposure_weighted": primary_row["c1a_exposure_weighted"],
        "a1_exposure_weighted": primary_row["a1_exposure_weighted"],
        "ubiquitin_exposure_adjusted": primary_row["ubiquitin_exposure_adjusted"],
        "pest_disorder_adjusted": primary_row["pest_disorder_adjusted"],
        "total_degradation_risk": primary_row["total_degradation_risk"],
        "risk_class": primary_row["risk_class"],
        "dominant_factor": primary_row["dominant_factor"],
    })

    out_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "degradation_structure_adjusted.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(out_df)} rows)")

    # Save detailed JSON for downstream use
    detail = {
        "structure_source": structure["structure_source"],
        "exposure_weighting_applied": True,
        "mean_surface_rasa": round(mean_rasa, 4),
        "lysine_exposure_factor": round(lys_exposure_factor, 4),
        "disorder_fraction": structure["disorder"]["disordered_fraction"],
        "primary_routing": primary,
        "primary_degradation_risk": primary_row["total_degradation_risk"],
        "primary_risk_class": primary_row["risk_class"],
        "primary_dominant_factor": primary_row["dominant_factor"],
        "phase2_naive_risk": stability.get("degradation_by_compartment", {}).get("extracellular", {}).get("degradation_score", "N/A"),
        "improvement_note": "Exposure-weighted scoring reduces overcounting of buried protease motifs.",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    detail_path = OUTPUT_DIR / "degradation_detail.json"
    with open(detail_path, "w") as fh:
        json.dump(detail, fh, indent=2)
    print(f"  Saved: {detail_path}")


if __name__ == "__main__":
    main()
