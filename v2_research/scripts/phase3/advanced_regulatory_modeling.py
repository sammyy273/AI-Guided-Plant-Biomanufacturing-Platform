#!/usr/bin/env python3
"""
STEP 8: Advanced Regulatory Modeling — 3D Genome + m6A + Phase Separation.

Integrates three high-order regulatory effect layers:
  1. 3D genome architecture (TADs, enhancer proximity, chromatin loops)
  2. m6A epitranscriptome (methylation sites, mRNA stability, translation)
  3. Phase separation / P-body risk (sequestration, translation suppression)

INPUTS:
  outputs/phase3/final_construct_sequences.fasta
  outputs/phase3/cds_optimization_advanced.csv
  outputs/phase3/deployability_matrix.csv
  outputs/phase3/degradation_detail.json

OUTPUTS:
  outputs/phase3/advanced_regulatory_analysis.json
  outputs/phase3/advanced_regulatory_summary.txt
"""

import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODULES_DIR = BASE_DIR / "modules"
OUTPUT_DIR = BASE_DIR / "outputs" / "phase3"
sys.path.insert(0, str(BASE_DIR))

from modules.genomics.three_d_genome import ThreeDGenomeModel
from modules.biophysics.m6a_prediction import analyze_m6a
from modules.biophysics.phase_separation import assess_pbody_localization_risk


def load_fasta(path: Path) -> dict:
    """Load FASTA file into {header: sequence} dict."""
    seqs = {}
    header = None
    seq_parts = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    seqs[header] = "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header:
        seqs[header] = "".join(seq_parts)
    return seqs


def extract_species(header: str) -> str:
    """Extract species name from FASTA header."""
    for part in header.split():
        if "construct_" in part:
            return part.replace("construct_", "")
    return header.split()[0].replace("construct_", "")


def extract_cds_sequence(full_construct: str, promoter_len: int = 800, terminator_len: int = 182) -> str:
    """Extract CDS from construct (skip promoter + terminator)."""
    return full_construct[promoter_len:len(full_construct) - terminator_len]


def get_insertion_positions() -> dict:
    """Get candidate insertion positions for each species.

    These are representative safe harbor positions derived from
    the genome context scoring in Phase 2.
    """
    return {
        "tomato": {"chr": "1", "pos": 42000000},
        "rice": {"chr": "2", "pos": 12500000},
        "nbenthamiana": {"chr": "1", "pos": 5000000},
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print("STEP 8: Advanced Regulatory Modeling")
    print("  (3D Genome + m6A + Phase Separation)")
    print("=" * 60)

    # ── Load inputs ──────────────────────────────────────────────────────
    fasta_path = OUTPUT_DIR / "final_construct_sequences.fasta"
    if not fasta_path.exists():
        print("ERROR: final_construct_sequences.fasta not found. Run steps 1-7 first.")
        return

    constructs = load_fasta(fasta_path)
    insertion_positions = get_insertion_positions()

    # Load existing Phase 3 data for context
    deploy_data = {}
    deploy_path = OUTPUT_DIR / "deployability_matrix.csv"
    if deploy_path.exists():
        import csv
        with open(deploy_path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                deploy_data[row["species"]] = row

    # ── Run analysis per species ─────────────────────────────────────────
    results = {}
    for header, sequence in constructs.items():
        species = extract_species(header)
        print(f"\n  Analyzing: {species}")
        print(f"  Construct length: {len(sequence)} bp")

        # Extract CDS portion for mRNA-level analyses
        cds = extract_cds_sequence(sequence)
        print(f"  CDS length: {len(cds)} bp")

        # ── STEP 1: 3D Genome Modeling ──────────────────────────────────
        ins_pos = insertion_positions.get(species, {"chr": "1", "pos": 5000000})
        model_3d = ThreeDGenomeModel(species)
        spatial_result = model_3d.model_insertion_site(
            chromosome=ins_pos["chr"],
            position_bp=ins_pos["pos"],
            insert_sequence=sequence,
        )
        print(f"  3D: TAD compartment={spatial_result['tad_location'].get('compartment', '?')}, "
              f"expression_potential={spatial_result['3d_expression_potential']:.3f}, "
              f"high_risk={spatial_result['high_risk_inactive_loop']}")

        # ── STEP 2: m6A Epitranscriptome ────────────────────────────────
        m6a_result = analyze_m6a(cds)
        print(f"  m6A: {m6a_result['n_predicted_sites']} sites, "
              f"stability={m6a_result['stability_effect']['effect']}, "
              f"translation={m6a_result['translation_effect']['effect']}")

        # ── STEP 3: Phase Separation / P-Body Risk ──────────────────────
        pbody_result = assess_pbody_localization_risk(cds)
        print(f"  P-body: risk={pbody_result['pbody_localization_risk']}, "
              f"suppression={pbody_result['translation_suppression_risk']:.3f}")

        # ── Combine into per-species result ─────────────────────────────
        results[species] = {
            "species": species,
            "construct_header": header,
            "insertion_site": ins_pos,
            "step1_3d_genome": {
                "tad_location": spatial_result["tad_location"],
                "enhancer_proximity": spatial_result["enhancer_proximity"],
                "chromatin_loops": spatial_result["chromatin_loops"],
                "3d_expression_potential": spatial_result["3d_expression_potential"],
                "high_risk_inactive_loop": spatial_result["high_risk_inactive_loop"],
                "confidence": spatial_result["confidence"],
            },
            "step2_m6a": {
                "m6a_sites": m6a_result["m6a_sites"],
                "n_sites": m6a_result["n_predicted_sites"],
                "stability_effect": m6a_result["stability_effect"],
                "translation_effect": m6a_result["translation_effect"],
            },
            "step3_phase_separation": {
                "pbody_localization_risk": pbody_result["pbody_localization_risk"],
                "translation_suppression_risk": pbody_result["translation_suppression_risk"],
                "risk_factors": pbody_result["risk_factors"],
            },
        }

    # ── Compute final aggregated scores ──────────────────────────────────
    final_scores = {}
    for species, data in results.items():
        spatial_expr = data["step1_3d_genome"]["3d_expression_potential"]
        m6a_stability = data["step2_m6a"]["stability_effect"]["stability_score"]
        m6a_translation = data["step2_m6a"]["translation_effect"]["translation_efficiency"]
        pbody_risk = data["step3_phase_separation"]["translation_suppression_risk"]

        # Weighted final scores
        spatial_control = spatial_expr
        post_transcriptional = (
            m6a_stability * 0.4 +
            m6a_translation * 0.3 +
            (1.0 - pbody_risk) * 0.3
        )

        # Confidence = minimum confidence across all modules
        conf_3d = data["step1_3d_genome"]["confidence"]
        conf_m6a = 0.55  # m6A motif-based prediction confidence
        conf_pbody = 0.45  # P-body sequence-based prediction confidence
        overall_confidence = min(conf_3d, conf_m6a, conf_pbody)

        # Low-confidence flags
        low_confidence_flags = []
        if conf_3d < 0.4:
            low_confidence_flags.append("3D genome: no Hi-C data for this species")
        if data["step1_3d_genome"]["high_risk_inactive_loop"]:
            low_confidence_flags.append("3D genome: insertion in inactive/B compartment")
        if data["step2_m6a"]["n_sites"] == 0:
            low_confidence_flags.append("m6A: no sites predicted (may indicate data gap)")
        if pbody_risk > 0.5:
            low_confidence_flags.append("Phase separation: elevated P-body risk")

        final_scores[species] = {
            "spatial_expression_control": round(spatial_control, 4),
            "post_transcriptional_regulation": round(post_transcriptional, 4),
            "advanced_confidence": round(overall_confidence, 3),
            "low_confidence_flags": low_confidence_flags,
            "deployment_adjustment": _compute_deployment_adjustment(
                spatial_control, post_transcriptional, overall_confidence
            ),
        }

    # ── Save JSON output ─────────────────────────────────────────────────
    output_json = {
        "analysis_type": "advanced_regulatory_modeling",
        "species_analyzed": list(results.keys()),
        "per_species_results": results,
        "final_scores": final_scores,
        "computation_time_s": round(time.time() - t0, 2),
        "notes": [
            "All predictions are computational. No experimental validation.",
            "3D genome: TAD boundaries from published Hi-C (Dong et al. 2017, Liu et al. 2020).",
            "m6A: DRACH + UGUA motif scanning. Plant-specific stability rules.",
            "Phase separation: sequence-based P-body/SG risk estimation.",
            "N. benthamiana has limited Hi-C data — low confidence for 3D genome.",
        ],
    }

    json_path = OUTPUT_DIR / "advanced_regulatory_analysis.json"
    with open(json_path, "w") as fh:
        json.dump(output_json, fh, indent=2, default=str)
    print(f"\n  Saved: {json_path}")

    # ── Generate text summary ────────────────────────────────────────────
    _generate_text_summary(results, final_scores)
    print(f"  Saved: {OUTPUT_DIR / 'advanced_regulatory_summary.txt'}")


def _compute_deployment_adjustment(spatial: float, post_tx: float, confidence: float) -> str:
    """Compute deployment recommendation based on advanced regulatory scores."""
    if spatial < 0.3 or post_tx < 0.3:
        return "HIGH_RISK — major regulatory concern identified"
    elif spatial < 0.5 and post_tx < 0.5:
        return "MODERATE_RISK — multiple suboptimal regulatory features"
    elif confidence < 0.35:
        return "LOW_CONFIDENCE — insufficient data for reliable prediction"
    elif spatial > 0.6 and post_tx > 0.6:
        return "FAVORABLE — regulatory context supports expression"
    else:
        return "CONDITIONAL — some regulatory risk factors present"


def _generate_text_summary(results: dict, final_scores: dict):
    """Generate human-readable text summary."""
    lines = [
        "ADVANCED REGULATORY MODELING SUMMARY",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=" * 60,
        "STEP 1 — 3D GENOME ARCHITECTURE",
        "=" * 60,
        "",
    ]

    for species, data in results.items():
        s1 = data["step1_3d_genome"]
        tad = s1["tad_location"]
        enh = s1["enhancer_proximity"]
        loops = s1["chromatin_loops"]
        lines.extend([
            f"  {species.upper()}",
            f"    TAD compartment: {tad.get('compartment', '?')}",
            f"    TAD position: {tad.get('tad_start_mb', '?')} - {tad.get('tad_end_mb', '?')} Mb",
            f"    Inactive loop risk: {'YES' if s1['high_risk_inactive_loop'] else 'NO'}",
            f"    Nearest enhancer: {enh.get('nearest_enhancer', 'None')} "
            f"({enh.get('distance_kb', '?')} kb, P={enh.get('interaction_probability', '?')})",
            f"    Chromatin loops nearby: {loops.get('n_nearby_loops', 0)}",
            f"    3D expression potential: {s1['3d_expression_potential']:.3f}",
            f"    Confidence: {s1['confidence']:.2f}",
            "",
        ])

    lines.extend([
        "=" * 60,
        "STEP 2 — m6A EPITRANSCRIPTOME",
        "=" * 60,
        "",
    ])

    for species, data in results.items():
        s2 = data["step2_m6a"]
        stab = s2["stability_effect"]
        trans = s2["translation_effect"]
        lines.extend([
            f"  {species.upper()}",
            f"    Predicted m6A sites: {s2['n_sites']}",
            f"    Stability effect: {stab['effect']} (score: {stab['stability_score']:.3f})",
            f"    Translation effect: {trans['effect']} (score: {trans['translation_efficiency']:.3f})",
            f"    3'UTR sites: {stab.get('n_3utr_sites', '?')} | "
            f"CDS sites: {stab.get('n_cds_sites', '?')} | "
            f"5'UTR sites: {stab.get('n_5utr_sites', '?')}",
            "",
        ])

    lines.extend([
        "=" * 60,
        "STEP 3 — PHASE SEPARATION / P-BODIES",
        "=" * 60,
        "",
    ])

    for species, data in results.items():
        s3 = data["step3_phase_separation"]
        rf = s3["risk_factors"]
        lines.extend([
            f"  {species.upper()}",
            f"    P-body localization risk: {s3['pbody_localization_risk'].upper()}",
            f"    Translation suppression risk: {s3['translation_suppression_risk']:.3f}",
            f"    3'UTR length: {rf['3utr_length']} nt ({rf['3utr_length_risk']} risk)",
            f"    ARE sites in 3'UTR: {rf['n_are_sites_in_3utr']}",
            f"    G-quadruplexes: {rf['n_g_quadruplexes']}",
            f"    3'UTR GC: {rf['3utr_gc']:.1%}",
            "",
        ])

    lines.extend([
        "=" * 60,
        "FINAL SCORES",
        "=" * 60,
        "",
    ])

    for species, scores in final_scores.items():
        lines.extend([
            f"  {species.upper()}",
            f"    spatial_expression_control:     {scores['spatial_expression_control']:.4f}",
            f"    post_transcriptional_regulation: {scores['post_transcriptional_regulation']:.4f}",
            f"    advanced_confidence:             {scores['advanced_confidence']:.3f}",
            f"    deployment_adjustment:           {scores['deployment_adjustment']}",
        ])
        if scores["low_confidence_flags"]:
            lines.append("    Low-confidence flags:")
            for flag in scores["low_confidence_flags"]:
                lines.append(f"      - {flag}")
        lines.append("")

    lines.extend([
        "=" * 60,
        "STRICT DISCLAIMERS",
        "=" * 60,
        "",
        "  - All predictions are computational. No experimental validation.",
        "  - 3D genome data is from published Hi-C. N. benthamiana has no",
        "    published Hi-C — those results use tomato as proxy.",
        "  - m6A predictions are motif-based (DRACH + UGUA). Actual m6A",
        "    requires MeRIP-seq or nanopore direct RNA sequencing.",
        "  - Phase separation risk is estimated from sequence features.",
        "    Actual P-body localization requires in vivo imaging (DCP1-GFP).",
        "  - Confidence scores reflect data availability, not accuracy.",
        "",
    ])

    summary_path = OUTPUT_DIR / "advanced_regulatory_summary.txt"
    with open(summary_path, "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
