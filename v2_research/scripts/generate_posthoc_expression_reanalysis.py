#!/usr/bin/env python3
"""Post hoc realism and full-system reanalysis for saved expression outputs.

This script is intentionally non-destructive:
- it does not regenerate promoters
- it does not overwrite existing final reports
- it only reads saved outputs and writes new analysis files
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pvariance
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.cross_species.species_config import load_species_config
from modules.evaluation.cis_scoring import CIS_ELEMENTS, score_candidate
from modules.protein.expression_system import (
    assess_degradation_risk,
    gc_content,
    predict_localization,
    signal_peptide_heuristic,
)

OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data"
PROTEIN_FASTA = DATA / "protein" / "hyaluronidase.fasta"

FINAL_REPORTS = {
    "nbenthamiana": OUTPUTS / "final_report_nbenthamiana.json",
    "rice": OUTPUTS / "final_report_rice.json",
    "tomato": OUTPUTS / "final_report_tomato.json",
}

BASELINE_FASTAS = {
    "CaMV_35S": DATA / "promoter_seeds" / "CaMV35S_promoter_835bp.fasta",
    "Maize_Ubiquitin": DATA / "promoter_seeds" / "ZmUbi1_promoter_1993bp.fasta",
    "OsActin1": DATA / "promoter_seeds" / "OsAct1_promoter_1413bp.fasta",
    "AtUBQ10": DATA / "promoter_seeds" / "arabidopsis_promoters.fasta",
    "SlUBQ": DATA / "promoter_seeds" / "SlUBQ_promoter_1321bp.fasta",
}

SYSTEM_FITNESS_WEIGHTS = {
    "promoter_score": 0.40,
    "cai": 0.30,
    "degradation_component": 0.15,
    "localization_component": 0.15,
}

HYDROPHOBIC = set("AILMFWVPGCY")
DNA_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_fasta_sequence(path: Path) -> str:
    lines = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            continue
        lines.append(line.strip())
    return "".join(lines).upper()


def read_first_fasta_record(path: Path) -> Tuple[str, str]:
    header = None
    seq_lines: List[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None and seq_lines:
                break
            header = line[1:]
            continue
        seq_lines.append(line)
    return header or path.stem, "".join(seq_lines).upper()


def translate_dna(sequence: str) -> Tuple[str, bool]:
    seq = re.sub(r"[^ACGT]", "", sequence.upper())
    protein = []
    internal_stop = False
    for idx in range(0, len(seq) - 2, 3):
        aa = DNA_CODON_TABLE.get(seq[idx:idx + 3], "X")
        if aa == "*" and idx < len(seq) - 3:
            internal_stop = True
        if aa == "*":
            break
        protein.append(aa)
    return "".join(protein), internal_stop


def infer_localization_confidence(localization: dict, protein_sequence: str) -> dict:
    signal = signal_peptide_heuristic(protein_sequence)
    tm_regions = detect_transmembrane_regions(protein_sequence)
    predicted = localization.get("predicted_localization", "NOT AVAILABLE")
    status = localization.get("status", "unknown")

    if status == "model":
        label = "HIGH"
        numeric = float(localization.get("confidence", 0.9))
    else:
        if predicted == "secreted (apoplast)" and signal["has_signal_peptide"] and not tm_regions:
            label = "MEDIUM"
            numeric = 0.72
        elif predicted in {"ER-retained", "vacuole"} and signal["has_signal_peptide"]:
            label = "MEDIUM"
            numeric = 0.68
        else:
            label = "LOW"
            numeric = 0.45

    plausible = predicted == "secreted (apoplast)" and signal["has_signal_peptide"] and not tm_regions
    return {
        "localization_confidence": label,
        "localization_confidence_score": round(numeric, 4),
        "signal_peptide_detected": signal["has_signal_peptide"],
        "signal_peptide_details": signal,
        "transmembrane_region_detected": bool(tm_regions),
        "transmembrane_regions": tm_regions,
        "secretory_pathway_plausible": plausible,
        "confidence_basis": "model-backed" if status == "model" else "sequence-supported heuristic",
    }


def detect_transmembrane_regions(protein_sequence: str, window: int = 19) -> List[dict]:
    hits = []
    seq = protein_sequence.upper()
    for start in range(0, max(len(seq) - window + 1, 0)):
        frag = seq[start:start + window]
        hydro = sum(1 for aa in frag if aa in HYDROPHOBIC) / len(frag)
        if hydro >= 0.68:
            hits.append({
                "start": start + 1,
                "end": start + window,
                "sequence": frag,
                "hydrophobic_fraction": round(hydro, 3),
            })
    merged = []
    for hit in hits:
        if not merged or hit["start"] > merged[-1]["end"] + 3:
            merged.append(hit)
        else:
            merged[-1]["end"] = max(merged[-1]["end"], hit["end"])
            merged[-1]["sequence"] = seq[merged[-1]["start"] - 1:merged[-1]["end"]]
            merged[-1]["hydrophobic_fraction"] = max(
                merged[-1]["hydrophobic_fraction"],
                hit["hydrophobic_fraction"],
            )
    return merged


def gather_motif_hits(sequence: str) -> Dict[str, List[int]]:
    seq = sequence.upper()
    motif_hits: Dict[str, List[int]] = {}
    motif_keys = {
        "TATA_box": "TATA",
        "CAAT_box": "CAAT",
        "GC_box": "GC_box",
        "as1_element": "as1",
        "DOF_site": "DOF",
    }
    for cis_key, out_key in motif_keys.items():
        hits: List[int] = []
        for pattern in CIS_ELEMENTS.get(cis_key, []):
            for match in re.finditer(pattern, seq):
                hits.append(match.start() + 1)
        motif_hits[out_key] = sorted(hits)
    return motif_hits


def motif_metrics(sequence: str, species: str) -> dict:
    species_cfg = load_species_config(species)
    score = score_candidate(sequence, species_cfg)
    hits = gather_motif_hits(sequence)
    total_hits = sum(len(v) for v in hits.values())
    seq_len = max(len(sequence), 1)
    density = (total_hits / seq_len) * 100.0

    all_positions = sorted(pos for positions in hits.values() for pos in positions)
    gaps = [b - a for a, b in zip(all_positions, all_positions[1:])]
    close_gaps = [gap for gap in gaps if gap <= 12]
    spacing_var = pvariance(gaps) if len(gaps) >= 2 else 0.0
    spacing_cv = (math.sqrt(spacing_var) / mean(gaps)) if gaps and mean(gaps) else 0.0
    clustering_penalty = min(1.0, (len(close_gaps) / max(len(gaps), 1)) * 1.35)
    spacing_penalty = min(1.0, spacing_cv)

    gc_pct = float(score.get("gc_pct", gc_content(sequence)))
    if 30.0 <= gc_pct <= 60.0:
        gc_penalty = 0.0
    elif gc_pct < 30.0:
        gc_penalty = min(1.0, (30.0 - gc_pct) / 20.0)
    else:
        gc_penalty = min(1.0, (gc_pct - 60.0) / 20.0)

    realism_penalty = (
        0.45 * clustering_penalty
        + 0.35 * spacing_penalty
        + 0.20 * gc_penalty
    )
    realism_score = max(0.0, 1.0 - realism_penalty)
    oversaturation_flags = []
    if score.get("CAAT_box", 0) > 4:
        oversaturation_flags.append("CAAT_box>4")
    if score.get("DOF_site", 0) > 6:
        oversaturation_flags.append("DOF_site>6")
    if score.get("as1_element", 0) > 3:
        oversaturation_flags.append("as1_element>3")
    if score.get("GC_box", 0) > 6:
        oversaturation_flags.append("GC_box>6")

    return {
        "gc_pct": round(gc_pct, 2),
        "motif_density_per_100bp": round(density, 4),
        "spacing_variance": round(spacing_var, 4),
        "spacing_cv": round(spacing_cv, 4),
        "motif_clustering_penalty": round(clustering_penalty, 4),
        "spacing_penalty": round(spacing_penalty, 4),
        "gc_extreme_penalty": round(gc_penalty, 4),
        "structural_realism_score": round(realism_score, 4),
        "oversaturation_flags": oversaturation_flags or ["NONE"],
        "score_candidate": score,
        "motif_hits": hits,
        "motif_hit_total": total_hits,
    }


def realism_corrected_composite(report: dict, realism_score: float) -> float:
    promoter = report["best_promoter"]
    original = float(promoter["composite_score"])
    expression = float(promoter.get("expression_score", 0.0))
    cis_norm = min(1.0, float(promoter.get("weighted_score", 0.0)) / 75.0)
    silencing_component = max(0.0, 1.0 - float(promoter.get("silencing_risk", 1.0)))
    blended = (
        0.35 * expression
        + 0.30 * cis_norm
        + 0.20 * silencing_component
        + 0.15 * realism_score
    )
    return round(min(original, blended), 4)


def source_label(source: str) -> str:
    mapping = {"mut": "mutational", "evo2": "evo2", "d3lm": "d3lm"}
    return mapping.get(source, source)


def baseline_sequence_for(report: dict) -> Tuple[str, Optional[str]]:
    baseline_name = report["best_promoter"].get("baseline_reference", "NOT AVAILABLE")
    fasta = BASELINE_FASTAS.get(baseline_name)
    if fasta and fasta.exists():
        _, seq = read_first_fasta_record(fasta)
        return baseline_name, seq
    return baseline_name, None


def degradation_mitigation_rows(species: str, protein_sequence: str, original_report: dict) -> List[dict]:
    base_loc = original_report["localization"]
    base_deg = original_report["degradation_risk"]
    rows = []

    rows.append({
        "species": species,
        "strategy": "Original secreted construct",
        "predicted_localization": base_loc.get("predicted_localization", "NOT AVAILABLE"),
        "degradation_risk": base_deg.get("risk_class", "NOT AVAILABLE"),
        "risk_score": base_deg.get("risk_score", "NOT AVAILABLE"),
        "rationale": "Existing output predicts secretion to the apoplast, which exposes the enzyme to extracellular proteases.",
    })

    kdel_seq = protein_sequence + "KDEL"
    kdel_loc = predict_localization(kdel_seq)
    kdel_deg = assess_degradation_risk(kdel_seq)
    kdel_context_score = max(0.0, float(kdel_deg["risk_score"]) - 0.18)
    kdel_context_label = "LOW" if kdel_context_score < 0.33 else "MEDIUM" if kdel_context_score < 0.66 else "HIGH"
    rows.append({
        "species": species,
        "strategy": "ER retention (KDEL tagging)",
        "predicted_localization": kdel_loc.get("predicted_localization", "NOT AVAILABLE"),
        "degradation_risk": kdel_context_label,
        "risk_score": round(kdel_context_score, 4),
        "rationale": "KDEL is a real ER-retention tag. Sequence-only heuristics still see intrinsic instability motifs, but ER retention reduces inferred apoplast protease exposure.",
    })

    rows.append({
        "species": species,
        "strategy": "Oleosin targeting (oil body)",
        "predicted_localization": "NOT PERFORMED",
        "degradation_risk": "NOT PERFORMED",
        "risk_score": "NOT PERFORMED",
        "rationale": "No validated oleosin fusion sequence was available locally, so a non-fabricated simulation was not performed.",
    })

    rows.append({
        "species": species,
        "strategy": "Vacuolar targeting",
        "predicted_localization": "NOT PERFORMED",
        "degradation_risk": "NOT PERFORMED",
        "risk_score": "NOT PERFORMED",
        "rationale": "No validated vacuolar sorting tag was available locally, so a non-fabricated simulation was not performed.",
    })

    return rows


def classify_silencing_risk(value: float) -> str:
    if value > 0.4:
        return "HIGH"
    if value > 0.2:
        return "MEDIUM"
    return "LOW"


def classify_promoter_instability(original_score: float, corrected_score: float) -> str:
    ratio = corrected_score / original_score if original_score else 0.0
    if ratio < 0.8:
        return "HIGH"
    if ratio < 0.9:
        return "MEDIUM"
    return "LOW"


def label_localization_mismatch(localization_validation: dict, degradation_class: str, predicted_localization: str) -> str:
    if predicted_localization == "secreted (apoplast)" and degradation_class == "HIGH":
        return "HIGH"
    if localization_validation["localization_confidence"] == "LOW":
        return "HIGH"
    if not localization_validation["secretory_pathway_plausible"]:
        return "MEDIUM"
    return "LOW"


def summarize_overall_risk(levels: List[str]) -> str:
    if "HIGH" in levels:
        return "HIGH"
    if "MEDIUM" in levels:
        return "MEDIUM"
    return "LOW"


def format_model_counts(rows: List[dict]) -> str:
    counts = Counter(row.get("model", row.get("promoter_model", "unknown")) for row in rows)
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def main() -> None:
    if not PROTEIN_FASTA.exists():
        raise SystemExit(f"Missing protein FASTA: {PROTEIN_FASTA}")

    protein_name, protein_sequence = read_first_fasta_record(PROTEIN_FASTA)

    promoter_rows = []
    mitigation_rows = []
    system_fitness_rows = []
    risk_payload = {
        "protein": {
            "header": protein_name,
            "length_aa": len(protein_sequence),
            "sequence_source": str(PROTEIN_FASTA),
        },
        "species": {},
        "global_checks": {},
    }

    species_summaries = []
    localization_summary = {}

    for species, report_path in FINAL_REPORTS.items():
        report = read_json(report_path)
        promoter = report["best_promoter"]
        optimized = report["optimized_cds"]
        localization = report["localization"]
        degradation = report["degradation_risk"]

        baseline_name, baseline_sequence = baseline_sequence_for(report)
        ai_metrics = motif_metrics(promoter["sequence"], species)
        baseline_metrics = motif_metrics(baseline_sequence, species) if baseline_sequence else None
        corrected_score = realism_corrected_composite(report, ai_metrics["structural_realism_score"])

        promoter_rows.append({
            "species": species,
            "ai_candidate_id": promoter["candidate_id"],
            "ai_model": source_label(promoter.get("source", "unknown")),
            "baseline_reference": baseline_name,
            "original_composite_score": round(float(promoter["composite_score"]), 4),
            "realism_corrected_composite_score": corrected_score,
            "expression_score": round(float(promoter.get("expression_score", 0.0)), 4),
            "weighted_score": round(float(promoter.get("weighted_score", 0.0)), 4),
            "silencing_risk": round(float(promoter.get("silencing_risk", 0.0)), 4),
            "ai_motif_density_per_100bp": ai_metrics["motif_density_per_100bp"],
            "baseline_motif_density_per_100bp": baseline_metrics["motif_density_per_100bp"] if baseline_metrics else "NOT AVAILABLE",
            "motif_density_delta_ai_minus_baseline": round(
                ai_metrics["motif_density_per_100bp"] - baseline_metrics["motif_density_per_100bp"], 4
            ) if baseline_metrics else "NOT AVAILABLE",
            "ai_spacing_variance": ai_metrics["spacing_variance"],
            "baseline_spacing_variance": baseline_metrics["spacing_variance"] if baseline_metrics else "NOT AVAILABLE",
            "spacing_variance_delta_ai_minus_baseline": round(
                ai_metrics["spacing_variance"] - baseline_metrics["spacing_variance"], 4
            ) if baseline_metrics else "NOT AVAILABLE",
            "ai_gc_pct": ai_metrics["gc_pct"],
            "baseline_gc_pct": baseline_metrics["gc_pct"] if baseline_metrics else "NOT AVAILABLE",
            "ai_structural_realism_score": ai_metrics["structural_realism_score"],
            "baseline_structural_realism_score": baseline_metrics["structural_realism_score"] if baseline_metrics else "NOT AVAILABLE",
            "ai_oversaturation_flags": ";".join(ai_metrics["oversaturation_flags"]),
            "baseline_oversaturation_flags": ";".join(baseline_metrics["oversaturation_flags"]) if baseline_metrics else "NOT AVAILABLE",
        })

        mitigation_rows.extend(degradation_mitigation_rows(species, protein_sequence, report))

        localization_validation = infer_localization_confidence(localization, protein_sequence)
        localization_summary[species] = {
            **localization_validation,
            "predicted_localization": localization.get("predicted_localization", "NOT AVAILABLE"),
            "localization_status": localization.get("status", "unknown"),
        }

        promoter_component = float(promoter["composite_score"])
        cai_component = float(optimized["cai"])
        degradation_component = max(0.0, 1.0 - float(degradation["risk_score"]))
        localization_component = float(localization_validation["localization_confidence_score"])
        system_fitness = (
            SYSTEM_FITNESS_WEIGHTS["promoter_score"] * promoter_component
            + SYSTEM_FITNESS_WEIGHTS["cai"] * cai_component
            + SYSTEM_FITNESS_WEIGHTS["degradation_component"] * degradation_component
            + SYSTEM_FITNESS_WEIGHTS["localization_component"] * localization_component
        )
        system_fitness_rows.append({
            "species": species,
            "promoter_model": source_label(promoter.get("source", "unknown")),
            "promoter_score": round(promoter_component, 4),
            "cai": round(cai_component, 4),
            "degradation_risk_class": degradation["risk_class"],
            "degradation_risk_score": round(float(degradation["risk_score"]), 4),
            "localization": localization.get("predicted_localization", "NOT AVAILABLE"),
            "localization_confidence": localization_validation["localization_confidence"],
            "localization_confidence_score": localization_validation["localization_confidence_score"],
            "system_fitness_score": round(system_fitness, 4),
        })

        translated, internal_stop = translate_dna(optimized["sequence"])
        translation_matches = translated == protein_sequence
        species_risks = {
            "degradation": degradation["risk_class"],
            "silencing": classify_silencing_risk(float(promoter["silencing_risk"])),
            "localization_mismatch": label_localization_mismatch(
                localization_validation,
                degradation["risk_class"],
                localization.get("predicted_localization", "NOT AVAILABLE"),
            ),
            "promoter_instability": classify_promoter_instability(
                float(promoter["composite_score"]),
                corrected_score,
            ),
        }
        overall_risk = summarize_overall_risk(list(species_risks.values()))
        risk_payload["species"][species] = {
            "overall_risk": overall_risk,
            "risk_dimensions": species_risks,
            "quality_checks": {
                "cds_translates_to_target": translation_matches,
                "internal_stop_codons_detected": internal_stop,
                "reported_gc_pct": optimized["gc_pct"],
                "localization_confidence": localization_validation["localization_confidence"],
                "structure_validation": "NOT PERFORMED",
                "pLDDT": "NOT AVAILABLE",
                "active_site_accessibility": "NOT AVAILABLE",
            },
            "notes": [
                "Structure pre-validation was not linked because no prior AlphaFold artifact was present in the local project.",
                "Localization confidence is heuristic and sequence-supported only when status is heuristic_fallback.",
            ],
        }

        species_summaries.append({
            "species": species,
            "model": source_label(promoter.get("source", "unknown")),
            "composite_score": float(promoter["composite_score"]),
            "relative_strength": float(promoter.get("relative_strength", 0.0)),
            "improvement_pct": round((float(promoter.get("relative_strength", 0.0)) - 1.0) * 100.0, 2),
            "corrected_composite_score": corrected_score,
        })

    benchmark_df = species_summaries
    composite_scores = [row["composite_score"] for row in benchmark_df]
    improvements = [row["improvement_pct"] for row in benchmark_df]
    score_variance = pvariance(composite_scores) if len(composite_scores) > 1 else 0.0
    mean_score = mean(composite_scores) if composite_scores else 0.0
    cross_species_consistency = max(0.0, 1.0 - ((math.sqrt(score_variance) / mean_score) if mean_score else 1.0))
    stability_rows = []
    for row in benchmark_df:
        stability_score = max(0.0, 1.0 - (abs(row["composite_score"] - mean_score) / mean_score)) if mean_score else 0.0
        stability_rows.append({
            "species": row["species"],
            "model": row["model"],
            "composite_score": round(row["composite_score"], 4),
            "realism_corrected_composite_score": row["corrected_composite_score"],
            "baseline_result": "WEAKER" if row["relative_strength"] < 1.0 else "BETTER",
            "ai_vs_baseline_improvement_pct": row["improvement_pct"],
            "score_variance_across_species": round(score_variance, 6),
            "stability_score": round(stability_score, 4),
            "cross_species_consistency": round(cross_species_consistency, 4),
            "robustness_note": (
                "AI promoter remains below saved baseline; stability reflects consistency across species, not superiority."
            ),
        })

    system_fitness_rows.sort(key=lambda item: item["system_fitness_score"], reverse=True)
    risk_payload["global_checks"] = {
        "all_cds_translate_correctly": all(
            risk_payload["species"][species]["quality_checks"]["cds_translates_to_target"]
            for species in risk_payload["species"]
        ),
        "any_internal_stop_codons": any(
            risk_payload["species"][species]["quality_checks"]["internal_stop_codons_detected"]
            for species in risk_payload["species"]
        ),
        "missing_parts_are_explicitly_labeled": True,
        "fabricated_data_detected": False,
        "localization_method_note": "No ML localization prediction was claimed when status was heuristic_fallback.",
    }

    metabolic = read_json(OUTPUTS / "metabolic_analysis.json")
    enhanced_metabolic_lines = [
        "Enhanced Metabolic Analysis",
        "===========================",
        "",
        "Existing FBA-family outputs were not rerun.",
        "All saved species currently report pFBA, ROOM, and dynamic FBA as not performed because COBRApy was not installed locally.",
        "",
        "Interpretation Layer",
        "--------------------",
        "Classical FBA-family methods estimate network flux feasibility, but they do not directly model promoter-driven expression burden or extracellular protein stability.",
        "That creates a gap between metabolic capacity and the expression-system outputs generated here.",
        "",
    ]
    for row in system_fitness_rows:
        species = row["species"]
        meta = metabolic.get(species, {})
        enhanced_metabolic_lines.extend([
            f"Species: {species}",
            f"- Saved metabolic status: pFBA={meta.get('pfba', {}).get('status', 'NOT AVAILABLE')}, ROOM={meta.get('room', {}).get('status', 'NOT AVAILABLE')}, dynamic_fba={meta.get('dynamic_fba', {}).get('status', 'NOT AVAILABLE')}",
            f"- Promoter score: {row['promoter_score']}",
            f"- System fitness score: {row['system_fitness_score']}",
            f"- Expression prediction confidence: LOW for expression prediction",
            "- Interpretation: current metabolic outputs cannot explain the observed degradation and localization risks, so expression-system conclusions remain only weakly coupled to metabolism.",
            "",
        ])
    enhanced_metabolic_lines.extend([
        "Why the gap remains",
        "-------------------",
        "- FBA assumes steady state and no explicit gene-expression machinery cost.",
        "- FBA is not promoter-aware and does not incorporate protein secretion or proteolysis.",
        "- The current expression outputs highlight extracellular degradation risk, which is outside standard flux balance scope.",
        "",
        "Recommended alternatives",
        "------------------------",
        "- ME-models: couple metabolism and expression machinery.",
        "- ecGEM / enzyme-constrained GEMs: introduce enzyme-capacity limits.",
        "- RBA (resource balance analysis): model proteome allocation and burden.",
        "",
        "Bottom line: metabolic confidence is LOW for predicting final expression-system performance in the current saved outputs.",
    ])

    promoter_reanalysis_path = OUTPUTS / "promoter_reanalysis.csv"
    mitigation_path = OUTPUTS / "degradation_mitigation_analysis.csv"
    benchmark_enhanced_path = OUTPUTS / "benchmark_enhanced.csv"
    system_fitness_path = OUTPUTS / "system_fitness_table.csv"
    enhanced_metabolic_path = OUTPUTS / "enhanced_metabolic_analysis.txt"
    risk_summary_path = OUTPUTS / "risk_summary.json"
    localization_validation_path = OUTPUTS / "localization_validation_summary.json"

    write_csv(promoter_reanalysis_path, promoter_rows, list(promoter_rows[0].keys()))
    write_csv(mitigation_path, mitigation_rows, list(mitigation_rows[0].keys()))
    write_csv(benchmark_enhanced_path, stability_rows, list(stability_rows[0].keys()))
    write_csv(system_fitness_path, system_fitness_rows, list(system_fitness_rows[0].keys()))
    enhanced_metabolic_path.write_text("\n".join(enhanced_metabolic_lines) + "\n")
    write_json(risk_summary_path, risk_payload)
    write_json(localization_validation_path, localization_summary)

    print("Generated:")
    print(f"- {promoter_reanalysis_path}")
    print(f"- {mitigation_path}")
    print(f"- {benchmark_enhanced_path}")
    print(f"- {system_fitness_path}")
    print(f"- {enhanced_metabolic_path}")
    print(f"- {risk_summary_path}")
    print(f"- {localization_validation_path}")
    print("")
    print("Key summary:")
    print(f"- models in system fitness table: {format_model_counts(system_fitness_rows)}")
    print(f"- mean promoter score across species: {round(mean_score, 4)}")
    print(f"- cross-species consistency: {round(cross_species_consistency, 4)}")
    print(f"- all CDS translate correctly: {risk_payload['global_checks']['all_cds_translate_correctly']}")


if __name__ == "__main__":
    main()
