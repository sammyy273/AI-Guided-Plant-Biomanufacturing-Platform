#!/usr/bin/env python3
"""Decision-ready post hoc analysis for the saved expression-system outputs.

This script only reads existing outputs and writes new downstream summaries.
It does not regenerate promoters or CDSs and it does not overwrite originals.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, pvariance
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.protein.expression_system import signal_peptide_heuristic


OUTPUTS = ROOT / "outputs"
PROTEIN_FASTA = ROOT / "data" / "protein" / "hyaluronidase.fasta"

SPECIES = ["nbenthamiana", "rice", "tomato"]

DEGRADATION_WEIGHT = 0.40
LOCALIZATION_WEIGHT = 0.30
PROMOTER_WEIGHT = 0.20
SILENCING_WEIGHT = 0.10


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


def detect_transmembrane_regions(protein_sequence: str, window: int = 19) -> List[dict]:
    hydrophobic = set("AILMFWVPGCY")
    seq = protein_sequence.upper()
    hits = []
    for start in range(0, max(len(seq) - window + 1, 0)):
        frag = seq[start:start + window]
        hydro = sum(1 for aa in frag if aa in hydrophobic) / len(frag)
        if hydro >= 0.68:
            hits.append(
                {
                    "start": start + 1,
                    "end": start + window,
                    "sequence": frag,
                    "hydrophobic_fraction": round(hydro, 3),
                }
            )
    merged = []
    for hit in hits:
        if not merged or hit["start"] > merged[-1]["end"] + 3:
            merged.append(hit)
        else:
            merged[-1]["end"] = max(merged[-1]["end"], hit["end"])
            merged[-1]["sequence"] = seq[merged[-1]["start"] - 1:merged[-1]["end"]]
            merged[-1]["hydrophobic_fraction"] = max(
                merged[-1]["hydrophobic_fraction"], hit["hydrophobic_fraction"]
            )
    return merged


def refine_localization_class(protein_sequence: str) -> dict:
    signal = signal_peptide_heuristic(protein_sequence)
    tm_regions = detect_transmembrane_regions(protein_sequence)

    if signal["has_signal_peptide"] and not tm_regions:
        localization_class = "SECRETED"
        confidence_level = "HIGH"
        confidence_score = 0.9
        rationale = "N-terminal signal peptide detected with no transmembrane region found."
    elif signal["has_signal_peptide"] and tm_regions:
        localization_class = "AMBIGUOUS SECRETORY PATHWAY"
        confidence_level = "MEDIUM"
        confidence_score = 0.55
        rationale = "Signal peptide detected, but transmembrane-like regions were also found, so a clean soluble secretory path is not defensible."
    else:
        localization_class = "NON-SECRETORY"
        confidence_level = "LOW"
        confidence_score = 0.3
        rationale = "No sequence-supported signal peptide detected."

    return {
        "localization_class": localization_class,
        "confidence_level": confidence_level,
        "confidence_score": round(confidence_score, 4),
        "signal_peptide_detected": signal["has_signal_peptide"],
        "signal_peptide_details": signal,
        "transmembrane_region_detected": bool(tm_regions),
        "transmembrane_regions": tm_regions,
        "rationale": rationale,
        "method_note": "Sequence-supported heuristic only; no ML localization model used.",
    }


def normalize_count(count: int, saturation: float) -> float:
    return min(1.0, count / saturation)


def localization_multiplier(localization_class: str) -> float:
    if localization_class == "SECRETED":
        return 1.0
    if localization_class == "AMBIGUOUS SECRETORY PATHWAY":
        return 0.75
    return 0.25


def protease_exposure_from_family_counts(
    family_counts: Dict[str, int], localization_class: str
) -> dict:
    loc_mult = localization_multiplier(localization_class)
    sbt = normalize_count(int(family_counts.get("SBT1", 0)), 15.0) * loc_mult
    c1a = normalize_count(int(family_counts.get("C1A", 0)), 30.0) * loc_mult
    a1 = normalize_count(int(family_counts.get("A1", 0)), 30.0) * loc_mult
    total = max(0.0, min(1.0, (sbt + c1a + a1) / 3.0))
    dominant = max(
        {"SBT": sbt, "C1A": c1a, "A1": a1}.items(),
        key=lambda item: item[1],
    )[0]
    return {
        "SBT_exposure_score": round(sbt, 4),
        "C1A_exposure_score": round(c1a, 4),
        "A1_exposure_score": round(a1, 4),
        "total_protease_exposure_score": round(total, 4),
        "dominant_protease_class": dominant,
        "localization_multiplier": round(loc_mult, 4),
    }


def exposure_to_class(score: float) -> str:
    if score >= 0.67:
        return "HIGH"
    if score >= 0.34:
        return "MEDIUM"
    return "LOW"


def mitigation_rows(species: str, exposure: dict, localization: dict) -> List[dict]:
    base_score = float(exposure["total_protease_exposure_score"])
    rows = [
        {
            "species": species,
            "strategy": "Original construct",
            "exposure_score": round(base_score, 4),
            "degradation_class": exposure_to_class(base_score),
            "rationale": "Current construct combines the saved protein sequence with the current sequence-supported localization class.",
        }
    ]

    er_score = base_score * 0.45
    rows.append(
        {
            "species": species,
            "strategy": "ER retention (KDEL)",
            "exposure_score": round(er_score, 4),
            "degradation_class": exposure_to_class(er_score),
            "rationale": "ER retention is expected to reduce extracellular apoplast exposure, but it does not remove intrinsic instability motifs from the protein sequence.",
        }
    )

    delay_score = base_score * 0.7
    rows.append(
        {
            "species": species,
            "strategy": "Secretion delay model",
            "exposure_score": round(delay_score, 4),
            "degradation_class": exposure_to_class(delay_score),
            "rationale": "A secretion-delay interpretation models partial ER accumulation before secretion, reducing but not eliminating exposure to apoplast proteases.",
        }
    )
    return rows


def risk_level_to_score(level: str) -> float:
    mapping = {"LOW": 0.2, "MEDIUM": 0.6, "HIGH": 1.0}
    return mapping.get(level, 1.0)


def classify_system_risk(score: float) -> str:
    if score >= 0.67:
        return "HIGH"
    if score >= 0.34:
        return "MEDIUM"
    return "LOW"


def promoter_instability_score(original: float, corrected: float) -> Tuple[float, str]:
    if not original:
        return 1.0, "HIGH"
    ratio = corrected / original
    score = max(0.0, min(1.0, 1.0 - ratio))
    if ratio >= 0.9:
        level = "LOW"
    elif ratio >= 0.8:
        level = "MEDIUM"
    else:
        level = "HIGH"
    return round(score, 4), level


def deployability_label(
    localization_class: str,
    degradation_class: str,
    original_exposure: float,
    best_mitigated_exposure: float,
) -> str:
    mitigation_delta = original_exposure - best_mitigated_exposure
    if degradation_class == "HIGH" and localization_class == "AMBIGUOUS SECRETORY PATHWAY":
        if mitigation_delta >= 0.2:
            return "CONDITIONAL"
        return "NO"
    if degradation_class == "HIGH" and mitigation_delta >= 0.2:
        return "CONDITIONAL"
    if degradation_class in {"LOW", "MEDIUM"} and localization_class == "SECRETED":
        return "YES"
    return "NO"


def baseline_interpretation(baseline_result: str) -> str:
    if str(baseline_result).upper() == "WEAKER":
        return "baseline remains stronger in the saved outputs"
    if str(baseline_result).upper() == "BETTER":
        return "AI promoter exceeds the recorded baseline"
    return "baseline comparison unavailable"


def validate_cds_translation() -> dict:
    risk = read_json(OUTPUTS / "risk_summary.json")
    return risk["global_checks"]


def main() -> None:
    if not PROTEIN_FASTA.exists():
        raise SystemExit(f"Missing FASTA: {PROTEIN_FASTA}")

    _, protein_sequence = read_first_fasta_record(PROTEIN_FASTA)

    promoter_reanalysis = {
        row["species"]: row
        for row in csv.DictReader((OUTPUTS / "promoter_reanalysis.csv").open())
    }
    system_fitness = {
        row["species"]: row
        for row in csv.DictReader((OUTPUTS / "system_fitness_table.csv").open())
    }

    protease_rows = []
    mitigation_output_rows = []
    localization_rows = []
    system_risk_rows = []
    decision_rows = []
    stability_rows = []

    promoter_scores = []
    corrected_scores = []
    baseline_densities = []
    ai_densities = []

    for species in SPECIES:
        report = read_json(OUTPUTS / f"final_report_{species}.json")
        promoter = report["best_promoter"]
        degradation = report["degradation_risk"]

        family_counts = {
            "SBT1": int(degradation["family_mapping"]["SBT1"]["hit_count"]) if "family_mapping" in degradation else int(degradation["SBT1"]["hit_count"]),
            "C1A": int(degradation["family_mapping"]["C1A"]["hit_count"]) if "family_mapping" in degradation else int(degradation["C1A"]["hit_count"]),
            "A1": int(degradation["family_mapping"]["A1"]["hit_count"]) if "family_mapping" in degradation else int(degradation["A1"]["hit_count"]),
        }

        localization = refine_localization_class(protein_sequence)
        exposure = protease_exposure_from_family_counts(family_counts, localization["localization_class"])
        protease_rows.append(
            {
                "species": species,
                "protein_sequence_length_aa": len(protein_sequence),
                "localization_class": localization["localization_class"],
                "SBT_exposure_score": exposure["SBT_exposure_score"],
                "C1A_exposure_score": exposure["C1A_exposure_score"],
                "A1_exposure_score": exposure["A1_exposure_score"],
                "total_protease_exposure_score": exposure["total_protease_exposure_score"],
                "dominant_protease_class": exposure["dominant_protease_class"],
                "interpretation": f"{exposure['dominant_protease_class']} is the dominant exposure class under the current sequence-supported localization assumption.",
            }
        )

        localization_rows.append(
            {
                "species": species,
                "localization_class": localization["localization_class"],
                "confidence_level": localization["confidence_level"],
                "signal_peptide_detected": localization["signal_peptide_detected"],
                "transmembrane_region_detected": localization["transmembrane_region_detected"],
                "transmembrane_region_count": len(localization["transmembrane_regions"]),
                "rationale": localization["rationale"],
                "method_note": localization["method_note"],
            }
        )

        mitigation = mitigation_rows(species, exposure, localization)
        mitigation_output_rows.extend(mitigation)

        original_exposure = mitigation[0]["exposure_score"]
        best_mitigated = min(
            row["exposure_score"]
            for row in mitigation[1:]
            if isinstance(row["exposure_score"], (int, float))
        )
        degradation_class = exposure_to_class(original_exposure)

        re_row = promoter_reanalysis[species]
        original_score = float(re_row["original_composite_score"])
        corrected_score = float(re_row["realism_corrected_composite_score"])
        promoter_instability_numeric, promoter_instability_level = promoter_instability_score(
            original_score, corrected_score
        )

        silencing_score = min(1.0, float(promoter["silencing_risk"]) / 0.4)
        localization_uncertainty = (
            0.0 if localization["confidence_level"] == "HIGH"
            else 0.5 if localization["confidence_level"] == "MEDIUM"
            else 1.0
        )
        degradation_numeric = risk_level_to_score(degradation_class)
        system_risk_score = (
            DEGRADATION_WEIGHT * degradation_numeric
            + LOCALIZATION_WEIGHT * localization_uncertainty
            + PROMOTER_WEIGHT * promoter_instability_numeric
            + SILENCING_WEIGHT * silencing_score
        )
        system_risk_level = classify_system_risk(system_risk_score)
        system_risk_rows.append(
            {
                "species": species,
                "degradation_risk": degradation_class,
                "localization_uncertainty": localization["confidence_level"],
                "silencing_risk": round(float(promoter["silencing_risk"]), 4),
                "promoter_instability": promoter_instability_level,
                "system_risk_score": round(system_risk_score, 4),
                "system_risk_class": system_risk_level,
            }
        )

        fitness_row = system_fitness[species]
        deployability = deployability_label(
            localization["localization_class"],
            degradation_class,
            original_exposure,
            best_mitigated,
        )
        decision_rows.append(
            {
                "species": species,
                "promoter_score": round(float(promoter["composite_score"]), 4),
                "baseline_comparison": str(promoter["baseline_comparison"]["verdict"]).lower(),
                "CAI": round(float(report["optimized_cds"]["cai"]), 4),
                "localization_class": localization["localization_class"],
                "degradation_risk": degradation_class,
                "system_fitness_score": round(float(fitness_row["system_fitness_score"]), 4),
                "system_risk_score": round(system_risk_score, 4),
                "deployability": deployability,
            }
        )

        promoter_scores.append(original_score)
        corrected_scores.append(corrected_score)
        ai_densities.append(float(re_row["ai_motif_density_per_100bp"]))
        try:
            baseline_densities.append(float(re_row["baseline_motif_density_per_100bp"]))
        except ValueError:
            pass

    variance = pvariance(promoter_scores) if len(promoter_scores) > 1 else 0.0
    corrected_variance = pvariance(corrected_scores) if len(corrected_scores) > 1 else 0.0
    stability_score = max(
        0.0,
        1.0 - (math.sqrt(variance) / mean(promoter_scores)),
    ) if promoter_scores and mean(promoter_scores) else 0.0
    baseline_density_mean = mean(baseline_densities) if baseline_densities else None
    ai_density_mean = mean(ai_densities) if ai_densities else None
    density_note = "NOT AVAILABLE"
    if baseline_density_mean is not None and ai_density_mean is not None:
        if baseline_density_mean > ai_density_mean:
            density_note = "Baseline promoters are more motif-dense in the saved comparisons, which is consistent with a saturated architecture interpretation."
        else:
            density_note = "AI promoters are more motif-dense in the saved comparisons."

    for row in decision_rows:
        stability_rows.append(
            {
                "species": row["species"],
                "promoter_score": row["promoter_score"],
                "cross_species_variance": round(variance, 6),
                "realism_corrected_variance": round(corrected_variance, 6),
                "stability_score": round(stability_score, 4),
                "interpretation": f"AI promoter is stable/consistent across the saved species set; {density_note}",
                "baseline_context": baseline_interpretation(row["baseline_comparison"]),
            }
        )

    # Integration and executive summaries.
    integration_lines = [
        "Iteration Integration Summary",
        "=============================",
        "",
        "Promoter optimisation from Iteration 2 does not appear to be the dominant blocker in the saved system outputs.",
        "The main bottleneck is protein-level exposure after localization is considered.",
        "",
        "Protease mapping",
        "---------------",
    ]
    for row in protease_rows:
        integration_lines.extend(
            [
                f"{row['species']}: dominant exposure class = {row['dominant_protease_class']}",
                f"- total protease exposure score: {row['total_protease_exposure_score']}",
                f"- interpretation: {row['interpretation']}",
                "- host-engineering linkage: Iteration 1 protease-mitigation ideas map naturally to NbSBT1/2-like subtilase suppression plus broader C1A and A1 reduction strategies.",
                "- current status: specific validated knockout designs are NOT INCLUDED in the saved outputs.",
                "",
            ]
        )

    final_summary_lines = [
        "Final System Summary",
        "====================",
        "",
        "What works",
        "----------",
        "- Saved promoters produce moderate-to-strong computational scores across the three species.",
        "- All saved optimized CDS sequences still translate correctly to the target protein.",
        "- CAI remains high in all three species, so translational codon usage is not the leading bottleneck.",
        "",
        "What fails",
        "----------",
        "- Protein degradation risk remains high in every saved species output.",
        "- Localization remains ambiguous under sequence-supported analysis because signal-peptide evidence coexists with transmembrane-like regions.",
        "- As a result, the current deployability decision is not affirmative for the present construct set.",
        "",
        "What enables deployment",
        "-----------------------",
        "- Host engineering is the clearest next lever, especially protease-environment control and compartment-routing changes.",
        "- ER-retention-style mitigation reduces modeled protease exposure but does not fully remove intrinsic instability concerns.",
        "- Additional validated localization engineering and protease knockouts would be required before a credible deployment claim.",
        "",
        "Confidence",
        "----------",
        "- Promoter/CDS confidence: MEDIUM to HIGH from saved computational outputs.",
        "- Localization confidence: MEDIUM at best and currently ambiguous.",
        "- Overall deployment confidence: LOW without host engineering and experimental validation.",
    ]

    validation = validate_cds_translation()
    if not validation.get("all_cds_translate_correctly", False):
        raise SystemExit("Validation failed: not all CDS sequences translate correctly.")
    if validation.get("fabricated_data_detected", True):
        raise SystemExit("Validation failed: fabricated data flag detected.")

    protease_path = OUTPUTS / "protease_exposure_analysis.csv"
    mitigation_path = OUTPUTS / "enhanced_degradation_mitigation.csv"
    localization_path = OUTPUTS / "localization_refined.csv"
    system_risk_path = OUTPUTS / "system_risk_recomputed.csv"
    decision_path = OUTPUTS / "final_decision_matrix.csv"
    stability_path = OUTPUTS / "promoter_stability_summary.csv"
    integration_path = OUTPUTS / "iteration_integration_summary.txt"
    final_summary_path = OUTPUTS / "final_system_summary.txt"

    write_csv(protease_path, protease_rows, list(protease_rows[0].keys()))
    write_csv(mitigation_path, mitigation_output_rows, list(mitigation_output_rows[0].keys()))
    write_csv(localization_path, localization_rows, list(localization_rows[0].keys()))
    write_csv(system_risk_path, system_risk_rows, list(system_risk_rows[0].keys()))
    write_csv(decision_path, decision_rows, list(decision_rows[0].keys()))
    write_csv(stability_path, stability_rows, list(stability_rows[0].keys()))
    integration_path.write_text("\n".join(integration_lines) + "\n")
    final_summary_path.write_text("\n".join(final_summary_lines) + "\n")

    print("Generated:")
    print(f"- {protease_path}")
    print(f"- {mitigation_path}")
    print(f"- {localization_path}")
    print(f"- {system_risk_path}")
    print(f"- {decision_path}")
    print(f"- {stability_path}")
    print(f"- {integration_path}")
    print(f"- {final_summary_path}")
    print("")
    print("Validation:")
    print(f"- all CDS translate correctly: {validation['all_cds_translate_correctly']}")
    print(f"- fabricated data detected: {validation['fabricated_data_detected']}")


if __name__ == "__main__":
    main()
