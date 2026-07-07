#!/usr/bin/env python3
"""Generate a decision-to-action strategy map from saved post hoc outputs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUTS = ROOT / "outputs"


def read_csv_rows(path: Path) -> List[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    decision_rows = read_csv_rows(OUTPUTS / "final_decision_matrix.csv")
    protease_rows = read_csv_rows(OUTPUTS / "protease_exposure_analysis.csv")
    localization_rows = read_csv_rows(OUTPUTS / "localization_refined.csv")
    mitigation_rows = read_csv_rows(OUTPUTS / "enhanced_degradation_mitigation.csv")

    species = [row["species"] for row in decision_rows]
    localization_class = sorted({row["localization_class"] for row in localization_rows})
    dominant_proteases = sorted({row["dominant_protease_class"] for row in protease_rows})
    avg_exposure = mean(float(row["total_protease_exposure_score"]) for row in protease_rows)
    avg_fitness = mean(float(row["system_fitness_score"]) for row in decision_rows)

    root_cause_lines = [
        "Root Cause Summary",
        "==================",
        "",
        "Current deployability outcome",
        "-----------------------------",
        f"- Saved species set: {', '.join(species)}",
        f"- Current deployability call across all saved systems: {', '.join(sorted(set(row['deployability'] for row in decision_rows)))}",
        "",
        "Primary failure modes",
        "---------------------",
        f"- Localization issue: {', '.join(localization_class)}",
        "- Interpretation: the saved protein shows signal-peptide support but also multiple transmembrane-like regions, so a clean soluble secretory route is not defensible from the current evidence.",
        "- Degradation issue: high intrinsic instability remains a concern in the saved reports, even after refining exposure into a mechanistic protease model.",
        f"- Dominant protease pressure: {', '.join(dominant_proteases)}",
        f"- Mean protease exposure score across species: {avg_exposure:.2f}",
        "",
        "Take-home message",
        "-----------------",
        "- Promoter strength and codon optimisation are not the leading blockers.",
        "- The system currently fails because protein routing and downstream protease exposure dominate the outcome.",
    ]

    host_engineering_lines = [
        "Host Engineering Strategy",
        "=========================",
        "",
        "Problem mapping",
        "---------------",
        "- Existing Iteration 1 logic already pointed to subtilase mitigation through NbSBT1/2-style intervention.",
        "- The current refined analysis extends that need to the A1 aspartic-protease class, which is the dominant exposure signal in the saved system outputs.",
        "",
        "Recommended intervention classes",
        "--------------------------------",
        "- Maintain NbSBT1/NbSBT2-style subtilase suppression as a supporting intervention, because SBT exposure remains non-trivial.",
        "- Add A1 aspartic-protease targeting as the primary new host-engineering layer.",
        "- Keep C1A cysteine-protease reduction in scope as a secondary protection layer.",
        "",
        "Candidate target classes",
        "------------------------",
        "- A1 class: secreted or vacuolar aspartic proteases likely to contribute to extracellular or post-secretory degradation pressure.",
        "- SBT class: subtilase-family serine proteases, including the previously identified NbSBT1/2-style targets.",
        "- C1A class: papain-like cysteine proteases that could contribute residual proteolysis if secretion still occurs.",
        "",
        "Why this is justified",
        "---------------------",
        "- The saved protease exposure analysis ranks A1 highest, followed by C1A and SBT.",
        "- ER-retention mitigation lowers exposure but does not remove intrinsic instability, so host engineering remains necessary.",
        "- No specific knockout construct is proposed here because none is currently validated in the saved outputs.",
    ]

    strategy_rows = [
        {
            "strategy": "ER retention (KDEL)",
            "biological_rationale": "Retains the protein in the ER and reduces direct exposure to apoplast proteases.",
            "expected_degradation_reduction": "Supported by current post hoc model; exposure score falls from 0.65 to 0.2925",
            "feasibility": "HIGH",
        },
        {
            "strategy": "Vacuolar targeting",
            "biological_rationale": "Could move the protein away from the apoplast, but vacuoles also contain proteases and no validated local targeting design is saved.",
            "expected_degradation_reduction": "NOT QUANTIFIED",
            "feasibility": "MEDIUM",
        },
        {
            "strategy": "Oil body targeting",
            "biological_rationale": "Compartmental sequestration may reduce extracellular exposure, but requires a validated fusion architecture not present in the saved outputs.",
            "expected_degradation_reduction": "NOT QUANTIFIED",
            "feasibility": "LOW to MEDIUM",
        },
    ]

    upgrade_path_lines = [
        "System Upgrade Path",
        "===================",
        "",
        "Current state",
        "-------------",
        "- Current system status: NOT deployable",
        f"- Mean system fitness score across saved species: {avg_fitness:.4f}",
        "",
        "Required upgrades",
        "-----------------",
        "1. Localization correction",
        "   - Resolve the current ambiguous secretory routing before any deployment decision.",
        "   - ER retention is the most evidence-backed first intervention from the current saved analyses.",
        "",
        "2. Protease-environment control",
        "   - Add host engineering against A1-class proteases as the primary intervention.",
        "   - Retain NbSBT1/2-style mitigation and broaden to C1A where needed.",
        "",
        "3. Compartment targeting",
        "   - Prioritize ER retention first because it has direct support from the saved mitigation model.",
        "   - Consider vacuolar or oil-body strategies only after validated construct designs exist.",
        "",
        "4. Re-evaluation checkpoint",
        "   - Reassess deployability only after host-engineering and routing changes are incorporated and experimentally checked.",
    ]

    er_rows = [row for row in mitigation_rows if row["strategy"] == "ER retention (KDEL)"]
    er_reduction_supported = all(float(row["exposure_score"]) < 0.34 for row in er_rows)
    conditional_lines = [
        "Conditional Deployability",
        "=========================",
        "",
        "Current decision",
        "----------------",
        "- Present saved systems remain NON-DEPLOYABLE.",
        "",
        "Conditional model",
        "-----------------",
        "- If A1-dominant protease pressure is reduced through host engineering, and",
        "- if ER retention is applied to lower extracellular exposure,",
        "- then expected deployability can move from NO to CONDITIONAL.",
        "",
        "Why only CONDITIONAL",
        "--------------------",
        f"- ER-retention support in current model: {'YES' if er_reduction_supported else 'PARTIAL'}",
        "- Intrinsic protein instability is not fully removed by ER retention alone.",
        "- Localization ambiguity is reduced by the strategy, but not resolved by current saved evidence without an updated construct and validation.",
        "- Therefore a full YES decision is not justified from the existing outputs.",
    ]

    root_cause_path = OUTPUTS / "root_cause_summary.txt"
    host_strategy_path = OUTPUTS / "host_engineering_strategy.txt"
    targeting_matrix_path = OUTPUTS / "targeting_strategy_matrix.csv"
    upgrade_path_path = OUTPUTS / "system_upgrade_path.txt"
    conditional_path = OUTPUTS / "conditional_deployability.txt"

    root_cause_path.write_text("\n".join(root_cause_lines) + "\n")
    host_strategy_path.write_text("\n".join(host_engineering_lines) + "\n")
    write_csv(targeting_matrix_path, strategy_rows, list(strategy_rows[0].keys()))
    upgrade_path_path.write_text("\n".join(upgrade_path_lines) + "\n")
    conditional_path.write_text("\n".join(conditional_lines) + "\n")

    print("Generated:")
    print(f"- {root_cause_path}")
    print(f"- {host_strategy_path}")
    print(f"- {targeting_matrix_path}")
    print(f"- {upgrade_path_path}")
    print(f"- {conditional_path}")


if __name__ == "__main__":
    main()
