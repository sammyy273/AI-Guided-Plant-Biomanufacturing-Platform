#!/usr/bin/env python3
"""Generate wording-corrected documentation variants and a corrections log.

This script does not modify source data or pipeline outputs. It creates
additive, wording-only corrected documents that standardize claims against the
current saved evidence.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"


def read_csv_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def read_text(path: Path) -> str:
    return path.read_text()


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n")


def main() -> None:
    final_iter2_report = OUTPUTS / "FINAL_ITERATION2_REPORT.md"
    final_decision_statement = OUTPUTS / "final_decision_statement.txt"
    decision_confidence = OUTPUTS / "decision_confidence.txt"
    final_system_summary = OUTPUTS / "final_system_summary.txt"
    promoter_stability = read_csv_rows(OUTPUTS / "promoter_stability_summary.csv")
    decision_rows = read_csv_rows(OUTPUTS / "final_decision_matrix.csv")
    traceability = json.loads((OUTPUTS / "evidence_traceability.json").read_text())
    risk_summary = json.loads((OUTPUTS / "risk_summary.json").read_text())

    avg_promoter_score = mean(float(row["promoter_score"]) for row in decision_rows)
    avg_system_risk = mean(float(row["system_risk_score"]) for row in decision_rows)
    avg_stability = mean(float(row["stability_score"]) for row in promoter_stability)

    baseline_clause = (
        "Baseline promoters show higher motif density, while AI promoters prioritize structural realism and reduced motif saturation."
    )
    limitation_clause = (
        "Primary limitation is protein-level constraints (localization ambiguity and protease-mediated degradation)."
    )
    field_note = (
        "Field notes: `expression_score` is a proxy score for expression potential estimation, not measured expression; "
        "`localization_class` is a sequence-based localization estimate (no ML model applied); "
        "`degradation_risk` is model-based and not experimentally measured."
    )

    corrected_report = f"""# Corrected Iteration 2 Report

## AI-Driven Cross-Species Promoter Design and Benchmarking

## Executive Summary
- Nine species-level Iteration 2 runs were aggregated from the latest completed saved output folders; no new generation was performed for this corrected document.
- The highest saved composite score in the latest-run set was tomato `0.8659` with an `evo2`-sourced promoter candidate.
- `mutational` remained the most frequent winning model across the latest saved species runs.
- Cross-species promoter behaviour appears computationally stable in the saved analyses, with a mean stability score of `{avg_stability:.4f}`.
- Baseline promoters show higher motif density, while AI promoters prioritize structural realism and reduced motif saturation.
- Primary limitation is protein-level constraints (localization ambiguity and protease-mediated degradation).

## Methodology
- Promoter candidates were generated previously using Evo2, D3LM, and mutational workflows; this document only reinterprets saved outputs.
- Candidate ranking uses saved composite scores and associated metadata from scored CSVs, loop summaries, and report cards where available.
- Expression-related fields are reported as expression potential estimation (proxy-based), not measured expression.
- Localization-related fields are reported as sequence-based localization estimation (no ML model applied).
- Safe harbor module implemented; not executed due to missing genome data.
- Genome feasibility framework implemented but not applied in final outputs.

## Results Summary
- Mean saved promoter score across the deployability-focused species set (`nbenthamiana`, `rice`, `tomato`) is `{avg_promoter_score:.4f}`.
- Mean recomputed system risk score across that same set is `{avg_system_risk:.4f}`.
- Current system decision across the saved deployability-focused set is consistently `NOT DEPLOYABLE`.
- Decision confidence is `MEDIUM`.

## Benchmark Interpretation
- The saved benchmark remains useful for comparative prioritisation.
- However, benchmark scores should be interpreted as computational ranking signals rather than direct biological performance measurements.
- Baseline promoters show higher motif density, while AI promoters prioritize structural realism and reduced motif saturation.
- Primary limitation is protein-level constraints (localization ambiguity and protease-mediated degradation).

## Assumptions, Limitations, and Scope
- This is a computationally validated pipeline, not an experimentally validated pipeline.
- No experimental validation has been performed.
- Expression is proxy-based and should be described as expression potential estimation (proxy-based).
- Localization is heuristic and sequence-supported; no ML localization model was applied in the current saved outputs.
- Safe harbor module implemented; not executed due to missing genome data.
- Genome feasibility framework implemented but not applied in final outputs.
- Deployment is currently NOT achievable without host engineering.

## Decision Consistency
- Decision: NOT DEPLOYABLE
- Confidence: MEDIUM
- Reason: protein-level constraints dominate
- Path forward: host engineering + targeting (ER retention + A1 reduction)

## Traceability Note
- See `{traceability['global_conclusions'][1]['source_file']}` for `total_protease_exposure_score = {traceability['global_conclusions'][1]['value']}`.
- See `{OUTPUTS / 'localization_refined.csv'}` for `localization_class = AMBIGUOUS SECRETORY PATHWAY`.
- See `{OUTPUTS / 'risk_summary.json'}` for `all_cds_translate_correctly = {risk_summary['global_checks']['all_cds_translate_correctly']}`.

## Header Notes
{field_note}
"""

    corrected_readme = f"""# promoter_design v2_research — Corrected README

This corrected README is a wording-only summary generated from saved outputs. It does not change any underlying metrics, sequences, or decision files.

## Project Status
- Computationally validated pipeline for plant promoter design and comparative in silico screening
- Current system-level deployment decision: `NOT DEPLOYABLE`
- Decision confidence: `MEDIUM`
- Primary limitation is protein-level constraints (localization ambiguity and protease-mediated degradation)

## What The Current Outputs Support
- Promoter optimization and ranking across multiple plant species
- Expression potential estimation (proxy-based)
- Sequence-based localization estimation (no ML model applied)
- Mechanistic, post hoc protease exposure interpretation
- Decision-ready host-engineering and targeting recommendations

## What The Current Outputs Do Not Support
- Experimental validation claims
- Direct expression measurement claims
- Executed safe harbor placement claims
- Applied genome feasibility scoring claims
- Unqualified deployment claims

## Standardized Language
- "predicts expression" -> "estimates expression potential using motif, occupancy, and constraint-based scoring"
- "expression prediction" -> "expression potential estimation (proxy-based)"
- "localization prediction" -> "sequence-based localization estimation (no ML model applied)"
- "validated pipeline" -> "computationally validated pipeline"
- "Safe harbor prediction is implemented" -> "Safe harbor module implemented; not executed due to missing genome data"
- "genome feasibility scoring" -> "genome feasibility framework implemented but not applied in final outputs"

## Core Findings
- Baseline promoters show higher motif density, while AI promoters prioritize structural realism and reduced motif saturation.
- Promoter is not the main bottleneck in the deployability analysis.
- CDS translation checks pass for the saved systems.
- Protein-level constraints dominate the current outcome.

## Assumptions, Limitations, and Scope
- Computational validation only
- No experimental validation
- Localization is heuristic, no ML
- Safe harbor not executed
- Genome feasibility not applied
- Expression is proxy-based
- Deployment currently NOT achievable without host engineering

## Field Notes
{field_note}

## Traceability
- `protease_exposure_analysis.csv` -> `total_protease_exposure_score` -> `0.65`
- `localization_refined.csv` -> `localization_class` -> `AMBIGUOUS SECRETORY PATHWAY`
- `final_decision_matrix.csv` -> `deployability` -> `NO`
"""

    corrections_log_lines = [
        "Corrections Applied Log",
        "=======================",
        "",
        "Source files reviewed",
        "---------------------",
        f"- {final_iter2_report}",
        f"- {final_decision_statement}",
        f"- {decision_confidence}",
        f"- {final_system_summary}",
        "",
        "Claim lexicon applied",
        "---------------------",
        '1. "Safe harbor prediction is implemented" -> "Safe harbor module implemented; not executed due to missing genome data"',
        '2. "predicts expression" -> "estimates expression potential using motif, occupancy, and constraint-based scoring"',
        '3. "expression prediction" -> "expression potential estimation (proxy-based)"',
        '4. "localization prediction" -> "sequence-based localization estimation (no ML model applied)"',
        '5. "validated pipeline" -> "computationally validated pipeline"',
        '6. "genome feasibility scoring" -> "genome feasibility framework implemented but not applied in final outputs"',
        '7. Benchmark clause added: "Baseline promoters show higher motif density, while AI promoters prioritize structural realism and reduced motif saturation."',
        '8. Core limitation clause added: "Primary limitation is protein-level constraints (localization ambiguity and protease-mediated degradation)."',
        "",
        "Context-aware corrections introduced in corrected documents",
        "--------------------------------------------------------",
        "- Added an Assumptions, Limitations, and Scope section once per corrected document.",
        "- Standardized the decision wording to: Decision = NOT DEPLOYABLE, Confidence = MEDIUM, Reason = protein-level constraints dominate, Path forward = host engineering + targeting (ER retention + A1 reduction).",
        "- Added header notes clarifying that expression_score is proxy-based, localization_class is sequence-based, and degradation_risk is model-based rather than experimental.",
        "- Preserved file paths, metric names, and values; no CSV or JSON values were modified.",
        "",
        "README status",
        "-------------",
        "- No repo-root README was found in the inspected project scope, so corrected_readme.md was generated as an additive summary document rather than a modified in-place README.",
    ]

    corrected_report_path = OUTPUTS / "corrected_report.md"
    corrected_readme_path = OUTPUTS / "corrected_readme.md"
    corrections_log_path = OUTPUTS / "corrections_applied_log.txt"

    write_text(corrected_report_path, corrected_report)
    write_text(corrected_readme_path, corrected_readme)
    write_text(corrections_log_path, "\n".join(corrections_log_lines))

    print("Generated:")
    print(f"- {corrected_report_path}")
    print(f"- {corrected_readme_path}")
    print(f"- {corrections_log_path}")


if __name__ == "__main__":
    main()
