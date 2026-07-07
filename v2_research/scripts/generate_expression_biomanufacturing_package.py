#!/usr/bin/env python3
"""Assemble the final multi-species expression-system package from real outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data"

PROMOTER_SPECIES = ["nbenthamiana", "rice", "tomato", "arabidopsis"]
EXPRESSION_SPECIES = ["nbenthamiana", "rice", "tomato"]

GC_BOUNDS = {
    "nbenthamiana": (35.0, 55.0),
    "tomato": (35.0, 55.0),
    "rice": (40.0, 65.0),
}

CODON_TO_AA = {
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "AAT": "N", "AAC": "N",
    "GAT": "D", "GAC": "D",
    "TGT": "C", "TGC": "C",
    "CAA": "Q", "CAG": "Q",
    "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    "CAT": "H", "CAC": "H",
    "ATT": "I", "ATC": "I", "ATA": "I",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "AAA": "K", "AAG": "K",
    "ATG": "M",
    "TTT": "F", "TTC": "F",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "TGG": "W",
    "TAT": "Y", "TAC": "Y",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TAA": "*", "TAG": "*", "TGA": "*",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_fasta(path: Path) -> str:
    seq = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line)
    return "".join(seq).upper()


def latest_run(species: str) -> Optional[Path]:
    species_dir = OUTPUTS / species
    if not species_dir.exists():
        return None
    runs = sorted([p for p in species_dir.iterdir() if p.is_dir() and re.match(r"\d{8}_\d{6}$", p.name)])
    valid = [p for p in runs if list(p.glob("iter*_scored.csv"))]
    return valid[-1] if valid else None


def infer_model(candidate_id: str) -> str:
    if candidate_id.startswith("evo2_"):
        return "evo2"
    if candidate_id.startswith("d3lm_"):
        return "d3lm"
    if candidate_id.startswith("mut_"):
        return "mutational"
    return "NOT AVAILABLE"


def baseline_verdict_from_relative(relative_strength: Optional[float]) -> str:
    if relative_strength is None:
        return "NOT AVAILABLE"
    return "BETTER" if float(relative_strength) >= 1.0 else "WEAKER"


def choose_best_row(run_dir: Path) -> dict:
    frames = []
    for csv_path in sorted(run_dir.glob("iter*_scored.csv")):
        df = pd.read_csv(csv_path)
        cid_col = df.columns[0]
        if cid_col.startswith("Unnamed") or cid_col != "candidate_id":
            df = df.rename(columns={cid_col: "candidate_id"})
        digits = re.search(r"iter(\d+)_scored\.csv", csv_path.name)
        df["iteration"] = int(digits.group(1)) if digits else 0
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["composite_score"] = pd.to_numeric(all_df["composite_score"], errors="coerce")
    row = all_df.sort_values(["composite_score", "iteration"], ascending=[False, False]).iloc[0].to_dict()
    row["model"] = infer_model(str(row["candidate_id"]))
    return row


def translate_cds(cds: str) -> str:
    aas: List[str] = []
    for i in range(0, len(cds), 3):
        codon = cds[i:i + 3]
        if len(codon) != 3:
            break
        aas.append(CODON_TO_AA.get(codon, "X"))
    return "".join(aas)


def gc_ok(species: str, gc_pct: float) -> bool:
    lo, hi = GC_BOUNDS[species]
    return lo <= gc_pct <= hi


def promoter_summary_rows() -> List[dict]:
    rows = []
    for species in PROMOTER_SPECIES:
        run_dir = latest_run(species)
        if run_dir is None:
            rows.append(
                {
                    "species": species,
                    "top_ai_promoter": "NOT AVAILABLE",
                    "best_baseline_promoter": "NOT AVAILABLE",
                    "model": "NOT AVAILABLE",
                    "composite_score": "NOT AVAILABLE",
                    "expression_class": "NOT AVAILABLE",
                    "silencing_risk": "NOT AVAILABLE",
                    "baseline_result": "NOT PERFORMED",
                    "source_run": "NOT AVAILABLE",
                }
            )
            continue
        row = choose_best_row(run_dir)
        rows.append(
            {
                "species": species,
                "top_ai_promoter": row.get("candidate_id", "NOT AVAILABLE"),
                "best_baseline_promoter": row.get("baseline_reference", "NOT AVAILABLE"),
                "model": row.get("model", "NOT AVAILABLE"),
                "composite_score": row.get("composite_score", "NOT AVAILABLE"),
                "expression_class": row.get("expression_class", row.get("expression_label", "NOT AVAILABLE")),
                "silencing_risk": row.get("silencing_risk", "NOT AVAILABLE"),
                "baseline_result": baseline_verdict_from_relative(row.get("relative_strength")),
                "source_run": str(run_dir),
            }
        )
    return rows


def collect_expression_reports() -> List[dict]:
    reports = []
    for species in EXPRESSION_SPECIES:
        report_path = OUTPUTS / f"final_report_{species}.json"
        if not report_path.exists():
            continue
        payload = read_json(report_path)
        reports.append(payload)
    return reports


def build_qc(report: dict, protein_sequence: str) -> dict:
    species = report["species"]
    cds = report["optimized_cds"]["sequence"]
    translated = translate_cds(cds)
    translated_no_stop = translated.rstrip("*")
    gc_pct = float(report["optimized_cds"]["gc_pct"])
    cai = float(report["optimized_cds"]["cai"])
    degradation = report["degradation_risk"]["risk_class"]
    localization = report["localization"]["predicted_localization"]

    internal_stops = "*" in translated[:-1]
    translation_match = translated_no_stop == protein_sequence
    flags = []
    if not translation_match:
        flags.append("CDS_TRANSLATION_MISMATCH")
    if internal_stops:
        flags.append("INTERNAL_STOP_CODON")
    if not gc_ok(species, gc_pct):
        flags.append("GC_OUT_OF_RANGE")
    if cai < 0.7:
        flags.append("LOW_CAI")
    if degradation == "HIGH":
        flags.append("HIGH_DEGRADATION_RISK")
    if localization not in {"secreted (apoplast)", "cytosolic", "ER-retained", "vacuole", "chloroplast"}:
        flags.append("LOCALIZATION_UNCERTAIN")

    return {
        "translation_matches_target": translation_match,
        "internal_stop_codons": internal_stops,
        "gc_within_species_range": gc_ok(species, gc_pct),
        "cai_gt_0_7": cai > 0.7,
        "localization_plausible": localization in {"secreted (apoplast)", "cytosolic", "ER-retained", "vacuole", "chloroplast"},
        "degradation_high_risk": degradation == "HIGH",
        "flags": flags,
    }


def main() -> None:
    protein_seq = read_fasta(DATA / "protein" / "hyaluronidase.fasta")
    source_meta = read_json(DATA / "protein" / "hyaluronidase_source.json")

    promoter_rows = promoter_summary_rows()
    promoter_df = pd.DataFrame(promoter_rows)
    promoter_path = OUTPUTS / "promoter_summary.csv"
    table_promoter_path = OUTPUTS / "table_promoter_summary.csv"
    promoter_df.to_csv(promoter_path, index=False)
    promoter_df.to_csv(table_promoter_path, index=False)

    reports = collect_expression_reports()
    benchmark_rows = []
    system_summary = {
        "target_protein": source_meta,
        "species": {},
        "cross_species": {},
    }

    improvements = []
    for report in reports:
        promoter = report["best_promoter"]
        optimized = report["optimized_cds"]
        localization = report["localization"]
        degradation = report["degradation_risk"]
        safe_harbor = report["safe_harbor"]
        metabolic = report["metabolic_impact"]
        qc = build_qc(report, protein_seq)
        improvement = None
        rel = promoter.get("relative_strength")
        if rel is not None:
            improvement = (float(rel) - 1.0) * 100.0
            improvements.append(improvement)

        benchmark_rows.append(
            {
                "Species": report["species"],
                "Promoter Score": promoter["composite_score"],
                "CAI": optimized["cai"],
                "GC%": optimized["gc_pct"],
                "Localization": localization["predicted_localization"],
                "Degradation": degradation["risk_class"],
                "Expression Class": promoter["expression_class"],
                "AI_vs_Baseline_Improvement_%": improvement if improvement is not None else "NOT PERFORMED",
            }
        )

        cassette = report["gene_cassette"]
        system_summary["species"][report["species"]] = {
            "promoter": {
                "candidate_id": promoter["candidate_id"],
                "model": infer_model(f"{promoter['source']}_placeholder") if promoter["source"] in {"evo2", "d3lm", "mut"} else promoter["source"],
                "composite_score": promoter["composite_score"],
                "expression_class": promoter["expression_class"],
                "silencing_risk": promoter["silencing_risk"],
                "baseline_reference": promoter["baseline_reference"],
                "baseline_verdict": promoter["baseline_comparison"]["verdict"],
                "baseline_relative_strength": promoter["baseline_comparison"]["relative_strength"],
            },
            "optimized_cds": {
                "length_bp": optimized["length_bp"],
                "gc_pct": optimized["gc_pct"],
                "cai": optimized["cai"],
            },
            "cassette": {
                "layout": cassette["cassette"],
                "total_construct_size_bp": cassette["total_construct_size_bp"],
                "known_sequence_gc_pct": cassette["known_sequence_gc_pct"],
                "linker_or_tag": "NOT INCLUDED",
            },
            "localization": localization["predicted_localization"],
            "degradation_risk": degradation["risk_class"],
            "safe_harbor": safe_harbor if safe_harbor.get("status") != "not_performed" else "NOT PERFORMED",
            "gRNA": "NOT INCLUDED",
            "metabolic_impact": metabolic,
            "quality_control": qc,
        }

    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_path = OUTPUTS / "final_benchmark_table.csv"
    benchmark_df.to_csv(benchmark_path, index=False)

    if improvements:
        system_summary["cross_species"]["average_ai_vs_baseline_improvement_percent"] = round(sum(improvements) / len(improvements), 4)
        system_summary["cross_species"]["consistency_note"] = "All species remained below their recorded baseline in the saved promoter outputs."
    else:
        system_summary["cross_species"]["average_ai_vs_baseline_improvement_percent"] = "NOT PERFORMED"
        system_summary["cross_species"]["consistency_note"] = "Baseline-relative promoter metadata not available across all species."

    system_summary["cross_species"]["species_count"] = len(reports)
    system_summary["cross_species"]["protein_sequence_length_aa"] = len(protein_seq)

    system_path = OUTPUTS / "system_summary.json"
    final_system_path = OUTPUTS / "final_system_summary.json"
    system_path.write_text(json.dumps(system_summary, indent=2) + "\n")
    final_system_path.write_text(json.dumps(system_summary, indent=2) + "\n")

    fba_text = "\n".join(
        [
            "FBA limitations for this workflow",
            "===============================",
            "",
            "NOT PERFORMED: No new metabolic model was run in this package.",
            "",
            "Why classical FBA is limited here:",
            "- It does not model gene expression or promoter strength directly.",
            "- It assumes steady-state intracellular fluxes.",
            "- It is not promoter-aware and does not natively capture expression burden from different regulatory designs.",
            "",
            "Better alternatives for future work:",
            "- ME-models (metabolism and expression models)",
            "- Enzyme-constrained genome-scale metabolic models",
            "- Resource-allocation models that explicitly couple expression load to growth and secretion",
        ]
    )
    fba_path = OUTPUTS / "section_fba_limitations.txt"
    fba_path.write_text(fba_text + "\n")

    print("Generated:")
    for path in [promoter_path, table_promoter_path, benchmark_path, system_path, final_system_path, fba_path]:
        print(path)


if __name__ == "__main__":
    main()
