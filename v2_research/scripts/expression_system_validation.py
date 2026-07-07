#!/usr/bin/env python3
"""Build downstream expression-system validation reports from promoter outputs.

This script does not modify promoter generation or scoring logic. It consumes
saved promoter outputs, a real protein amino-acid sequence, and optional extra
assets to produce species-level expression-system reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.cross_species.species_config import load_species_config
from modules.evaluation.cis_scoring import compute_hybrid_expression, score_candidate
from modules.protein.expression_system import (
    analyse_structure,
    assess_degradation_risk,
    build_cassette_map,
    design_grna_library,
    gc_content,
    metabolic_feasibility_summary,
    optimize_cds,
    predict_localization,
    read_fasta_sequence,
    report_confidence,
    restriction_site_summary,
    write_fasta,
)


TARGET_SPECIES = ["nbenthamiana", "rice", "tomato"]
OUTPUTS_DIR = ROOT / "outputs"


def load_candidate_row(species: str) -> Tuple[dict, Path]:
    """Select the strongest available promoter row for a species.

    Preference order:
    1. Runs that contain extended downstream fields from the current pipeline
       (`expression_label` and `relative_strength`)
    2. Highest composite score among those runs
    3. Fall back to any historical run if necessary
    """
    species_dir = OUTPUTS_DIR / species
    if not species_dir.exists():
        raise FileNotFoundError(f"Missing outputs directory for {species}")

    best_extended = None
    best_any = None

    for run_dir in sorted(p for p in species_dir.iterdir() if p.is_dir()):
        for csv_path in sorted(run_dir.glob("iter*_scored.csv")):
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if df.empty:
                continue
            cid_col = df.columns[0]
            if cid_col.startswith("Unnamed"):
                df = df.rename(columns={cid_col: "candidate_id"})
            elif "candidate_id" not in df.columns:
                df = df.rename(columns={cid_col: "candidate_id"})

            if "composite_score" not in df.columns:
                continue

            idx = df["composite_score"].astype(float).idxmax()
            row = df.loc[idx].to_dict()
            row["_csv_path"] = str(csv_path)
            row["_run_dir"] = str(run_dir)
            row["_iteration"] = int("".join(ch for ch in csv_path.stem if ch.isdigit()) or "0")
            row["_composite_score"] = float(row["composite_score"])

            has_extended = "expression_label" in df.columns and "relative_strength" in df.columns

            if best_any is None or row["_composite_score"] > best_any["_composite_score"]:
                best_any = row
            if has_extended and (best_extended is None or row["_composite_score"] > best_extended["_composite_score"]):
                best_extended = row

    chosen = best_extended or best_any
    if chosen is None:
        raise FileNotFoundError(f"No scored promoter candidates found for {species}")
    return chosen, Path(chosen["_run_dir"])


def ensure_expression_metrics(row: dict, species: str) -> dict:
    if all(key in row for key in ("expression_score", "expression_label", "relative_strength")):
        return row

    species_config = load_species_config(species)
    cis = score_candidate(row["sequence"], species_config)
    expr = compute_hybrid_expression(
        cis,
        silencing_risk=float(row.get("silencing_risk", 0.0)),
        embedding_similarity=float(row.get("embedding_similarity", 0.95)),
        internal_div=float(row.get("internal_div", row.get("internal_diversity", 0.4))),
    )
    row["expression_score"] = expr["expression_score"]
    row["expression_label"] = expr["expression_class"]
    row["expression_class"] = expr["expression_class"]
    row["occupancy"] = expr["occupancy"]
    row["binding_score"] = expr["binding_score"]
    row["expression_penalty"] = expr["penalty"]
    row["spacing_penalty"] = expr["spacing_penalty"]
    row["expression_confidence"] = expr["confidence"]
    row["norm_cis"] = expr["norm_cis"]
    row["norm_gc"] = expr["norm_gc"]
    row["norm_silencing"] = expr["norm_silencing"]
    row["norm_embed"] = expr["norm_embed"]
    row["norm_div"] = expr["norm_div"]
    if "relative_strength" not in row:
        species_type = species_config.get("species", {}).get("type", "dicot")
        baseline = 0.70 if species_type == "monocot" else 0.60
        row["relative_strength"] = round(float(row["expression_score"]) / baseline, 4)
    if "baseline_reference" not in row:
        species_type = species_config.get("species", {}).get("type", "dicot")
        row["baseline_reference"] = "Maize_Ubiquitin" if species_type == "monocot" else "CaMV_35S"
    return row


def safe_harbor_summary(species: str) -> dict:
    cfg = load_species_config(species)
    genome_cfg = cfg.get("genome", {})
    fasta = genome_cfg.get("fasta")
    annotation = genome_cfg.get("annotation")
    fasta_exists = bool(fasta and (ROOT / fasta).exists())
    ann_exists = bool(annotation and (ROOT / annotation).exists())
    if not (fasta_exists and ann_exists):
        return {
            "status": "not_performed",
            "message": "Safe harbor prediction not performed (genome data unavailable)",
        }
    return {
        "status": "available_but_not_run",
        "message": "Genome assets are available locally, but safe-harbor integration should be run through the genome-aware promoter pipeline.",
        "genome_fasta": fasta,
        "annotation": annotation,
    }


def load_optional_sequence(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Missing optional sequence file: {p}")
    return read_fasta_sequence(str(p))


def load_target_cds_records(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    records: Dict[str, str] = {}
    p = Path(path).expanduser()
    if p.is_dir():
        for fasta in sorted(p.glob("*.fasta")):
            records[fasta.stem] = read_fasta_sequence(str(fasta))
        return records
    current = None
    seq_parts: List[str] = []
    with open(p) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current and seq_parts:
                    records[current] = "".join(seq_parts).upper()
                current = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
        if current and seq_parts:
            records[current] = "".join(seq_parts).upper()
    return records


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def build_species_report(
    species: str,
    protein_name: str,
    protein_sequence: str,
    promoter_row: dict,
    optimized_cds: dict,
    localization: dict,
    degradation: dict,
    cassette: dict,
    safe_harbor: dict,
    metabolic: dict,
    grna_records: List[dict],
) -> dict:
    promoter_metrics = {
        "candidate_id": promoter_row.get("candidate_id"),
        "sequence": promoter_row["sequence"],
        "composite_score": float(promoter_row.get("composite_score", 0.0)),
        "weighted_score": float(promoter_row.get("weighted_score", 0.0)),
        "silencing_risk": float(promoter_row.get("silencing_risk", 0.0)),
        "novelty_35s": float(promoter_row.get("novelty_35s", 0.0)),
        "internal_diversity": float(promoter_row.get("internal_div", promoter_row.get("internal_diversity", 0.0))),
        "gc_content": float(promoter_row.get("gc_pct", promoter_row.get("gc_content", 0.0))),
        "expression_score": float(promoter_row.get("expression_score", 0.0)),
        "expression_class": promoter_row.get("expression_label", promoter_row.get("expression_class")),
        "relative_strength": float(promoter_row.get("relative_strength", 0.0)),
        "baseline_reference": promoter_row.get("baseline_reference"),
        "baseline_comparison": {
            "promoter": promoter_row.get("baseline_reference"),
            "relative_strength": float(promoter_row.get("relative_strength", 0.0)),
            "verdict": "BETTER" if float(promoter_row.get("relative_strength", 0.0)) >= 1.0 else "WEAKER",
        },
        "source": promoter_row.get("candidate_id", "").split("_")[0],
        "iteration": int(promoter_row.get("_iteration", 0)),
    }
    confidence = report_confidence(promoter_metrics, localization, degradation, optimized_cds)
    return {
        "species": species,
        "protein_name": protein_name,
        "best_promoter": promoter_metrics,
        "optimized_cds": {
            "sequence": optimized_cds["optimized_cds"],
            "length_bp": optimized_cds["length_bp"],
            "gc_pct": optimized_cds["gc_pct"],
            "cai": optimized_cds["cai"],
            "warnings": optimized_cds["codon_warnings"],
        },
        "localization": localization,
        "degradation_risk": degradation,
        "gene_cassette": cassette,
        "safe_harbor": safe_harbor,
        "metabolic_impact": metabolic,
        "grna_strategy": {
            "status": "available" if grna_records else "not_available",
            "count": len(grna_records),
            "records": grna_records[:5],
        },
        "confidence": confidence,
        "notes": [
            "All promoter metrics are computational predictions.",
            "Experimental validation is required to confirm in vivo expression.",
            "DeepPlantCRE is not implemented in this workflow; it remains future work.",
        ],
    }


def write_cassette_map(path: Path, cassette: dict, promoter_name: str, localization: dict) -> None:
    lines = [
        f"Species: {cassette['species']}",
        f"Promoter: {promoter_name}",
        f"Predicted localization: {localization.get('predicted_localization')}",
        "",
        "Cassette:",
        " -> ".join(part for part in cassette["cassette"] if part),
        "",
        f"Promoter bp: {cassette['parts_bp']['promoter_bp']}",
        f"5'UTR bp: {cassette['parts_bp']['utr_bp']}",
        f"Signal peptide bp: {cassette['parts_bp']['signal_peptide_bp']}",
        f"CDS bp: {cassette['parts_bp']['cds_bp']}",
        f"Terminator bp: {cassette['parts_bp']['terminator_bp']}",
        f"Total construct size bp: {cassette['total_construct_size_bp']}",
        f"Known sequence GC%: {cassette['known_sequence_gc_pct']}",
        "",
        "Restriction sites:",
    ]
    if cassette["restriction_sites"]:
        for hit in cassette["restriction_sites"]:
            lines.append(f"- {hit['enzyme']} at {hit['position']} ({hit['motif']})")
    else:
        lines.append("- none in provided sequence parts")
    lines.extend(["", f"Notes: {cassette['notes']}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downstream expression-system validation")
    parser.add_argument("--protein-fasta", help="Target enzyme amino-acid FASTA")
    parser.add_argument("--protein-name", default="target_enzyme")
    parser.add_argument("--species", nargs="*", default=TARGET_SPECIES)
    parser.add_argument("--utr-fasta", help="Optional exact 5' UTR sequence FASTA")
    parser.add_argument("--terminator-fasta", help="Optional exact terminator sequence FASTA")
    parser.add_argument("--alphafold-json", help="Optional AlphaFold JSON with pLDDT values")
    parser.add_argument("--alphafold-pdb", help="Optional AlphaFold PDB path")
    parser.add_argument("--target-cds-fasta", help="Optional FASTA or directory of protease-target CDS sequences for gRNA design")
    parser.add_argument("--check-only", action="store_true", help="Validate local assets without generating outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.check_only:
        status = {
            "protein_fasta_provided": bool(args.protein_fasta),
            "species": args.species,
            "genome_assets": {species: safe_harbor_summary(species) for species in args.species},
        }
        print(json.dumps(status, indent=2))
        return 0

    if not args.protein_fasta:
        raise SystemExit("A real protein FASTA is required: --protein-fasta /path/to/target_enzyme.fasta")

    protein_sequence = read_fasta_sequence(os.path.expanduser(args.protein_fasta))
    if not protein_sequence:
        raise SystemExit(f"Protein FASTA appears empty: {args.protein_fasta}")

    utr_sequence = load_optional_sequence(args.utr_fasta)
    terminator_sequence = load_optional_sequence(args.terminator_fasta)
    target_cds_records = load_target_cds_records(args.target_cds_fasta)

    structure_payload = analyse_structure(
        protein_sequence,
        alphafold_json=args.alphafold_json,
        alphafold_pdb=args.alphafold_pdb,
    )
    write_json(OUTPUTS_DIR / "protein" / "structure_analysis.json", structure_payload)

    grna_rows: List[dict] = []
    metabolic_payload: Dict[str, dict] = {}

    for species in args.species:
        promoter_row, run_dir = load_candidate_row(species)
        promoter_row = ensure_expression_metrics(promoter_row, species)

        species_out = OUTPUTS_DIR / species
        species_out.mkdir(parents=True, exist_ok=True)

        promoter_header = (
            f"{species}|candidate={promoter_row.get('candidate_id')}|"
            f"score={float(promoter_row.get('composite_score', 0.0)):.4f}"
        )
        write_fasta(
            str(species_out / "best_promoter.fasta"),
            promoter_header,
            promoter_row["sequence"],
        )

        optimized = optimize_cds(protein_sequence, species)
        write_fasta(
            str(species_out / "optimized_cds.fasta"),
            f"{args.protein_name}|{species}|optimized_cds",
            optimized["optimized_cds"],
        )

        localization = predict_localization(protein_sequence)
        write_json(species_out / "localization.json", localization)

        degradation = assess_degradation_risk(protein_sequence)
        write_json(species_out / "degradation_risk.json", degradation)

        cassette = build_cassette_map(
            species,
            promoter_row.get("candidate_id", "best_promoter"),
            promoter_row["sequence"],
            optimized["optimized_cds"],
            localization,
            utr_sequence=utr_sequence,
            terminator_sequence=terminator_sequence,
        )
        write_cassette_map(
            species_out / "gene_cassette_map.txt",
            cassette,
            promoter_row.get("candidate_id", "best_promoter"),
            localization,
        )

        safe_harbor = safe_harbor_summary(species)
        metabolic = metabolic_feasibility_summary(species, protein_sequence)
        metabolic_payload[species] = metabolic

        genome_available = safe_harbor.get("status") != "not_performed"
        species_grnas = design_grna_library(target_cds_records, genome_available) if target_cds_records else []
        if species == "nbenthamiana":
            grna_rows.extend(species_grnas)

        report = build_species_report(
            species,
            args.protein_name,
            protein_sequence,
            promoter_row,
            optimized,
            localization,
            degradation,
            cassette,
            safe_harbor,
            metabolic,
            species_grnas,
        )
        write_json(OUTPUTS_DIR / f"final_report_{species}.json", report)

    write_json(OUTPUTS_DIR / "metabolic_analysis.json", metabolic_payload)

    grna_csv = OUTPUTS_DIR / "grna_library.csv"
    with open(grna_csv, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["gene", "gRNA_sequence", "efficiency", "off_target_count", "position", "status"],
        )
        writer.writeheader()
        for row in grna_rows:
            writer.writerow(row)

    print("Generated:")
    print(f"- {OUTPUTS_DIR / 'protein' / 'structure_analysis.json'}")
    print(f"- {OUTPUTS_DIR / 'metabolic_analysis.json'}")
    print(f"- {grna_csv}")
    for species in args.species:
        print(f"- {OUTPUTS_DIR / species / 'best_promoter.fasta'}")
        print(f"- {OUTPUTS_DIR / species / 'optimized_cds.fasta'}")
        print(f"- {OUTPUTS_DIR / species / 'localization.json'}")
        print(f"- {OUTPUTS_DIR / species / 'degradation_risk.json'}")
        print(f"- {OUTPUTS_DIR / species / 'gene_cassette_map.txt'}")
        print(f"- {OUTPUTS_DIR / f'final_report_{species}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
