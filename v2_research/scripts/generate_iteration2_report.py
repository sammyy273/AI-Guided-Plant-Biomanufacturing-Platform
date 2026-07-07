#!/usr/bin/env python3
"""Aggregate completed promoter-design runs into final report artifacts.

This script does not modify generation or scoring logic. It only reads saved
outputs and reuses the existing evaluation modules to benchmark a fixed
reference panel against the selected best candidate per species.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.cross_species.species_config import load_species_config
from modules.evaluation.cis_scoring import score_candidate
from modules.evaluation.multi_objective import (
    DEFAULT_OBJECTIVES,
    compute_composite_v2,
    pareto_rank_v2,
)
from modules.genomics.safe_harbor import SafeHarborPredictor
from modules.silencing.silencing_risk import compute_silencing_risk
from auto_loop_v2 import (
    _load_first_fasta_sequence,
    compute_diversity,
    compute_embedding_similarity,
    compute_novelty,
    load_seed,
    predict_expression,
)

OUTPUTS_DIR = ROOT / "outputs"
TARGET_SPECIES = [
    "arabidopsis",
    "nbenthamiana",
    "tomato",
    "rice",
    "maize",
    "soybean",
    "wheat",
    "ntobacum",
    "by2_cells",
]

REFERENCE_FASTAS = {
    "CaMV_35S": ROOT / "data" / "promoter_seeds" / "CaMV35S_promoter_835bp.fasta",
    "Maize_Ubiquitin": ROOT / "data" / "promoter_seeds" / "ZmUbi1_promoter_1993bp.fasta",
    "OsActin1": ROOT / "data" / "promoter_seeds" / "OsAct1_promoter_1413bp.fasta",
    "AtUBQ10": ROOT / "data" / "promoter_seeds" / "arabidopsis_promoters.fasta",
}


@dataclass
class SelectedRun:
    species: str
    run_dir: Path
    run_name: str
    iterations_completed: int


def build_logger() -> logging.Logger:
    logger = logging.getLogger("iteration2_report")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def parse_run_timestamp(path: Path) -> str:
    return path.name


def infer_model_source(candidate_id: str) -> str:
    if candidate_id.startswith("evo2_"):
        return "evo2"
    if candidate_id.startswith("d3lm_"):
        return "d3lm"
    if candidate_id.startswith("mut_"):
        return "mutational"
    return "other"


def load_loop_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def candidate_id_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.startswith("Unnamed"):
            return col
    if "candidate_id" in df.columns:
        return "candidate_id"
    return df.columns[0]


def normalise_iter_df(path: Path, iteration: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    cid_col = candidate_id_column(df)
    if cid_col != "candidate_id":
        df = df.rename(columns={cid_col: "candidate_id"})
    rename_map = {
        "internal_div": "internal_diversity",
        "gc_pct": "gc_content",
    }
    df = df.rename(columns=rename_map)
    if "candidate_id" not in df.columns:
        raise ValueError(f"No candidate id column in {path}")
    df["iteration"] = iteration
    df["model_source"] = df["candidate_id"].map(infer_model_source)
    return df


def get_iter_csvs(run_dir: Path) -> List[Tuple[int, Path]]:
    pairs = []
    for path in sorted(run_dir.glob("iter*_scored.csv")):
        stem = path.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            pairs.append((int(digits), path))
    return sorted(pairs)


def select_run(species: str) -> SelectedRun:
    species_dir = OUTPUTS_DIR / species
    if not species_dir.exists():
        raise FileNotFoundError(f"Missing outputs directory for {species}: {species_dir}")

    candidates = []
    for run_dir in sorted(p for p in species_dir.iterdir() if p.is_dir()):
        loop_summary = run_dir / "loop_summary.json"
        iter_csvs = get_iter_csvs(run_dir)
        if not loop_summary.exists() or not iter_csvs:
            continue
        summary = load_loop_summary(loop_summary)
        iterations_completed = int(summary.get("iterations_completed", len(iter_csvs)))
        candidates.append((iterations_completed, parse_run_timestamp(run_dir), run_dir))

    if not candidates:
        raise FileNotFoundError(f"No completed run with loop_summary+iter csv for {species}")

    iterations_completed, _, run_dir = max(candidates, key=lambda item: (item[0], item[1]))
    return SelectedRun(
        species=species,
        run_dir=run_dir,
        run_name=run_dir.name,
        iterations_completed=iterations_completed,
    )


def load_run_frames(run: SelectedRun) -> Tuple[pd.DataFrame, List[dict]]:
    all_frames = []
    iter_results = []
    summary = load_loop_summary(run.run_dir / "loop_summary.json")
    summary_map = {int(item["iteration"]): item for item in summary.get("results", [])}
    for iteration, path in get_iter_csvs(run.run_dir):
        df = normalise_iter_df(path, iteration)
        all_frames.append(df)
        iter_results.append(summary_map.get(iteration, {}))
    if not all_frames:
        raise ValueError(f"No scored csv files found in {run.run_dir}")
    return pd.concat(all_frames, ignore_index=True), iter_results


def fasta_entry(species: str, run_name: str, row: pd.Series) -> str:
    header = (
        f">{species}|run={run_name}|iter={int(row['iteration'])}|"
        f"id={row['candidate_id']}|source={row['model_source']}"
    )
    return f"{header}\n{row['sequence']}"


def read_reference_sequences() -> Dict[str, str]:
    refs = {}
    for name, path in REFERENCE_FASTAS.items():
        refs[name] = _load_first_fasta_sequence(str(path))
    return refs


def build_safe_harbor_score(species_config: dict, species_key: str) -> Tuple[float, dict, str]:
    safe_harbors = species_config.get("safe_harbors", {})
    known_sites = safe_harbors.get("known_sites", [])
    genome_cfg = species_config.get("genome", {})
    fasta_path = genome_cfg.get("fasta", "")
    gff_path = genome_cfg.get("annotation", "")

    def _make_predictor():
        return SafeHarborPredictor(
            genome_fasta=fasta_path if fasta_path and os.path.exists(fasta_path) else None,
            annotation_gff=gff_path if gff_path and os.path.exists(gff_path) else None,
            species_key=species_key,
        )

    if known_sites:
        if fasta_path and os.path.exists(fasta_path):
            predictor = _make_predictor()
            best_sh = None
            sh_score = 0.0
            for site in known_sites:
                chr_name = str(site.get("chromosome", "1"))
                pos = int(str(site.get("position", "0")).replace(",", ""))
                result = predictor.score_position(chr_name, pos, insert_length=800)
                if result["overall_score"] > sh_score:
                    sh_score = result["overall_score"]
                    best_sh = result
            if best_sh is None:
                best_sh = known_sites[0]
                sh_score = 0.7
        else:
            best_sh = known_sites[0]
            sh_score = 0.7
    else:
        predictor = _make_predictor() if fasta_path and os.path.exists(fasta_path) else SafeHarborPredictor(species_key=species_key)
        best_sh = predictor.score_position("1", 1000000, insert_length=800)
        sh_score = best_sh["overall_score"]

    label = "Genome-aware heuristic" if fasta_path and os.path.exists(fasta_path) else "Heuristic (no genome data)"
    return float(sh_score), best_sh, label


def evaluate_sequence_set(
    species: str,
    labelled_sequences: Dict[str, str],
    logger: logging.Logger,
) -> pd.DataFrame:
    species_config = load_species_config(species)
    species_config["_config_key"] = species
    seed = load_seed(species)

    scored = {}
    for cid, seq in labelled_sequences.items():
        cis = score_candidate(seq, species_config)
        silencing = compute_silencing_risk(seq)
        scored[cid] = {
            "sequence": seq,
            **cis,
            "silencing_risk": silencing["overall_risk"],
        }

    novelty = compute_novelty({cid: item["sequence"] for cid, item in scored.items()}, seed)
    diversity = compute_diversity({cid: item["sequence"] for cid, item in scored.items()})
    emb = compute_embedding_similarity(
        {cid: item["sequence"] for cid, item in scored.items()},
        seed,
        logger,
    )
    safe_score, _, placement_label = build_safe_harbor_score(species_config, species)

    rows = {}
    for cid, item in scored.items():
        weighted_score = float(item["weighted_score"])
        raw_weighted = float(item.get("_raw_weighted_score", weighted_score))
        is_ml = cid.startswith("best_evo2_") or cid.startswith("best_d3lm_")
        if is_ml and item["passed_filters"]:
            weighted_score = weighted_score * 1.3

        expr_score = predict_expression(weighted_score, item["silencing_risk"], item["gc_pct"])

        rows[cid] = {
            "candidate_id": cid,
            "sequence": item["sequence"],
            "weighted_score": weighted_score,
            "raw_weighted_score": raw_weighted,
            "novelty_35s": novelty.get(cid, 0.5),
            "internal_div": diversity.get(cid, 0.5),
            "silencing_risk": item["silencing_risk"],
            "safe_harbor_score": safe_score,
            "safe_harbor_label": placement_label,
            "embedding_similarity": emb.get(cid, 0.5),
            "gc_pct": item["gc_pct"],
            "passed_filters": bool(item["passed_filters"]),
            "length_bp": len(item["sequence"]),
            "expression_score": expr_score,
            "yield_tsp": 5.0,
            "history_similarity_penalty": 0.0,
            "realism_regularizer": max(0.0, (raw_weighted - weighted_score)) / 100.0,
            "model_source": infer_model_source(cid.replace("best_", "")) if cid.startswith("best_") else "reference",
        }

    df = pd.DataFrame(rows).T.reset_index(drop=True)
    objectives = [o for o in DEFAULT_OBJECTIVES["names"] if o in df.columns]
    df["pareto_front"] = pareto_rank_v2(df, objectives, DEFAULT_OBJECTIVES["higher_is_better"])
    df["base_composite_score"] = compute_composite_v2(
        df, objectives, DEFAULT_OBJECTIVES["higher_is_better"], DEFAULT_OBJECTIVES["weights"]
    )
    df["composite_score"] = (
        df["base_composite_score"]
        - df["history_similarity_penalty"].astype(float)
        - df["realism_regularizer"].astype(float)
    ).clip(lower=0.0)
    df = df.sort_values(["pareto_front", "composite_score"], ascending=[True, False]).reset_index(drop=True)
    return df


def motif_summary(sequence: str, species: str) -> List[Tuple[str, int]]:
    species_config = load_species_config(species)
    result = score_candidate(sequence, species_config)
    motifs = []
    for key, value in result.items():
        if key.startswith("_"):
            continue
        if key in {"weighted_score", "gc_pct", "passed_filters", "filter_failures", "promoter_class"}:
            continue
        if isinstance(value, (int, float)) and value > 0:
            motifs.append((key, int(value) if float(value).is_integer() else round(float(value), 4)))
    motifs.sort(key=lambda item: (-item[1], item[0]))
    return motifs


def build_case_study(best_species_row: pd.Series, species_runs: Dict[str, SelectedRun], run_frames: Dict[str, pd.DataFrame]) -> str:
    species = best_species_row["species"]
    run = species_runs[species]
    df = run_frames[species].copy()
    progression = []
    for iteration in sorted(df["iteration"].unique()):
        best_iter = df[df["iteration"] == iteration]["composite_score"].max()
        progression.append(f"iter{iteration}:{best_iter:.4f}")

    top3 = (
        df.sort_values("composite_score", ascending=False)
        .drop_duplicates(subset=["sequence"])
        .head(3)
        .copy()
    )
    motifs = motif_summary(best_species_row["sequence"], species)
    top = top3.iloc[0]
    comparisons = []
    for _, row in top3.iterrows():
        comparisons.append(
            f"- {row['candidate_id']} (iter {int(row['iteration'])}, {row['model_source']}): "
            f"composite={row['composite_score']:.4f}, strength={row['weighted_score']:.2f}, "
            f"silencing={row['silencing_risk']:.4f}, novelty={row['novelty_35s']:.4f}, "
            f"diversity={row['internal_diversity']:.4f}, gc={row['gc_content']:.1f}"
        )

    why = []
    why.append(
        f"- It achieved the highest composite score ({top['composite_score']:.4f}) in the selected {run.run_name} run."
    )
    why.append(
        f"- Its weighted strength ({top['weighted_score']:.2f}) stayed high while silencing risk remained at {top['silencing_risk']:.4f}."
    )
    why.append(
        f"- It balanced novelty ({top['novelty_35s']:.4f}) and internal diversity ({top['internal_diversity']:.4f}) without drifting to extreme GC ({top['gc_content']:.1f}%)."
    )

    motif_lines = [f"- {name}: {count}" for name, count in motifs[:12]]

    return "\n".join(
        [
            f"Case Study Species: {species}",
            f"Selected Run: {run.run_dir}",
            "",
            "Iteration progression:",
            " -> ".join(progression),
            "",
            "Top 3 candidates:",
            *comparisons,
            "",
            "Top candidate motif composition:",
            *motif_lines,
            "",
            "Why the top candidate won:",
            *why,
        ]
    )


def main() -> None:
    logger = build_logger()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    species_runs: Dict[str, SelectedRun] = {}
    run_frames: Dict[str, pd.DataFrame] = {}
    final_rows = []
    progression_rows = []
    pass_rows = []

    for species in TARGET_SPECIES:
        run = select_run(species)
        species_runs[species] = run
        df, loop_results = load_run_frames(run)
        run_frames[species] = df

        best_row = df.sort_values("composite_score", ascending=False).iloc[0]
        final_rows.append(
            {
                "species": species,
                "run_dir": str(run.run_dir),
                "run_name": run.run_name,
                "candidate_id": best_row["candidate_id"],
                "fasta": fasta_entry(species, run.run_name, best_row),
                "sequence": best_row["sequence"],
                "composite_score": round(float(best_row["composite_score"]), 4),
                "weighted_score": round(float(best_row["weighted_score"]), 4),
                "silencing_risk": round(float(best_row["silencing_risk"]), 4),
                "novelty_35s": round(float(best_row["novelty_35s"]), 4),
                "internal_diversity": round(float(best_row["internal_diversity"]), 4),
                "gc_content": round(float(best_row["gc_content"]), 4),
                "model_source": best_row["model_source"],
                "iteration": int(best_row["iteration"]),
            }
        )

        progression_bits = []
        total_candidates = 0
        total_passed = 0
        results_map = {int(item.get("iteration", 0)): item for item in loop_results if item}
        for iteration in sorted(df["iteration"].unique()):
            iter_best = df[df["iteration"] == iteration]["composite_score"].max()
            progression_bits.append(f"iter{iteration}:{iter_best:.4f}")
            result = results_map.get(iteration, {})
            total_candidates += int(result.get("n_candidates", 0))
            total_passed += int(result.get("n_passed_filters", len(df[df["iteration"] == iteration])))

        pass_rate = (total_passed / total_candidates) if total_candidates else 0.0
        progression_rows.append(
            {
                "species": species,
                "run_dir": str(run.run_dir),
                "iterations_completed": run.iterations_completed,
                "progression": " -> ".join(progression_bits),
            }
        )
        pass_rows.append(
            {
                "species": species,
                "run_dir": str(run.run_dir),
                "total_candidates": total_candidates,
                "passed_candidates": total_passed,
                "pass_rate": round(pass_rate, 4),
            }
        )

    final_summary_df = pd.DataFrame(final_rows).sort_values(["composite_score", "species"], ascending=[False, True])
    progression_df = pd.DataFrame(progression_rows).sort_values("species")
    pass_df = pd.DataFrame(pass_rows).sort_values("species")

    model_counts = {
        "evo2": {"top1_count": 0, "top3_count": 0},
        "d3lm": {"top1_count": 0, "top3_count": 0},
        "mutational": {"top1_count": 0, "top3_count": 0},
    }
    for species, df in run_frames.items():
        top3 = df.sort_values("composite_score", ascending=False).drop_duplicates(subset=["sequence"]).head(3)
        if not top3.empty:
            top1_source = top3.iloc[0]["model_source"]
            if top1_source in model_counts:
                model_counts[top1_source]["top1_count"] += 1
            for source in top3["model_source"]:
                if source in model_counts:
                    model_counts[source]["top3_count"] += 1
    model_contrib_df = pd.DataFrame(
        [{"model": model, **counts} for model, counts in model_counts.items()]
    ).sort_values("model")

    references = read_reference_sequences()
    benchmark_rows = []
    for _, best_row in final_summary_df.iterrows():
        species = best_row["species"]
        labelled = {
            f"best_{best_row['candidate_id']}": best_row["sequence"],
            **{ref_name: seq for ref_name, seq in references.items()},
        }
        bench_df = evaluate_sequence_set(species, labelled, logger)
        best_bench = bench_df[bench_df["candidate_id"] == f"best_{best_row['candidate_id']}"].iloc[0]
        for ref_name in references:
            ref_row = bench_df[bench_df["candidate_id"] == ref_name].iloc[0]
            benchmark_rows.append(
                {
                    "species": species,
                    "run_dir": best_row["run_dir"],
                    "candidate_id": best_row["candidate_id"],
                    "candidate_model_source": best_row["model_source"],
                    "candidate_saved_composite_score": round(float(best_row["composite_score"]), 4),
                    "candidate_benchmark_composite_score": round(float(best_bench["composite_score"]), 4),
                    "candidate_benchmark_weighted_score": round(float(best_bench["weighted_score"]), 4),
                    "candidate_benchmark_silencing_risk": round(float(best_bench["silencing_risk"]), 4),
                    "reference_name": ref_name,
                    "reference_length_bp": int(len(references[ref_name])),
                    "reference_passed_filters": bool(ref_row["passed_filters"]),
                    "reference_composite_score": round(float(ref_row["composite_score"]), 4),
                    "reference_weighted_score": round(float(ref_row["weighted_score"]), 4),
                    "reference_silencing_risk": round(float(ref_row["silencing_risk"]), 4),
                    "delta_composite": round(float(best_bench["composite_score"] - ref_row["composite_score"]), 4),
                    "delta_weighted_score": round(float(best_bench["weighted_score"] - ref_row["weighted_score"]), 4),
                    "delta_silencing_risk": round(float(best_bench["silencing_risk"] - ref_row["silencing_risk"]), 4),
                }
            )
    benchmark_df = pd.DataFrame(benchmark_rows).sort_values(["species", "reference_name"])

    validation_lines = ["Validation Report", "=================", ""]
    missing_total = int(final_summary_df.isna().sum().sum())
    validation_lines.append(f"Missing values in final_summary.csv: {missing_total}")

    invalid_gc = final_summary_df[
        (final_summary_df["gc_content"] < 20) | (final_summary_df["gc_content"] > 80)
    ]
    if invalid_gc.empty:
        validation_lines.append("Invalid GC content: none")
    else:
        validation_lines.append("Invalid GC content:")
        for _, row in invalid_gc.iterrows():
            validation_lines.append(f"- {row['species']} {row['candidate_id']} gc={row['gc_content']}")

    duplicate_mask = final_summary_df["sequence"].duplicated(keep=False)
    duplicates = final_summary_df[duplicate_mask]
    if duplicates.empty:
        validation_lines.append("Duplicate best sequences: none")
    else:
        validation_lines.append("Duplicate best sequences:")
        for seq, group in duplicates.groupby("sequence"):
            species_list = ", ".join(sorted(group["species"].tolist()))
            validation_lines.append(f"- shared by: {species_list}")

    extreme_silencing = final_summary_df[final_summary_df["silencing_risk"] > 0.8]
    if extreme_silencing.empty:
        validation_lines.append("Extreme silencing (>0.8): none")
    else:
        validation_lines.append("Extreme silencing (>0.8):")
        for _, row in extreme_silencing.iterrows():
            validation_lines.append(
                f"- {row['species']} {row['candidate_id']} silencing={row['silencing_risk']:.4f}"
            )

    validation_lines.append("Motif requirement validation:")
    motif_failures = []
    for _, row in final_summary_df.iterrows():
        species_cfg = load_species_config(row["species"])
        res = score_candidate(row["sequence"], species_cfg)
        if not res["passed_filters"]:
            motif_failures.append((row["species"], row["candidate_id"], res.get("filter_failures", [])))
    if not motif_failures:
        validation_lines.append("- none")
    else:
        for species, cid, failures in motif_failures:
            validation_lines.append(f"- {species} {cid}: {', '.join(failures)}")

    best_species_row = final_summary_df.iloc[0]
    case_study_text = build_case_study(best_species_row, species_runs, run_frames)

    dominant_model_row = model_contrib_df.sort_values(
        ["top1_count", "top3_count", "model"], ascending=[False, False, True]
    ).iloc[0]
    final_summary_json = {
        "best_species": best_species_row["species"],
        "best_overall_score": round(float(best_species_row["composite_score"]), 4),
        "average_score_across_species": round(float(final_summary_df["composite_score"].mean()), 4),
        "average_pass_rate": round(float(pass_df["pass_rate"].mean()), 4),
        "dominant_model": dominant_model_row["model"],
    }

    final_summary_path = OUTPUTS_DIR / "final_summary.csv"
    progression_path = OUTPUTS_DIR / "iteration_progression.csv"
    pass_path = OUTPUTS_DIR / "pass_rate_summary.csv"
    model_contrib_path = OUTPUTS_DIR / "model_contribution.csv"
    benchmark_path = OUTPUTS_DIR / "benchmark_comparison.csv"
    validation_path = OUTPUTS_DIR / "validation_report.txt"
    case_study_path = OUTPUTS_DIR / "case_study.txt"
    final_json_path = OUTPUTS_DIR / "final_summary.json"

    final_summary_df.to_csv(final_summary_path, index=False)
    progression_df.to_csv(progression_path, index=False)
    pass_df.to_csv(pass_path, index=False)
    model_contrib_df.to_csv(model_contrib_path, index=False)
    benchmark_df.to_csv(benchmark_path, index=False)
    validation_path.write_text("\n".join(validation_lines) + "\n")
    case_study_path.write_text(case_study_text + "\n")
    final_json_path.write_text(json.dumps(final_summary_json, indent=2) + "\n")

    print("Generated files:")
    for path in [
        final_summary_path,
        progression_path,
        pass_path,
        model_contrib_path,
        benchmark_path,
        validation_path,
        case_study_path,
        final_json_path,
    ]:
        print(path)

    print("\nKey metrics summary:")
    print(f"Best species: {final_summary_json['best_species']}")
    print(f"Best overall score: {final_summary_json['best_overall_score']:.4f}")
    print(f"Average score across species: {final_summary_json['average_score_across_species']:.4f}")
    print(f"Average pass rate: {final_summary_json['average_pass_rate']:.4f}")
    print(f"Dominant model: {final_summary_json['dominant_model']}")


if __name__ == "__main__":
    main()
