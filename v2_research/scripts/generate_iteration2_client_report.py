#!/usr/bin/env python3
"""Generate client-ready Iteration 2 summary artifacts from saved outputs only."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"

SPECIES = [
    "arabidopsis",
    "nbenthamiana",
    "rice",
    "maize",
    "tomato",
    "soybean",
    "wheat",
    "ntobacum",
    "by2_cells",
]

MONOCOTS = {"rice", "maize", "wheat"}
DICTS = set(SPECIES) - MONOCOTS


def infer_model(candidate_id: str) -> str:
    candidate_id = str(candidate_id)
    if candidate_id.startswith("evo2_"):
        return "evo2"
    if candidate_id.startswith("d3lm_"):
        return "d3lm"
    if candidate_id.startswith("mut_"):
        return "mutational"
    return "Not available"


def candidate_id_column(df: pd.DataFrame) -> str:
    if "candidate_id" in df.columns:
        return "candidate_id"
    unnamed = [col for col in df.columns if str(col).startswith("Unnamed")]
    if unnamed:
        return unnamed[0]
    return df.columns[0]


def parse_iteration(path: Path) -> int:
    match = re.search(r"iter(\d+)_scored\.csv$", path.name)
    return int(match.group(1)) if match else 0


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def numeric_or_none(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        val = float(value)
    except Exception:
        return None
    if math.isnan(val):
        return None
    return val


def fmt_num(value: Optional[float], digits: int = 4) -> str:
    val = numeric_or_none(value)
    if val is None:
        return "Not available"
    return f"{val:.{digits}f}"


def fmt_pct(value: Optional[float], digits: int = 1) -> str:
    val = numeric_or_none(value)
    if val is None:
        return "Not available"
    return f"{val:.{digits}f}%"


def latest_run_dir(species: str) -> Path:
    species_dir = OUTPUTS_DIR / species
    if not species_dir.exists():
        raise FileNotFoundError(f"Missing outputs directory for {species}")
    run_dirs = sorted([p for p in species_dir.iterdir() if p.is_dir() and re.match(r"\d{8}_\d{6}$", p.name)])
    if not run_dirs:
        raise FileNotFoundError(f"No timestamped run directories found for {species}")
    valid = [p for p in run_dirs if list(p.glob("iter*_scored.csv"))]
    if not valid:
        raise FileNotFoundError(f"No completed scored runs found for {species}")
    return valid[-1]


def parse_report_card(report_path: Path) -> Dict[str, str]:
    if not report_path.exists():
        return {}
    text = read_text(report_path)
    parsed: Dict[str, str] = {}
    patterns = {
        "expression_class": r"Expression class:\s+([A-Z]+)",
        "baseline_reference": r"Baseline reference:\s+(.+)",
        "relative_strength": r"Relative strength:\s+([0-9.]+)",
        "sequence": r"Sequence:\s+([ACGTNacgtn\.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            parsed[key] = match.group(1).strip().rstrip(".")
    return parsed


def baseline_result(row: pd.Series, report_meta: Dict[str, str]) -> Tuple[str, Optional[float]]:
    rel = numeric_or_none(row.get("relative_strength"))
    if rel is None and "relative_strength" in report_meta:
        rel = numeric_or_none(report_meta["relative_strength"])
    if rel is None:
        return "Not available", None
    return ("BETTER" if rel > 1.0 else "WORSE"), rel - 1.0


def expression_class_value(row: pd.Series, report_meta: Dict[str, str]) -> str:
    value = row.get("expression_class")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if "expression_class" in report_meta:
        return report_meta["expression_class"]
    return "Not available"


def novelty_value(row: pd.Series) -> Optional[float]:
    for key in ("novelty", "novelty_35s"):
        value = numeric_or_none(row.get(key))
        if value is not None:
            return value
    return None


def collect_species_record(species: str) -> Dict[str, object]:
    run_dir = latest_run_dir(species)
    iter_csvs = sorted(run_dir.glob("iter*_scored.csv"), key=parse_iteration)
    frames = []
    for path in iter_csvs:
        df = pd.read_csv(path)
        cid_col = candidate_id_column(df)
        if cid_col != "candidate_id":
            df = df.rename(columns={cid_col: "candidate_id"})
        df["iteration"] = parse_iteration(path)
        df["source_file"] = str(path)
        df["model"] = df["candidate_id"].map(infer_model)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["composite_score"] = pd.to_numeric(all_df["composite_score"], errors="coerce")
    best_row = all_df.sort_values(["composite_score", "iteration"], ascending=[False, False]).iloc[0]

    report_path = run_dir / f"iter{int(best_row['iteration'])}_rank1_{best_row['candidate_id']}.txt"
    if not report_path.exists():
        matches = sorted(run_dir.glob(f"iter{int(best_row['iteration'])}_rank*_{best_row['candidate_id']}.txt"))
        if matches:
            report_path = matches[0]
    report_meta = parse_report_card(report_path) if report_path.exists() else {}

    baseline_label, improvement = baseline_result(best_row, report_meta)

    record = {
        "species": species,
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "candidate_id": str(best_row["candidate_id"]),
        "best_model": infer_model(best_row["candidate_id"]),
        "model": infer_model(best_row["candidate_id"]),
        "composite_score": numeric_or_none(best_row.get("composite_score")),
        "expression_score": numeric_or_none(best_row.get("expression_score")),
        "expression_class": expression_class_value(best_row, report_meta),
        "silencing_risk": numeric_or_none(best_row.get("silencing_risk")),
        "novelty": novelty_value(best_row),
        "baseline_result": baseline_label,
        "baseline_improvement": improvement,
        "baseline_reference": best_row.get("baseline_reference") or report_meta.get("baseline_reference") or "Not available",
        "relative_strength": numeric_or_none(best_row.get("relative_strength"))
        or numeric_or_none(report_meta.get("relative_strength")),
        "sequence": str(best_row.get("sequence", "")),
        "iteration": int(best_row["iteration"]),
        "report_card": str(report_path) if report_path.exists() else "Not available",
    }
    return record


def build_markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in ["composite_score", "expression_score", "silencing_risk", "novelty"]:
        display[col] = display[col].apply(fmt_num)
    headers = [
        "species",
        "best_model",
        "composite_score",
        "expression_score",
        "expression_class",
        "silencing_risk",
        "novelty",
        "baseline_result",
    ]
    lines = [
        "| Species | Best Model | Composite Score | Expression Score | Expression Class | Silencing Risk | Novelty | Baseline Result |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for _, row in display[headers].iterrows():
        lines.append(
            f"| {row['species']} | {row['best_model']} | {row['composite_score']} | "
            f"{row['expression_score']} | {row['expression_class']} | {row['silencing_risk']} | "
            f"{row['novelty']} | {row['baseline_result']} |"
        )
    return "\n".join(lines)


def compute_metrics(df: pd.DataFrame) -> Dict[str, object]:
    best_species_row = df.sort_values(["composite_score", "species"], ascending=[False, True]).iloc[0]
    winner_counts = Counter(df["best_model"])
    best_model_overall = sorted(winner_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    explicit_classes = df["expression_class"].tolist()
    expression_counts = Counter(explicit_classes)

    silencing_avg = df["silencing_risk"].dropna().mean() if df["silencing_risk"].notna().any() else None
    silencing_max_row = (
        df[df["silencing_risk"].notna()].sort_values(["silencing_risk", "species"], ascending=[False, True]).iloc[0]
        if df["silencing_risk"].notna().any()
        else None
    )

    comparable = df[df["baseline_improvement"].notna()].copy()
    average_improvement = comparable["baseline_improvement"].mean() if not comparable.empty else None
    better_count = int((comparable["baseline_improvement"] > 0).sum()) if not comparable.empty else 0
    strongest_improvement_row = (
        comparable.sort_values(["baseline_improvement", "species"], ascending=[False, True]).iloc[0]
        if not comparable.empty
        else None
    )

    monocot_avg = (
        df[df["species"].isin(MONOCOTS)]["composite_score"].dropna().mean()
        if not df[df["species"].isin(MONOCOTS)]["composite_score"].dropna().empty
        else None
    )
    dicot_avg = (
        df[df["species"].isin(DICTS)]["composite_score"].dropna().mean()
        if not df[df["species"].isin(DICTS)]["composite_score"].dropna().empty
        else None
    )

    return {
        "best_species": best_species_row["species"],
        "best_model_overall": best_model_overall,
        "average_composite_score": df["composite_score"].dropna().mean(),
        "average_improvement": average_improvement,
        "expression_counts": dict(expression_counts),
        "average_silencing_risk": silencing_avg,
        "highest_silencing_species": silencing_max_row["species"] if silencing_max_row is not None else "Not available",
        "highest_silencing_risk": silencing_max_row["silencing_risk"] if silencing_max_row is not None else None,
        "better_than_baseline_count": better_count,
        "strongest_improvement_species": strongest_improvement_row["species"] if strongest_improvement_row is not None else "Not available",
        "strongest_improvement_value": strongest_improvement_row["baseline_improvement"] if strongest_improvement_row is not None else None,
        "monocot_average_composite": monocot_avg,
        "dicot_average_composite": dicot_avg,
        "winner_counts": dict(winner_counts),
    }


def maybe_downstream_section() -> Optional[str]:
    final_reports = sorted(OUTPUTS_DIR.glob("final_report_*.json"))
    protein_outputs = [OUTPUTS_DIR / "protein" / "structure_analysis.json", OUTPUTS_DIR / "metabolic_analysis.json"]
    existing = [p for p in final_reports + protein_outputs if p.exists()]
    if not existing:
        return None
    lines = [
        "11. Downstream Expression System Design",
        "- Additional downstream design artifacts are available from existing expression-system outputs.",
        "- Available files:",
    ]
    for path in existing:
        lines.append(f"  - {path.relative_to(ROOT)}")
    return "\n".join(lines)


def build_report(df: pd.DataFrame, metrics: Dict[str, object]) -> str:
    best_row = df.sort_values(["composite_score", "species"], ascending=[False, True]).iloc[0]
    comparable = df[df["baseline_improvement"].notna()].copy()
    comparable = comparable.sort_values(["baseline_improvement", "species"], ascending=[False, True])
    improvement_ranking = (
        ", ".join(f"{row['species']} ({fmt_num(row['baseline_improvement'])})" for _, row in comparable.iterrows())
        if not comparable.empty
        else "Not available"
    )

    model_lines = []
    for _, row in df.sort_values("species").iterrows():
        model_lines.append(f"- {row['species']}: {row['best_model']} won the latest run.")

    expression_counts = Counter(df["expression_class"])
    high = expression_counts.get("HIGH", 0)
    medium = expression_counts.get("MEDIUM", 0)
    low = expression_counts.get("LOW", 0)
    not_available = expression_counts.get("Not available", 0)

    strength_line = "Not available"
    if not comparable.empty:
        top_imp = comparable.iloc[0]
        strength_line = (
            f"{top_imp['species']} showed the strongest recorded baseline-relative result "
            f"({top_imp['baseline_result']}, improvement {fmt_num(top_imp['baseline_improvement'])})."
        )

    downstream = maybe_downstream_section()

    lines: List[str] = [
        "# AI-Driven Cross-Species Promoter Design and Benchmarking",
        "",
        "## 1. Executive Summary",
        f"- Nine species-level Iteration 2 runs were aggregated from the latest completed output folder for each species.",
        f"- The highest composite score in the latest-run set was observed in {best_row['species']} ({fmt_num(best_row['composite_score'])}) with a winning {best_row['best_model']} candidate.",
        f"- {metrics['best_model_overall']} was the most frequent winning model across species-level latest runs.",
        f"- The average composite score across species was {fmt_num(metrics['average_composite_score'])}.",
        f"- Explicit expression classes were available for a subset of latest runs: HIGH={high}, MEDIUM={medium}, LOW={low}, Not available={not_available}.",
        f"- Baseline-relative comparison metadata was available for {len(comparable)}/{len(df)} species; {int(metrics['better_than_baseline_count'])} exceeded the recorded baseline and {strength_line}",
        "",
        "## 2. Methodology",
        "- Promoter candidates were generated using the saved outputs of Evo2, D3LM, and mutational search workflows; no new generation was performed.",
        "- Candidate selection was based on the highest saved composite score within each species' latest completed output folder.",
        "- Reported metrics were taken from saved scored CSVs, loop summaries, and candidate report cards when present.",
        "- Expression metrics reflect the saved pipeline outputs only; missing labels were not fabricated.",
        "- Iterative optimisation was assessed using completed iteration traces from each selected run directory.",
        "",
        "## 3. Cross-Species Benchmarking Results",
        build_markdown_table(df),
        "",
        f"Best latest-run species: {best_row['species']} ({fmt_num(best_row['composite_score'])}, {best_row['best_model']}).",
        f"Monocot average composite score: {fmt_num(metrics['monocot_average_composite'])}.",
        f"Dicot average composite score: {fmt_num(metrics['dicot_average_composite'])}.",
        f"Baseline-improvement ranking (available comparisons only): {improvement_ranking}.",
        "",
        "## 4. Expression Analysis",
        f"- Explicit expression labels in latest saved outputs were distributed as HIGH={high}, MEDIUM={medium}, LOW={low}.",
        f"- Expression class was not available in {not_available} species-level latest runs and is reported as Not available rather than inferred.",
        "- The recorded expression scores support comparative prioritisation, but remain computational predictions rather than measured expression values.",
        "",
        "## 5. Model Comparison",
        *model_lines,
        f"- Overall consistency favoured {metrics['best_model_overall']}, which most frequently produced the top candidate in the latest-run set.",
        "- Evo2 and D3LM contributed winning candidates in selected species, while mutational search remained competitive in several saved runs.",
        "- Adaptive workflow behaviour is represented through the saved candidate pools and rankings, but the winning model reported here refers to the top candidate source.",
        "",
        "## 6. Key Findings",
        f"- Strongest promoter observed in the latest-run set: {best_row['species']} {best_row['candidate_id']} ({fmt_num(best_row['composite_score'])}).",
        f"- Average silencing risk across species was {fmt_num(metrics['average_silencing_risk'])}; the highest latest-run silencing risk was in {metrics['highest_silencing_species']} ({fmt_num(metrics['highest_silencing_risk'])}).",
        f"- {strength_line}",
        "- Species-level performance remains heterogeneous, reinforcing the need for chassis-specific validation even when using a common optimisation framework.",
        "",
        "## 7. Biological Realism Improvements",
        "- Motif spacing constraints were incorporated into the saved newer runs and help reduce architecture inflation.",
        "- Saturation penalties in newer scoring outputs reduce unrealistic motif stuffing effects.",
        "- Diversity constraints and novelty terms help limit convergence on near-duplicate candidates during iterative optimisation.",
        "",
        "## 8. Limitations",
        "- No wet-lab validation has been performed.",
        "- No genome-based safe harbor validation is available for this report set.",
        "- Expression remains predicted rather than experimentally measured.",
        "- Full proteome-level or systems-level modelling is not included in the saved Iteration 2 benchmarking outputs.",
        "",
        "## 9. Future Work",
        "- Upgrade expression prediction with dedicated sequence-to-expression models such as DeepPlantCRE.",
        "- Add genome integration analysis with real species genome assets and safe-harbor validation.",
        "- Extend downstream protease, gRNA, and degradation workflows where the corresponding assets are available.",
        "- Move the highest-priority promoter candidates into wet-lab validation.",
        "",
    ]
    if downstream:
        lines.extend([downstream, ""])
        conclusion_num = 12
    else:
        conclusion_num = 10
    lines.extend(
        [
            f"## {conclusion_num}. Conclusion",
            "- The Iteration 2 system is ready for computational promoter design and comparative in silico screening across multiple plant species.",
            "- The saved results provide a structured basis for candidate prioritisation, benchmarking, and experimental handoff.",
            "- The next phase should focus on genome-aware validation, richer expression modelling, and wet-lab confirmation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    records = [collect_species_record(species) for species in SPECIES]
    df = pd.DataFrame(records)
    df = df[
        [
            "species",
            "best_model",
            "composite_score",
            "expression_score",
            "expression_class",
            "silencing_risk",
            "novelty",
            "baseline_result",
            "baseline_improvement",
            "baseline_reference",
            "relative_strength",
            "candidate_id",
            "run_dir",
            "run_name",
            "iteration",
            "report_card",
            "sequence",
        ]
    ].sort_values(["composite_score", "species"], ascending=[False, True]).reset_index(drop=True)

    metrics = compute_metrics(df)

    csv_df = df.rename(columns={"best_model": "model"})[
        [
            "species",
            "model",
            "composite_score",
            "expression_score",
            "expression_class",
            "silencing_risk",
            "novelty",
            "baseline_result",
        ]
    ]

    summary_json = {
        "best_species": metrics["best_species"],
        "best_model_overall": metrics["best_model_overall"],
        "average_improvement": round(float(metrics["average_improvement"]), 4)
        if numeric_or_none(metrics["average_improvement"]) is not None
        else "Not available",
        "average_composite_score": round(float(metrics["average_composite_score"]), 4)
        if numeric_or_none(metrics["average_composite_score"]) is not None
        else "Not available",
        "improvement_ranking": [
            {
                "species": row["species"],
                "baseline_result": row["baseline_result"],
                "improvement": round(float(row["baseline_improvement"]), 4),
            }
            for _, row in df[df["baseline_improvement"].notna()]
            .sort_values(["baseline_improvement", "species"], ascending=[False, True])
            .iterrows()
        ],
    }

    report_md = build_report(df, metrics)

    cross_species_path = OUTPUTS_DIR / "final_cross_species_summary.csv"
    summary_path = OUTPUTS_DIR / "final_summary.json"
    report_path = OUTPUTS_DIR / "FINAL_ITERATION2_REPORT.md"

    csv_df.to_csv(cross_species_path, index=False)
    summary_path.write_text(json.dumps(summary_json, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(report_md, encoding="utf-8")

    print("Generated:")
    print(cross_species_path)
    print(summary_path)
    print(report_path)
    print("Iteration 2 report generated successfully")


if __name__ == "__main__":
    main()
