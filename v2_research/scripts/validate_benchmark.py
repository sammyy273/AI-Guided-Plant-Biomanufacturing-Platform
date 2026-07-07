"""
Promoter Validation & Benchmarking Script.

Scores Evo2-generated promoters against known reference promoters using
the cis_scoring module. Runs the pipeline for all species, compares to
benchmarks, and generates a comprehensive validation report.

Usage:
    python scripts/validate_benchmark.py
    python scripts/validate_benchmark.py --species nbenthamiana rice
"""

import argparse
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from modules.evaluation.cis_scoring import score_candidate, scan_cis_elements
from modules.silencing.silencing_risk import compute_silencing_risk
from modules.cross_species.species_config import load_species_config
from modules.generation.mutational_generator import generate_candidates

SEED_DIR = os.path.join(PROJECT_ROOT, "data", "promoter_seeds")
DATA_SEED_DIR = os.path.join(PROJECT_ROOT, "data", "seeds")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "validation_benchmark")

ALL_SPECIES = [
    "nbenthamiana", "ntobacum", "by2_cells", "tomato",
    "arabidopsis", "rice", "maize", "wheat", "soybean",
]


def read_fasta(path):
    if not path or not os.path.exists(path):
        return None
    seq_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                if seq_lines and line.startswith(">"):
                    break
                continue
            seq_lines.append(line.upper())
    return "".join(seq_lines).replace(" ", "") or None


# ── Reference promoters for benchmarking ──────────────────────────────────

REFERENCE_PROMOTERS = {
    "CaMV35S": {
        "path": os.path.join(SEED_DIR, "CaMV35S_promoter_835bp.fasta"),
        "strength": "Strong",
        "class": "Viral constitutive",
        "species_scope": "Universal dicot",
        "activity_pct_tsp": "10-25%",
    },
    "CaMV35S_minimal": {
        "path": os.path.join(SEED_DIR, "CaMV35S_minimal_-90bp.fasta"),
        "strength": "Weak (minimal core)",
        "class": "Viral minimal",
        "species_scope": "Universal dicot",
        "activity_pct_tsp": "1-3%",
    },
    "NOS": {
        "path": os.path.join(SEED_DIR, "NOS_promoter_307bp.fasta"),
        "strength": "Moderate",
        "class": "Bacterial",
        "species_scope": "Universal",
        "activity_pct_tsp": "3-8%",
    },
    "OsAct1": {
        "path": os.path.join(SEED_DIR, "OsAct1_promoter_1413bp.fasta"),
        "strength": "Very strong",
        "class": "Plant constitutive (monocot)",
        "species_scope": "Rice, monocots",
        "activity_pct_tsp": "15-30%",
    },
    "ZmUbi1": {
        "path": os.path.join(SEED_DIR, "ZmUbi1_promoter_1993bp.fasta"),
        "strength": "Very strong",
        "class": "Plant constitutive (monocot)",
        "species_scope": "Maize, monocots",
        "activity_pct_tsp": "15-30%",
    },
    "E8_ripening": {
        "path": os.path.join(SEED_DIR, "E8_promoter_1086bp.fasta"),
        "strength": "Strong (fruit-specific)",
        "class": "Tissue-specific",
        "species_scope": "Tomato, fruit",
        "activity_pct_tsp": "5-15% (fruit only)",
    },
    "OsUbiquitin2": {
        "path": os.path.join(SEED_DIR, "OsUbiquitin2_promoter_1719bp.fasta"),
        "strength": "Very strong",
        "class": "Plant constitutive (monocot)",
        "species_scope": "Rice, monocots",
        "activity_pct_tsp": "10-20%",
    },
    "SlUBQ": {
        "path": os.path.join(SEED_DIR, "SlUBQ_promoter_1321bp.fasta"),
        "strength": "Strong",
        "class": "Plant constitutive (dicot)",
        "species_scope": "Tomato, dicots",
        "activity_pct_tsp": "8-15%",
    },
}

SPECIES_TO_SEED_PROMOTER = {
    "nbenthamiana": "CaMV35S",
    "ntobacum": "CaMV35S",
    "by2_cells": "CaMV35S",
    "tomato": "CaMV35S",
    "arabidopsis": "CaMV35S",
    "rice": "OsAct1",
    "maize": "ZmUbi1",
    "wheat": "ZmUbi1",
    "soybean": "CaMV35S",
}


def score_promoter(sequence, species_config):
    """Full scoring of a promoter sequence."""
    scoring = score_candidate(sequence, species_config)
    cis = scan_cis_elements(sequence)
    silencing = compute_silencing_risk(sequence)

    gc = (sequence.count("G") + sequence.count("C")) / len(sequence) * 100 if sequence else 0
    length = len(sequence)

    return {
        "weighted_score": scoring.get("weighted_score", scoring.get("composite_score", 0)),
        "passed_filters": scoring.get("passed_filters", scoring.get("passes_filters", False)),
        "component_scores": scoring.get("component_scores", {}),
        "cis_counts": {k: v for k, v in cis.items() if not k.startswith("_")},
        "tata_in_zone": cis.get("_tata_in_zone", 0),
        "caat_in_zone": cis.get("_caat_in_zone", 0),
        "spacing_bonus": cis.get("_spacing_bonus", 0),
        "correct_order": cis.get("_correct_order", False),
        "silencing_risk": silencing.get("silencing_risk_score", silencing.get("risk_score", "N/A")),
        "gc_pct": round(gc, 1),
        "length_bp": length,
    }


def generate_evo2_promoter(species, seed_seq, species_config):
    """Generate best Evo2 promoter for a species."""
    variants = generate_candidates(
        species_key=species,
        seed_sequence=seed_seq,
        n_variants=20,
        species_config=species_config,
    )

    best_seq = None
    best_score = -1
    best_scoring = None
    for vid, seq in variants.items():
        scoring = score_candidate(seq, species_config)
        score = scoring.get("weighted_score", scoring.get("composite_score", 0))
        if score > best_score:
            best_score = score
            best_seq = seq
            best_scoring = scoring

    return best_seq, best_score, best_scoring


def main():
    parser = argparse.ArgumentParser(description="Promoter Validation & Benchmarking")
    parser.add_argument("--species", nargs="+", default=ALL_SPECIES,
                        help="Species to validate (default: all 9)")
    parser.add_argument("--output", default=OUTPUT_DIR,
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print("=" * 70)
    print("  PROMOTER VALIDATION & BENCHMARKING")
    print("=" * 70)

    # ── Part 1: Score all reference promoters ──────────────────────────────
    print("\n[1] Scoring reference promoters (benchmarks)...")
    ref_scores = {}
    for name, info in REFERENCE_PROMOTERS.items():
        seq = read_fasta(info["path"])
        if not seq:
            print(f"  {name}: FASTA not found, skipping")
            continue

        # Score with nbenthamiana config (dicot) and rice config (monocot)
        nb_config = load_species_config("nbenthamiana")
        rice_config = load_species_config("rice")

        nb_result = score_promoter(seq, nb_config)
        rice_result = score_promoter(seq, rice_config)

        ref_scores[name] = {
            "info": info,
            "length": len(seq),
            "dicot_score": nb_result,
            "monocot_score": rice_result,
        }
        print(f"  {name:20s} | {len(seq):5d} bp | dicot={nb_result['weighted_score']:5.1f} | "
              f"monocot={rice_result['weighted_score']:5.1f} | {info['strength']}")

    # ── Part 2: Generate and score Evo2 promoters for each species ─────────
    print(f"\n[2] Generating Evo2 promoters for {len(args.species)} species...")
    evo2_results = {}

    for species in args.species:
        print(f"\n  --- {species} ---")
        species_config = load_species_config(species)

        # Load seed
        seed_paths = [
            os.path.join(DATA_SEED_DIR, f"{species}.fasta"),
            os.path.join(PROJECT_ROOT, "configs", "seeds", f"{species}.fasta"),
        ]
        seed_seq = None
        for p in seed_paths:
            seed_seq = read_fasta(p)
            if seed_seq:
                break

        if not seed_seq:
            print(f"    No seed promoter found for {species}, skipping")
            continue

        seed_score_data = score_promoter(seed_seq, species_config)
        print(f"    Seed score: {seed_score_data['weighted_score']:.1f} "
              f"(GC={seed_score_data['gc_pct']:.1f}%, len={len(seed_seq)} bp)")

        # Generate Evo2 variants
        t0 = time.time()
        best_seq, best_score, best_scoring = generate_evo2_promoter(
            species, seed_seq, species_config
        )
        gen_time = time.time() - t0

        if best_seq:
            evo2_scored = score_promoter(best_seq, species_config)
            ref_name = SPECIES_TO_SEED_PROMOTER.get(species, "CaMV35S")
            ref_data = ref_scores.get(ref_name, {})
            ref_dicot = ref_data.get("dicot_score", {}).get("weighted_score", 0)
            ref_mono = ref_data.get("monocot_score", {}).get("weighted_score", 0)
            ref_score = ref_mono if species in ["rice", "maize", "wheat"] else ref_dicot

            improvement = ((evo2_scored["weighted_score"] - seed_score_data["weighted_score"])
                           / max(1, seed_score_data["weighted_score"]) * 100)

            evo2_results[species] = {
                "seed_score": seed_score_data,
                "evo2_score": evo2_scored,
                "improvement_vs_seed_pct": round(improvement, 1),
                "reference_promoter": ref_name,
                "reference_score": ref_score,
                "vs_reference_pct": round(
                    (evo2_scored["weighted_score"] - ref_score) / max(1, ref_score) * 100, 1
                ),
                "generation_time_s": round(gen_time, 2),
                "sequence": best_seq,
            }

            print(f"    Evo2 best:  {evo2_scored['weighted_score']:.1f} "
                  f"(+{improvement:.1f}% vs seed)")
            print(f"    Reference ({ref_name}): {ref_score:.1f} | "
                  f"Evo2 vs ref: {evo2_results[species]['vs_reference_pct']:+.1f}%")
            print(f"    TATA in-zone: {evo2_scored['tata_in_zone']} | "
                  f"CAAT in-zone: {evo2_scored['caat_in_zone']} | "
                  f"Correct order: {evo2_scored['correct_order']}")
            print(f"    Silencing risk: {evo2_scored['silencing_risk']}")
        else:
            print(f"    No valid promoter generated")

    # ── Part 3: Cross-species promoter comparison matrix ───────────────────
    print(f"\n[3] Cross-species comparison matrix...")

    # ── Part 4: Generate reports ───────────────────────────────────────────
    print(f"\n[4] Generating reports...")

    # JSON dump
    benchmark_data = {
        "reference_promoters": {
            name: {
                "strength": info["info"]["strength"],
                "class": info["info"]["class"],
                "length": info["length"],
                "dicot_score": info["dicot_score"]["weighted_score"],
                "monocot_score": info["monocot_score"]["weighted_score"],
                "dicot_tata": info["dicot_score"]["tata_in_zone"],
                "dicot_caat": info["dicot_score"]["caat_in_zone"],
            }
            for name, info in ref_scores.items()
        },
        "evo2_promoters": {
            species: {
                "seed_score": data["seed_score"]["weighted_score"],
                "evo2_score": data["evo2_score"]["weighted_score"],
                "improvement_vs_seed_pct": data["improvement_vs_seed_pct"],
                "reference_promoter": data["reference_promoter"],
                "reference_score": data["reference_score"],
                "vs_reference_pct": data["vs_reference_pct"],
                "gc_pct": data["evo2_score"]["gc_pct"],
                "length_bp": data["evo2_score"]["length_bp"],
                "tata_in_zone": data["evo2_score"]["tata_in_zone"],
                "caat_in_zone": data["evo2_score"]["caat_in_zone"],
                "correct_order": data["evo2_score"]["correct_order"],
                "silencing_risk": data["evo2_score"]["silencing_risk"],
                "generation_time_s": data["generation_time_s"],
            }
            for species, data in evo2_results.items()
        },
    }

    with open(os.path.join(args.output, "benchmark_results.json"), "w") as f:
        json.dump(benchmark_data, f, indent=2, default=str)

    # Markdown report
    md = generate_report(benchmark_data, ref_scores, evo2_results)
    with open(os.path.join(args.output, "validation_report.md"), "w") as f:
        f.write(md)

    print(f"\n  Output: {args.output}/")
    print(f"    benchmark_results.json")
    print(f"    validation_report.md")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)

    if evo2_results:
        scores = [d["evo2_score"]["weighted_score"] for d in evo2_results.values()]
        ref_scores_list = [d["reference_score"] for d in evo2_results.values()]
        improvements = [d["improvement_vs_seed_pct"] for d in evo2_results.values()]
        vs_refs = [d["vs_reference_pct"] for d in evo2_results.values()]

        print(f"\n  Evo2 promoter scores: {min(scores):.1f} - {max(scores):.1f} "
              f"(mean={np.mean(scores):.1f})")
        print(f"  Reference scores:     {min(ref_scores_list):.1f} - {max(ref_scores_list):.1f} "
              f"(mean={np.mean(ref_scores_list):.1f})")
        print(f"  Improvement vs seed:  {min(improvements):+.1f}% - {max(improvements):+.1f}%")
        print(f"  Evo2 vs reference:    {min(vs_refs):+.1f}% - {max(vs_refs):+.1f}%")
        print(f"  Evo2 ≥ reference:     {sum(1 for v in vs_refs if v >= 0)}/{len(vs_refs)} species")

        # Strength classification
        for species, data in evo2_results.items():
            score = data["evo2_score"]["weighted_score"]
            ref_score = data["reference_score"]
            if score >= 50:
                strength = "VERY STRONG"
            elif score >= 35:
                strength = "STRONG"
            elif score >= 25:
                strength = "MODERATE"
            elif score >= 15:
                strength = "WEAK"
            else:
                strength = "VERY WEAK"
            print(f"    {species:15s}: {score:5.1f} ({strength}) | "
                  f"ref={ref_score:.1f} | {data['vs_reference_pct']:+.1f}%")

    print("\n" + "=" * 70)


def generate_report(benchmark_data, ref_scores, evo2_results):
    lines = [
        "# Promoter Validation & Benchmarking Report",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Method:** Evo2-40B mutational generation + cis-element scoring",
        f"**Species tested:** {len(evo2_results)}",
        "",
        "---",
        "",
    ]

    # Section 1: Reference promoter benchmarks
    lines += [
        "## 1. Reference Promoter Benchmarks",
        "",
        "Known promoters scored with the same cis-element scoring system:",
        "",
        "| Promoter | Length | Class | Dicot Score | Monocot Score | Strength |",
        "|----------|--------|-------|-------------|---------------|----------|",
    ]
    for name, info in ref_scores.items():
        meta = info["info"]
        lines.append(
            f"| {name} | {info['length']} bp | {meta['class']} | "
            f"{info['dicot_score']['weighted_score']:.1f} | "
            f"{info['monocot_score']['weighted_score']:.1f} | {meta['strength']} |"
        )
    lines.append("")

    # Score calibration guide
    lines += [
        "### Score Calibration",
        "",
        "| Score Range | Classification | Known Example |",
        "|-------------|---------------|---------------|",
        "| 50-75 | Very strong | OsAct1, ZmUbi1 |",
        "| 35-50 | Strong | CaMV35S |",
        "| 25-35 | Moderate | NOS |",
        "| 15-25 | Weak | CaMV35S minimal |",
        "| 0-15 | Very weak / non-functional | Random DNA |",
        "",
    ]

    # Section 2: Evo2-generated promoters
    lines += [
        "## 2. Evo2-Generated Promoter Scores",
        "",
        "| Species | Seed Score | Evo2 Score | Improvement | Reference | Ref Score | vs Reference | GC% | Silencing |",
        "|---------|-----------|-----------|-------------|-----------|-----------|-------------|-----|-----------|",
    ]
    for species, data in evo2_results.items():
        lines.append(
            f"| {species} | {data['seed_score']['weighted_score']:.1f} | "
            f"{data['evo2_score']['weighted_score']:.1f} | "
            f"{data['improvement_vs_seed_pct']:+.1f}% | "
            f"{data['reference_promoter']} | {data['reference_score']:.1f} | "
            f"{data['vs_reference_pct']:+.1f}% | "
            f"{data['evo2_score']['gc_pct']:.1f}% | "
            f"{data['evo2_score']['silencing_risk']} |"
        )
    lines.append("")

    # Section 3: Promoter architecture comparison
    lines += [
        "## 3. Promoter Architecture Comparison",
        "",
        "| Species | TATA in-zone | CAAT in-zone | Correct Order | Spacing Bonus | Top Cis-Elements |",
        "|---------|-------------|-------------|---------------|---------------|------------------|",
    ]
    for species, data in evo2_results.items():
        cis = data["evo2_score"]["cis_counts"]
        top_cis = sorted(
            [(k, v) for k, v in cis.items() if v > 0],
            key=lambda x: -x[1]
        )[:4]
        cis_str = ", ".join(f"{k}({v})" for k, v in top_cis)
        lines.append(
            f"| {species} | {data['evo2_score']['tata_in_zone']} | "
            f"{data['evo2_score']['caat_in_zone']} | "
            f"{'Yes' if data['evo2_score']['correct_order'] else 'No'} | "
            f"{data['evo2_score']['spacing_bonus']:.0f} | {cis_str} |"
        )
    lines.append("")

    # Section 4: Strength classification per species
    lines += [
        "## 4. Strength Classification & Expression Prediction",
        "",
        "| Species | Evo2 Score | Strength | Expected Expression | When | Where |",
        "|---------|-----------|----------|--------------------|------|-------|",
    ]
    expression_map = {
        "VERY STRONG": ("15-30% TSP", "Constitutive (always on)", "All tissues"),
        "STRONG": ("8-15% TSP", "Constitutive (always on)", "All tissues"),
        "MODERATE": ("3-8% TSP", "Constitutive (always on)", "Most tissues"),
        "WEAK": ("1-3% TSP", "Weak constitutive", "Variable"),
        "VERY WEAK": ("<1% TSP", "Minimal", "Unreliable"),
    }
    for species, data in evo2_results.items():
        score = data["evo2_score"]["weighted_score"]
        if score >= 50:
            strength = "VERY STRONG"
        elif score >= 35:
            strength = "STRONG"
        elif score >= 25:
            strength = "MODERATE"
        elif score >= 15:
            strength = "WEAK"
        else:
            strength = "VERY WEAK"
        expr, when, where = expression_map[strength]
        lines.append(
            f"| {species} | {score:.1f} | {strength} | {expr} | {when} | {where} |"
        )
    lines.append("")

    # Section 5: Gene editing context (leaf vs seed)
    lines += [
        "## 5. Gene Editing: Leaf vs Seed Context",
        "",
        "### Where CRISPR Gene Editing Occurs",
        "",
        "| Step | Tissue | Method | Notes |",
        "|------|--------|--------|-------|",
        "| Protease KO (SDN-1) | **Leaf** (callus) | Agrobacterium + CRISPR | Leaf disc → callus → regenerate |",
        "| Protease KO (SDN-1, DNA-free) | **Leaf** (protoplast) | PEG + Cas9 RNP | Protoplast from leaf mesophyll |",
        "| Safe harbor integration (SDN-3) | **Leaf** (callus) | Agrobacterium + HDR template | Leaf disc → callus → select → regenerate |",
        "| Transient expression | **Leaf** (intact) | Agroinfiltration | No editing, no integration |",
        "| Seed-specific expression | **Seed** (mature) | N/A — expression only | Gene is already integrated; seeds just express |",
        "",
        "### What Happens in Each Tissue",
        "",
        "**In leaf (where editing happens):**",
        "- Agrobacterium delivers CRISPR construct to leaf mesophyll cells",
        "- Cas9 protein + gRNA create double-strand breaks in NbSBT1 genes",
        "- NHEJ repair creates frameshift indels → protease knockout",
        "- HDR template integrates the transgene at the safe harbor locus",
        "- Selection on antibiotic media kills unedited cells",
        "- Surviving callus regenerates into whole plants",
        "- These edited plants are then the chassis for recombinant protein production",
        "",
        "**In seed (where protein accumulates):**",
        "- NO gene editing occurs in seeds",
        "- Seeds are the **production platform** — the transgene expresses in developing seeds",
        "- Oleosin fusion partitions hyaluronidase to oil body surface in seeds",
        "- ER-retained variant concentrates hyaluronidase in protein bodies in seeds",
        "- Harvest seeds → extract oil bodies (float) → cleave with TEV → purified hyaluronidase",
        "",
        "### Summary: Editing in Leaf, Production in Seed",
        "",
        "| Phase | Tissue | What Happens |",
        "|-------|--------|-------------|",
        "| Gene editing | Leaf (callus) | CRISPR KO of proteases + transgene integration |",
        "| Plant regeneration | Callus → shoot → root | Edited cells regenerate into whole plants |",
        "| Transient production | Leaf (intact) | Agroinfiltration → 4-7 days expression |",
        "| Stable production (leaf) | Leaf (mature plant) | Transgene expressed in leaf tissue |",
        "| Stable production (seed) | Seed (mature) | Oleosin fusion → oil body extraction |",
        "",
    ]

    # Section 6: Enhancers present
    lines += [
        "## 6. Enhancers in Evo2-Generated Promoters",
        "",
        "Enhancer elements detected by cis-element scanning:",
        "",
    ]
    enhancer_elements = [
        "as1_element", "GCN4_motif", "ocs_like", "G_box", "GC_box",
        "W_box", "ABRE", "MBS", "Box_II", "GT1_motif", "DOF_site",
    ]
    lines.append("| Species | " + " | ".join(e.replace("_", " ") for e in enhancer_elements) + " |")
    lines.append("|" + "|".join(["---------"] * (len(enhancer_elements) + 1)) + "|")
    for species, data in evo2_results.items():
        cis = data["evo2_score"]["cis_counts"]
        vals = [str(cis.get(e, 0)) for e in enhancer_elements]
        lines.append(f"| {species} | " + " | ".join(vals) + " |")
    lines.append("")

    lines += [
        "### Enhancer Functions",
        "",
        "| Element | Function | Effect on Expression |",
        "|---------|----------|---------------------|",
        "| as-1 element | TGACG motif, bZIP binding | Constitutive activation (2-3x) |",
        "| GCN4 motif | Endosperm/seed enhancer | Tissue-specific boost |",
        "| G box | bZIP/MYC binding, stress-responsive | Inducible enhancement |",
        "| GC box | SP1 zinc finger binding | Constitutive activation |",
        "| W box | WRKY TF binding, defense | Pathogen-inducible |",
        "| ABRE | ABA-responsive, stress | Hormone-responsive |",
        "| DOF site | DOF TF binding, monocot-specific | Strong activation in monocots |",
        "| ocs-like | Octopine synthase element | Auxin-responsive enhancement |",
        "",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    main()
