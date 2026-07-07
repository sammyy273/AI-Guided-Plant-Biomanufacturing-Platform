# Evaluation Axis Correlation Matrix
#
# Computes pairwise correlations between all scoring components to detect
# double-counting (redundant metrics that inflate composite scores).
#
# Gap 8 analysis: if two axes are highly correlated (|r| > 0.7), they
# contribute redundant signal and one should be dropped or merged to
# prevent the composite score from overweighting that information.
#
# Usage:
#   python scripts/eval_correlation_matrix.py --species nbenthamiana
#   python scripts/eval_correlation_matrix.py --species nbenthamiana --n-samples 50

import argparse
import logging
import os
import sys
import json

import numpy as np
import pandas as pd

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.cross_species.species_config import load_species_config
from modules.evaluation.cis_scoring import (
    scan_cis_elements, score_candidate, gc_content,
    score_architecture, score_diversity, score_gc,
    penalty_overuse, penalty_scattered, penalty_repeats, penalty_realism,
)
from modules.silencing.silencing_risk import compute_silencing_risk
from modules.generation.mutational_generator import (
    build_species_scaffold, generate_from_seed,
)
from auto_loop_v2 import load_seed


AXIS_NAMES = [
    "architecture", "diversity", "gc_score",
    "overuse_penalty", "scattered_penalty", "repeat_penalty",
    "realism_penalty", "gc_pct", "silencing_risk",
    "binding_score", "motif_count",
    "tata_count", "caat_count", "as1_count",
]


def extract_all_axes(sequence: str, species_config: dict) -> dict:
    """Extract all evaluation axes for a single sequence."""
    counts = scan_cis_elements(sequence)
    gc = gc_content(sequence)
    weights = species_config.get("cis_element_weights", {})

    result = score_candidate(sequence, species_config)

    # Silencing risk
    try:
        silencing = compute_silencing_risk(sequence, species_config)
        silencing_val = silencing.get("silencing_risk", 0.0)
    except Exception:
        silencing_val = 0.0

    return {
        "architecture": result.get("_architecture", 0.0),
        "diversity": result.get("_diversity", 0.0),
        "gc_score": result.get("_gc_score", 0.0),
        "overuse_penalty": result.get("_overuse_penalty", 0.0),
        "scattered_penalty": result.get("_scattered_penalty", 0.0),
        "repeat_penalty": result.get("_repeat_penalty", 0.0),
        "realism_penalty": result.get("_realism_penalty", 0.0),
        "gc_pct": gc,
        "silencing_risk": silencing_val,
        "binding_score": (
            counts.get("TATA_box", 0) * 3.0
            + counts.get("CAAT_box", 0) * 2.0
            + counts.get("GC_box", 0) * 1.5
            + counts.get("as1_element", 0) * 2.5
            + counts.get("DOF_site", 0) * 2.5
        ),
        "motif_count": sum(counts.get(k, 0) for k in [
            "TATA_box", "CAAT_box", "as1_element", "GCN4_motif",
            "G_box", "W_box", "ABRE", "DOF_site",
        ]),
        "tata_count": counts.get("TATA_box", 0),
        "caat_count": counts.get("CAAT_box", 0),
        "as1_count": counts.get("as1_element", 0),
        "weighted_score": result.get("weighted_score", 0.0),
    }


def generate_sample_sequences(species_config: dict, n_samples: int = 50) -> list:
    """Generate a diverse set of sequences for correlation analysis."""
    import random

    sequences = []
    species_key = species_config.get("_config_key", "nbenthamiana")

    # 1. Generated scaffolds (designed promoters)
    for _ in range(n_samples // 2):
        seq = build_species_scaffold(species_config, 800)
        sequences.append(("generated", seq))

    # 2. Mutated variants from seed
    try:
        seed = load_seed(species_key)
        for i in range(n_samples // 4):
            for rate in [0.05, 0.10, 0.20, 0.30, 0.50]:
                seq = generate_from_seed(
                    seed, species_config, n_variants=1,
                    mutation_rate=rate,
                )
                if seq:
                    sequences.append((f"mutated_r{rate}", list(seq.values())[0]))
    except Exception:
        pass

    # 3. Random DNA at various GC levels
    for _ in range(n_samples // 4):
        for gc_target in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
            at = int(800 * (1 - gc_target) / 2)
            gc = int(800 * gc_target / 2)
            bases = ["A"] * at + ["T"] * at + ["G"] * gc + ["C"] * gc
            random.shuffle(bases)
            seq = "".join(bases[:800])
            sequences.append((f"random_gc{int(gc_target*100)}", seq))

    return sequences


def compute_correlation_matrix(df: pd.DataFrame) -> tuple:
    """Compute pairwise Pearson correlations and flag double-counting."""
    numeric_df = df.select_dtypes(include=[np.number])
    cols = [c for c in numeric_df.columns if c != "weighted_score"]
    corr_matrix = numeric_df[cols].corr(method="pearson")

    # Flag pairs with |r| > 0.7
    flagged = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.7:
                flagged.append({
                    "axis_1": cols[i],
                    "axis_2": cols[j],
                    "correlation": round(r, 4),
                    "severity": "HIGH" if abs(r) > 0.9 else "MODERATE",
                })

    return corr_matrix, flagged


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation Axis Correlation Matrix (Gap 8 Analysis)",
    )
    parser.add_argument("--species", default="nbenthamiana")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    species_config = load_species_config(args.species)
    species_config["_config_key"] = args.species

    print(f"Generating {args.n_samples} diverse sequences for {args.species}...")

    sequences = generate_sample_sequences(species_config, args.n_samples)
    print(f"  Generated {len(sequences)} sequences")

    # Extract all axes
    records = []
    for source, seq in sequences:
        try:
            axes = extract_all_axes(seq, species_config)
            axes["source"] = source
            records.append(axes)
        except Exception as e:
            print(f"  Warning: failed to score sequence ({source}): {e}")

    df = pd.DataFrame(records)
    print(f"  Scored {len(df)} sequences across {len([c for c in df.columns if c not in ['source']])} axes")

    # Compute correlation matrix
    corr_matrix, flagged = compute_correlation_matrix(df)

    print("\n" + "=" * 70)
    print("CORRELATION MATRIX (Pearson r)")
    print("=" * 70)
    print(corr_matrix.round(3).to_string())

    # Correlation with weighted_score
    print("\n" + "=" * 70)
    print("CORRELATION WITH WEIGHTED_SCORE")
    print("=" * 70)
    numeric_df = df.select_dtypes(include=[np.number])
    score_corrs = numeric_df.corrwith(numeric_df["weighted_score"], method="pearson").drop("weighted_score", errors="ignore")
    for name, r in score_corrs.abs().sort_values(ascending=False).items():
        direction = "+" if score_corrs[name] > 0 else "-"
        print(f"  {name:25s}  r = {direction}{r:.4f}")

    # Flagged pairs
    print("\n" + "=" * 70)
    print("DOUBLE-COUNTING FLAGS (|r| > 0.7)")
    print("=" * 70)
    if not flagged:
        print("  No axis pairs exceed |r| > 0.7. No double-counting detected.")
    else:
        for f in flagged:
            print(f"  [{f['severity']:8s}] {f['axis_1']:25s} <-> {f['axis_2']:25s}  "
                  f"r = {f['correlation']:+.4f}")

        # Recommendations
        print("\n  RECOMMENDATIONS:")
        seen = set()
        for f in flagged:
            pair = tuple(sorted([f["axis_1"], f["axis_2"]]))
            if pair in seen:
                continue
            seen.add(pair)
            if f["severity"] == "HIGH":
                print(f"    - MERGE or DROP one of: {f['axis_1']} / {f['axis_2']} "
                      f"(r={f['correlation']:.3f})")
            else:
                print(f"    - CONSIDER merging: {f['axis_1']} / {f['axis_2']} "
                      f"(r={f['correlation']:.3f})")

    print("=" * 70)

    # Save results
    output_dir = args.output or os.path.join(
        _project_root, "outputs",
        f"correlation_matrix_{args.species}",
    )
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "species": args.species,
        "n_samples": len(df),
        "correlation_matrix": corr_matrix.round(4).to_dict(),
        "score_correlations": {k: round(v, 4) for k, v in score_corrs.items()},
        "flagged_pairs": flagged,
        "summary": {
            "total_axes": len([c for c in df.columns if c not in ["source", "weighted_score"]]),
            "flagged_high": sum(1 for f in flagged if f["severity"] == "HIGH"),
            "flagged_moderate": sum(1 for f in flagged if f["severity"] == "MODERATE"),
        },
    }

    out_path = os.path.join(output_dir, "correlation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved: {out_path}")

    return results


if __name__ == "__main__":
    main()
