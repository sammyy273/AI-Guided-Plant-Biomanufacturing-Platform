#!/usr/bin/env python3
"""Generate a compact summary from promoter design outputs."""

import argparse
import csv
import glob
import os


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def latest_species_dir(root: str, species: str) -> str | None:
    pattern = os.path.join(root, species, "*")
    candidates = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if candidates:
        return sorted(candidates)[-1]

    # Also support direct validation directories created by validation.py
    pattern = os.path.join(root, f"validation_{species}_*")
    candidates = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if candidates:
        return sorted(candidates)[-1]
    return None


def read_best_row(species_dir: str) -> dict | None:
    scored_files = sorted(glob.glob(os.path.join(species_dir, "iter*_scored.csv")))
    if not scored_files:
        return None
    latest = scored_files[-1]
    with open(latest, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[0] if rows else None


def read_loop_summary(species_dir: str) -> dict | None:
    path = os.path.join(species_dir, "loop_summary.json")
    if not os.path.exists(path):
        return None
    import json
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compact promoter summary")
    parser.add_argument(
        "--root",
        default=os.path.join(PROJECT_DIR, "outputs"),
        help="Outputs root directory",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        required=True,
        help="Species to summarize",
    )
    args = parser.parse_args()

    print(f"{'Species':<16} | {'Best Score':>10} | {'Candidates':>10} | {'Passed':>6} | Progression")
    print("-" * 96)
    for species in args.species:
        species_dir = latest_species_dir(args.root, species)
        if not species_dir:
            print(f"{species:<16} | {'N/A':>10} | {'N/A':>10} | {'N/A':>6} | N/A")
            continue
        row = read_best_row(species_dir)
        summary = read_loop_summary(species_dir)
        if not row or not summary:
            print(f"{species:<16} | {'N/A':>10} | {'N/A':>10} | {'N/A':>6} | N/A")
            continue
        results = summary.get("results", [])
        total_cands = sum(int(r.get("n_candidates", 0)) for r in results)
        total_passed = sum(int(r.get("n_passed_filters", 0)) for r in results)
        progression = " -> ".join(
            f"{float(r['top_composite_score']):.3f}"
            for r in results if r.get("top_composite_score") is not None
        )
        best_score = row.get("composite_score", row.get("weighted_score", "N/A"))
        print(f"{species:<16} | {str(best_score):>10} | {total_cands:>10} | {total_passed:>6} | {progression}")


if __name__ == "__main__":
    main()
