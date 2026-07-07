"""
Run Full Benchmark Pipeline — Multi-Protein Assessment.

Runs the ConstructOptimizer orchestrator on all validation benchmark
proteins and produces a comparative report.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "benchmark_pipeline")


def get_benchmark_proteins():
    """Collect all benchmark proteins from validation datasets."""
    from benchmarks.validation_benchmarks import (
        GLYCOSYLATION_BENCHMARKS,
        SOLUBILITY_BENCHMARKS,
    )

    proteins = {}
    for bm in GLYCOSYLATION_BENCHMARKS:
        name = bm["protein_name"]
        if name not in proteins:
            proteins[name] = {
                "name": name,
                "sequence": bm["sequence"],
                "species": "nbenthamiana",
                "localization": "ER_retained",
            }

    for bm in SOLUBILITY_BENCHMARKS:
        name = bm["protein_name"]
        if name not in proteins:
            proteins[name] = {
                "name": name,
                "sequence": bm["sequence"],
                "species": "nbenthamiana",
                "localization": "ER_retained",
            }

    return list(proteins.values())


def main():
    t0 = time.time()
    print()
    print("=" * 60)
    print("FULL BENCHMARK PIPELINE — Multi-Protein Assessment")
    print("=" * 60)
    print()

    proteins = get_benchmark_proteins()
    print(f"Running {len(proteins)} benchmark proteins through full pipeline:")
    for p in proteins:
        print(f"  - {p['name']} ({len(p['sequence'])} aa)")
    print()

    from modules.orchestration.construct_optimizer import MultiProteinRunner

    runner = MultiProteinRunner(output_dir=OUTPUT_DIR)
    results = runner.run_benchmark_suite(proteins)

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"BENCHMARK PIPELINE COMPLETE — {elapsed:.1f}s")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print()

    return results


if __name__ == "__main__":
    main()
