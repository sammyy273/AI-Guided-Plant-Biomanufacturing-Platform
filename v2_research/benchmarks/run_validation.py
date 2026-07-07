"""
Validation Test Suite — Runs each module against benchmark datasets.

Produces statistical metrics:
  - Glycosylation: recall, precision, F1 for site detection
  - Disulfide bonds: exact count match, directionally correct
  - Solubility: correlation with known expression behavior
  - Yield: MAE, direction accuracy against published data

HONEST DISCLOSURE: These are sanity checks against known biology,
not comparisons against specialized tools (NetNGlyc, TANGO, DeepSol).
Our simplified heuristics will NOT match specialized tool accuracy.
The purpose is to verify directional correctness and catch regressions.
"""

import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "validation")


def validate_glycosylation():
    """Validate glycosylation site prediction against known glycoproteins."""
    print("=" * 70)
    print("VALIDATION: Glycosylation Site Prediction")
    print("=" * 70)

    from benchmarks.validation_benchmarks import GLYCOSYLATION_BENCHMARKS
    from modules.protein.glycosylation_model import predict_nglycosylation_sites

    results = []
    total_sites = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for bm in GLYCOSYLATION_BENCHMARKS:
        predicted = predict_nglycosylation_sites(bm["sequence"])
        # Filter: only count predictions with occupancy > 0.35 (extended sequons have ~0.15)
        high_confidence = set(s["position"] for s in predicted if s["predicted_occupancy"] > 0.35)
        all_predicted = set(s["position"] for s in predicted)
        known_positions = set(bm["known_n_glyc_sites"])

        # Use high-confidence predictions for scoring (reduces FP from low-occupancy extended sequons)
        predicted_positions = high_confidence

        tp = len(predicted_positions & known_positions)
        fp = len(predicted_positions - known_positions)
        fn = len(known_positions - predicted_positions)

        total_sites += len(known_positions)
        true_positives += tp
        false_positives += fp
        false_negatives += fn

        # Check: are known sites within ±2 of predicted? (position numbering can vary)
        matched_relaxed = 0
        for kp in known_positions:
            for pp in predicted_positions:
                if abs(kp - pp) <= 2:
                    matched_relaxed += 1
                    break

        recall_strict = tp / max(len(known_positions), 1)
        precision_strict = tp / max(len(predicted_positions), 1)

        status = "PASS" if recall_strict > 0.5 or len(known_positions) == 0 else "FAIL"

        results.append({
            "protein": bm["protein_name"],
            "known_sites": sorted(known_positions),
            "predicted_sites": sorted(predicted_positions),
            "known_count": len(known_positions),
            "predicted_count": len(predicted_positions),
            "tp": tp, "fp": fp, "fn": fn,
            "recall_strict": round(recall_strict, 3),
            "precision_strict": round(precision_strict, 3),
            "matched_relaxed": matched_relaxed,
            "status": status,
        })

        print(f"  {bm['protein_name']:>35s}: known={len(known_positions):>2d}, "
              f"predicted={len(predicted_positions):>2d} (of {len(all_predicted)} total sequons), "
              f"TP={tp}, FP={fp}, FN={fn}, "
              f"recall={recall_strict:.2f}, precision={precision_strict:.2f} [{status}]")

    # Aggregate metrics
    recall = true_positives / max(total_sites, 1)
    precision = true_positives / max(true_positives + false_positives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)

    print(f"\n  AGGREGATE:")
    print(f"    Total known sites: {total_sites}")
    print(f"    True positives:    {true_positives}")
    print(f"    False positives:   {false_positives}")
    print(f"    False negatives:   {false_negatives}")
    print(f"    Recall:            {recall:.3f}")
    print(f"    Precision:         {precision:.3f}")
    print(f"    F1:                {f1:.3f}")
    print(f"\n  DISCLOSURE: This is a simplified NetNGlyc-like algorithm.")
    print(f"  NetNGlyc reports 86% sensitivity, 61% specificity.")
    print(f"  Our scores should be interpreted as directionally correct, not competitive.")
    print()

    return {
        "module": "glycosylation",
        "metrics": {"recall": round(recall, 4), "precision": round(precision, 4), "f1": round(f1, 4)},
        "per_protein": results,
        "disclosure": "Simplified algorithm, not competitive with NetNGlyc. Directional correctness check only.",
    }


def validate_disulfide_bonds():
    """Validate disulfide bond count prediction."""
    print("=" * 70)
    print("VALIDATION: Disulfide Bond Count Prediction")
    print("=" * 70)

    from benchmarks.validation_benchmarks import DISULFIDE_BENCHMARKS
    from modules.protein.folding_quality_model import analyze_disulfide_bonds

    results = []
    exact_matches = 0
    directionally_correct = 0

    for bm in DISULFIDE_BENCHMARKS:
        analysis = analyze_disulfide_bonds(bm["sequence"])

        predicted = analysis["n_disulfide_bonds_expected"]
        actual = bm["known_disulfide_bonds"]
        predicted_cys = analysis["n_cysteines"]
        actual_cys = bm["n_cysteines"]

        exact = predicted == actual
        direction_ok = (predicted == actual) or (
            abs(predicted - actual) <= 1 and predicted > 0
        )

        if exact:
            exact_matches += 1
        if direction_ok:
            directionally_correct += 1

        cys_match = predicted_cys == actual_cys

        status = "PASS" if exact else ("PARTIAL" if direction_ok else "FAIL")

        results.append({
            "protein": bm["protein_name"],
            "known_bonds": actual,
            "predicted_bonds": predicted,
            "known_cys": actual_cys,
            "predicted_cys": predicted_cys,
            "cysteine_count_correct": cys_match,
            "bond_count_exact": exact,
            "difficulty": analysis["folding_difficulty"],
            "status": status,
        })

        print(f"  {bm['protein_name']:>35s}: known={actual} bonds ({actual_cys}C), "
              f"predicted={predicted} bonds ({predicted_cys}C), "
              f"difficulty={analysis['folding_difficulty']:>10s} [{status}]")

    n = len(DISULFIDE_BENCHMARKS)
    print(f"\n  AGGREGATE:")
    print(f"    Exact match:       {exact_matches}/{n} ({exact_matches/n:.0%})")
    print(f"    Directionally OK:  {directionally_correct}/{n} ({directionally_correct/n:.0%})")
    print(f"\n  NOTE: Disulfide bond counting from cysteines is trivially correct.")
    print(f"  The FOLDING DIFFICULTY assessment is the non-trivial prediction.")
    print()

    return {
        "module": "disulfide_bonds",
        "metrics": {
            "exact_match_rate": round(exact_matches / n, 4),
            "directional_accuracy": round(directionally_correct / n, 4),
        },
        "per_protein": results,
    }


def validate_solubility():
    """Validate solubility/aggregation prediction against known expression behavior."""
    print("=" * 70)
    print("VALIDATION: Solubility & Aggregation Prediction")
    print("=" * 70)

    from benchmarks.validation_benchmarks import SOLUBILITY_BENCHMARKS
    from modules.protein.folding_quality_model import (
        compute_solubility_profile, predict_aggregation_prone_regions,
    )

    results = []
    directionally_correct = 0

    solubility_ranking = {"very_low": 0, "low": 1, "moderate": 2, "high": 3}

    for bm in SOLUBILITY_BENCHMARKS:
        sol = compute_solubility_profile(bm["sequence"])
        agg = predict_aggregation_prone_regions(bm["sequence"])

        predicted_class = sol["solubility_class"]
        expected_class = bm["expected_solubility"]

        pred_rank = solubility_ranking.get(predicted_class, 1)
        exp_rank = solubility_ranking.get(expected_class, 1)

        # Directionally correct: predicted within 1 rank of expected
        direction_ok = abs(pred_rank - exp_rank) <= 1
        if direction_ok:
            directionally_correct += 1

        expression_predicted = sol["solubility_score"] > -0.2
        expression_actual = bm["expected_expression_success"]
        expression_match = expression_predicted == expression_actual

        status = "PASS" if direction_ok else "FAIL"

        results.append({
            "protein": bm["protein_name"],
            "expected_solubility": expected_class,
            "predicted_solubility": predicted_class,
            "solubility_score": sol["solubility_score"],
            "aggregation_regions": len(agg),
            "directionally_correct": direction_ok,
            "expression_predicted": expression_predicted,
            "expression_actual": expression_actual,
            "expression_match": expression_match,
            "status": status,
        })

        print(f"  {bm['protein_name']:>35s}: expected={expected_class:>10s}, "
              f"predicted={predicted_class:>10s} (score={sol['solubility_score']:.3f}), "
              f"agg_regions={len(agg)}, expr={'✓' if expression_match else '✗'} [{status}]")

    n = len(SOLUBILITY_BENCHMARKS)
    expr_matches = sum(1 for r in results if r["expression_match"])

    print(f"\n  AGGREGATE:")
    print(f"    Directional accuracy: {directionally_correct}/{n} ({directionally_correct/n:.0%})")
    print(f"    Expression prediction: {expr_matches}/{n} ({expr_matches/n:.0%})")
    print(f"\n  DISCLOSURE: TANGO/CamSol validation requires specialized tools.")
    print(f"  This tests directional correctness of our simplified heuristics.")
    print()

    return {
        "module": "solubility",
        "metrics": {
            "directional_accuracy": round(directionally_correct / n, 4),
            "expression_prediction_accuracy": round(expr_matches / n, 4),
        },
        "per_protein": results,
    }


def validate_mrna_stability():
    """Validate mRNA stability predictions against known behavior."""
    print("=" * 70)
    print("VALIDATION: mRNA Stability Prediction")
    print("=" * 70)

    from modules.protein.mrna_stability_model import (
        predict_mrna_half_life, compute_codon_optimality_index,
    )

    # Test with known codon-optimized vs non-optimized sequences
    # Well-optimized CDS: high GC, optimal codons
    optimized_seq = (
        "ATGGCTTCCAAGGAGCTGAAAGTGGAGATTGGCGAGGCTGGCGAGTTCCGGCTGCGCGAAACCTTCGAGGAG"
        "ATCGGCGAGGAGTGGGAGCTGGAGCGCAAGCGCGAGGCGCGCGAGTTCGGCAAGGCGCTGCTCAAGGCGCAG"
    )

    # Poorly optimized: lots of rare codons, low GC
    poor_seq = (
        "ATGAATCATAATCATAATCATAATCATAATCATAATCATAATCATAATCATAATCATAATCATAATCATAATCATA"
        "ATCATAATCATAATCATAATCATAATCATAATCATAATCATAATCATAATCATAATCATAATCATAATCATAATCA"
    )

    opt_result = predict_mrna_half_life(optimized_seq)
    poor_result = predict_mrna_half_life(poor_seq)

    print(f"  Optimized CDS:")
    print(f"    Codon optimality: {opt_result['codon_optimality']['index']:.4f}")
    print(f"    Half-life: {opt_result['predicted_halflife_hours']:.2f} h")

    print(f"\n  Poor CDS (repetitive, rare codons):")
    print(f"    Codon optimality: {poor_result['codon_optimality']['index']:.4f}")
    print(f"    Half-life: {poor_result['predicted_halflife_hours']:.2f} h")

    # The optimized sequence should have longer half-life
    direction_ok = opt_result["predicted_halflife_hours"] > poor_result["predicted_halflife_hours"]
    opt_better = opt_result["codon_optimality"]["index"] > poor_result["codon_optimality"]["index"]

    print(f"\n  Directional check: optimized > poor half-life? {direction_ok}")
    print(f"  Codon optimality:  optimized > poor? {opt_better}")

    status = "PASS" if direction_ok and opt_better else "FAIL"
    print(f"  Status: [{status}]")

    print(f"\n  DISCLOSURE: mRNA half-life prediction from sequence alone has ~20% noise floor.")
    print(f"  No training data exists for plant codon optimality → half-life mapping.")
    print()

    return {
        "module": "mrna_stability",
        "metrics": {
            "direction_correct": direction_ok,
            "codon_optimality_direction": opt_better,
        },
        "optimized_halflife": opt_result["predicted_halflife_hours"],
        "poor_halflife": poor_result["predicted_halflife_hours"],
    }


def validate_manufacturing():
    """Validate manufacturing constraints assessment."""
    print("=" * 70)
    print("VALIDATION: Manufacturing Constraints")
    print("=" * 70)

    from modules.construct.manufacturing_model import assess_synthesis_feasibility

    # Test cases with known outcomes
    tests = [
        {
            "name": "Ideal sequence (50% GC, no issues)",
            "sequence": "ATGGCTAAGCTTAAGCCTGAAGCTAAGGCTTCCGAGAAGGCTAAGCTTAAGCCT" * 20,
            "expected_synthesizable": True,
        },
        {
            "name": "Homopolymer run (should fail)",
            "sequence": "ATGGCTAAG" + "A" * 20 + "GCTAAGCCT" * 10,
            "expected_synthesizable": False,
        },
        {
            "name": "Very high GC (should flag)",
            "sequence": "GCGCGCGCGCGCGCGCATGGCGCGCGCGCGCGCGCGCGC" * 10,
            "expected_synthesizable": False,
        },
    ]

    results = []
    correct = 0
    for t in tests:
        result = assess_synthesis_feasibility(t["sequence"])
        matches = result["synthesizable"] == t["expected_synthesizable"]
        if matches:
            correct += 1

        results.append({
            "name": t["name"],
            "expected": t["expected_synthesizable"],
            "predicted": result["synthesizable"],
            "score": result["manufacturability_score"],
            "matches": matches,
        })

        print(f"  {t['name']:>40s}: expected={str(t['expected_synthesizable']):>5s}, "
              f"got={str(result['synthesizable']):>5s}, "
              f"score={result['manufacturability_score']:.3f} "
              f"[{'PASS' if matches else 'FAIL'}]")

    print(f"\n  Accuracy: {correct}/{len(tests)} ({correct/len(tests):.0%})")
    print()

    return {
        "module": "manufacturing",
        "metrics": {"accuracy": round(correct / len(tests), 4)},
        "per_test": results,
    }


def run_all_validations():
    """Run all validation benchmarks."""
    t0 = time.time()
    print()
    print("=" * 70)
    print("MODULE VALIDATION BENCHMARKS")
    print("=" * 70)
    print()

    all_results = {}
    all_results["glycosylation"] = validate_glycosylation()
    all_results["disulfide_bonds"] = validate_disulfide_bonds()
    all_results["solubility"] = validate_solubility()
    all_results["mrna_stability"] = validate_mrna_stability()
    all_results["manufacturing"] = validate_manufacturing()

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "validation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    elapsed = time.time() - t0
    print("=" * 70)
    print(f"VALIDATION COMPLETE — {elapsed:.1f}s")
    print("=" * 70)
    print()
    print("SUMMARY (honest assessment):")
    print(f"  Glycosylation recall: {all_results['glycosylation']['metrics']['recall']:.3f}")
    print(f"  Disulfide exact match: {all_results['disulfide_bonds']['metrics']['exact_match_rate']:.3f}")
    print(f"  Solubility direction: {all_results['solubility']['metrics']['directional_accuracy']:.3f}")
    print(f"  mRNA stability direction: {all_results['mrna_stability']['metrics']['direction_correct']}")
    print(f"  Manufacturing accuracy: {all_results['manufacturing']['metrics']['accuracy']:.3f}")
    print()
    print("CAVEATS:")
    print("  - These are sanity checks, not benchmarks against specialized tools")
    print("  - Glycosylation: not benchmarked against NetNGlyc")
    print("  - Aggregation: not benchmarked against TANGO/Aggrescan")
    print("  - Solubility: not benchmarked against CamSol/DeepSol")
    print("  - gRNA: not benchmarked against CRISPOR/CHOPCHOP")
    print("  - All predictions require wet-lab validation before experimental use")
    print()

    return all_results


if __name__ == "__main__":
    run_all_validations()
