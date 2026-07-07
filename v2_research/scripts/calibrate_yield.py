"""
Yield Calibration Runner.

Orchestrates the full biological calibration pipeline:
1. Unified yield model calibration (component multipliers via optimization)
2. Yield regression retraining (experimental + synthetic data)
3. Benchmark validation (hold-out yield records)
4. Chassis validation (edit recommendation consistency)
5. Experimental success validation (go/no-go accuracy)

Usage:
    cd v2_research/
    python scripts/calibrate_yield.py
"""

import sys
import os
import json
import time

# Ensure modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_full_calibration():
    results = {}
    start = time.time()

    # ── Step 1: Unified Yield Model Calibration ────────────────────────────
    print("=" * 70)
    print("STEP 1: Unified Yield Model Calibration")
    print("=" * 70)

    from modules.prediction.unified_yield_model import (
        calibrate_from_experimental,
        validate_yield_calibration,
        apply_fitted_multipliers,
    )

    cal_result = calibrate_from_experimental()
    results["unified_calibration"] = cal_result

    print(f"\nFitted multipliers:")
    if "grouped_multipliers" in cal_result:
        print(f"  Grouped (sequence / protein / construct):")
        for k, v in cal_result["grouped_multipliers"].items():
            print(f"    {k:15s}: {v:6.3f}")
    print(f"  Per-component:")
    for k, v in cal_result["multipliers"].items():
        marker = " *" if abs(v - 1.0) > 0.1 else ""
        print(f"    {k:35s}: {v:6.3f}{marker}")

    print(f"\nFit metrics:")
    print(f"  R² (log-yield):  {cal_result['r2_log_yield']:.4f}")
    print(f"  Median fold-error: {cal_result['median_fold_error']:.2f}x")
    print(f"  Mean fold-error:   {cal_result['mean_fold_error']:.2f}x")
    print(f"  Max fold-error:    {cal_result['max_fold_error']:.2f}x")

    print(f"\nPer-record comparison:")
    print(f"  {'Protein':25s} {'Vector':15s} {'Obs':>8s} {'Pred':>8s} {'Fold':>6s}")
    print(f"  {'-'*25} {'-'*15} {'-'*8} {'-'*8} {'-'*6}")
    for c in cal_result["comparisons"]:
        print(f"  {c['protein']:25s} {c['vector']:15s} "
              f"{c['observed']:8.1f} {c['predicted']:8.1f} {c['fold_error']:6.1f}x")

    # Apply fitted multipliers and correction factor
    cf = cal_result.get("correction_factor", 1.0)
    apply_fitted_multipliers(cal_result["multipliers"], correction_factor=cf)
    print(f"\nMultipliers and correction factor ({cf:.2f}x) applied.")

    # ── Step 2: Leave-One-Out Validation ────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Leave-One-Out Cross-Validation")
    print("=" * 70)

    loo_result = validate_yield_calibration()
    results["loo_validation"] = loo_result

    print(f"\nLOO metrics (n={loo_result['n_folds']}):")
    print(f"  Median fold-error:    {loo_result['median_fold_error']:.2f}x")
    print(f"  Mean fold-error:      {loo_result['mean_fold_error']:.2f}x")
    print(f"  Fraction within 2x:   {loo_result['fraction_within_2x']:.1%}")
    print(f"  Fraction within 5x:   {loo_result['fraction_within_5x']:.1%}")

    print(f"\nPer-fold details:")
    print(f"  {'Protein':25s} {'Vector':15s} {'Obs':>8s} {'Pred':>8s} {'Fold':>6s} {'≤2x':>5s} {'≤5x':>5s}")
    print(f"  {'-'*25} {'-'*15} {'-'*8} {'-'*8} {'-'*6} {'-'*5} {'-'*5}")
    for r in loo_result["loo_results"]:
        mark2x = "Y" if r["within_2x"] else "N"
        mark5x = "Y" if r["within_5x"] else "N"
        print(f"  {r['protein']:25s} {r['vector']:15s} "
              f"{r['observed']:8.1f} {r['predicted']:8.1f} "
              f"{r['fold_error']:6.1f}x {mark2x:>5s} {mark5x:>5s}")

    # ── Step 3: Yield Regression Retraining ─────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Yield Regression Retraining (experimental + synthetic)")
    print("=" * 70)

    from modules.prediction.yield_regression import (
        train_yield_model,
        validate_against_benchmarks,
    )

    model = train_yield_model(include_experimental=True)
    results["regression_metrics"] = model["metrics"]

    print(f"\nGBR model metrics:")
    print(f"  R²:           {model['metrics']['r2']:.4f}")
    print(f"  MAE (log10):  {model['metrics']['mae_log10']:.4f}")
    print(f"  MAE (mg/kg):  {model['metrics']['mae_mg_kg']:.1f}")
    print(f"  Train size:   {model['metrics']['n_train']}")
    print(f"  Test size:    {model['metrics']['n_test']}")

    print(f"\nTop feature importances:")
    for feat, imp in list(model["feature_importances"].items())[:6]:
        print(f"  {feat:30s}: {imp:.4f}")

    # ── Step 4: Benchmark Validation ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: Hold-out Benchmark Validation")
    print("=" * 70)

    bench_result = validate_against_benchmarks(model)
    results["benchmark_validation"] = bench_result

    print(f"\nBenchmark metrics (n={bench_result['n_benchmarks']}):")
    print(f"  Median fold-error:    {bench_result['median_fold_error']:.2f}x")
    print(f"  Mean fold-error:      {bench_result['mean_fold_error']:.2f}x")
    print(f"  Direction accuracy:   {bench_result['direction_accuracy']:.1%}")

    print(f"\nPer-benchmark predictions:")
    print(f"  {'Protein':25s} {'Species':15s} {'Obs':>8s} {'Pred':>8s} {'Fold':>6s} {'Dir':>4s}")
    print(f"  {'-'*25} {'-'*15} {'-'*8} {'-'*8} {'-'*6} {'-'*4}")
    for r in bench_result["results"]:
        dir_mark = "Y" if r["direction_correct"] else "N"
        print(f"  {r['protein']:25s} {r['species']:15s} "
              f"{r['observed_mg_kg']:8.1f} {r['predicted_mg_kg']:8.1f} "
              f"{r['fold_error']:6.1f}x {dir_mark:>4s}")

    # ── Step 5: Chassis Validation ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: Chassis Optimization Validation")
    print("=" * 70)

    from modules.chassis.chassis_optimizer import validate_chassis_recommendations

    chassis_result = validate_chassis_recommendations()
    results["chassis_validation"] = chassis_result

    print(f"\nAll tests passed: {chassis_result['all_passed']}")
    for test, value in chassis_result["tests"].items():
        status = "PASS" if value else "FAIL"
        if not isinstance(value, bool):
            status = f"{value}"
        print(f"  {test:40s}: {status}")
    if chassis_result["issues"]:
        print("Issues:")
        for issue in chassis_result["issues"]:
            print(f"  - {issue}")

    # ── Step 6: Experimental Success Validation ─────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6: Experimental Success Prediction Validation")
    print("=" * 70)

    from modules.prediction.experimental_success import run_success_calibration

    success_result = run_success_calibration()
    results["success_validation"] = success_result

    print(f"\nDirectional accuracy: {success_result['directional_accuracy']:.1%} "
          f"({success_result['n_correctly_classified']}/{success_result['n_classifiable']})")
    print(f"Spearman correlation: {success_result['spearman_correlation']:.3f}")

    print(f"\nPer-record success predictions:")
    print(f"  {'Protein':25s} {'Vector':15s} {'Obs':>8s} {'PredY':>8s} {'YCls':>6s} {'Expected':>9s} {'Pred':>9s} {'Prob':>6s}")
    print(f"  {'-'*25} {'-'*15} {'-'*8} {'-'*8} {'-'*6} {'-'*9} {'-'*9} {'-'*6}")
    for r in success_result["per_record"]:
        print(f"  {r['protein']:25s} {r['vector']:15s} "
              f"{r['observed_mg_kg']:8.1f} {r['predicted_yield_mg_kg']:8.1f} "
              f"{r['yield_class']:>6s} {r['expected_class']:>9s} "
              f"{r['predicted_go_no_go']:>9s} {r['success_probability']:6.2f}")

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"Time: {elapsed:.1f}s")
    print(f"Unified model R² (log):     {cal_result['r2_log_yield']:.4f}")
    print(f"LOO median fold-error:      {loo_result['median_fold_error']:.2f}x")
    print(f"LOO fraction within 5x:     {loo_result['fraction_within_5x']:.1%}")
    print(f"GBR R²:                     {model['metrics']['r2']:.4f}")
    print(f"Benchmark direction accuracy: {bench_result['direction_accuracy']:.1%}")
    print(f"Chassis tests passed:       {chassis_result['all_passed']}")
    print(f"Success prediction accuracy: {success_result['directional_accuracy']:.1%}")

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "calibration_results.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_full_calibration()
