"""
Phase 5: Predictive Scoring Engine, Multi-Objective Optimizer & Recommendation Engine

Integration pipeline that combines:
  1. Yield regression model (GBR with quantile CI)
  2. 6-objective construct optimizer with scenario weights
  3. Recommendation engine (enumerate → score → Pareto → rank → risk → report)

Loads Phase 3 + Phase 4 data, trains model, generates recommendations.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from modules.prediction.yield_regression import (
    train_yield_model, predict_yield, generate_synthetic_training_data,
)
from modules.optimization.multi_objective_v2 import (
    pareto_rank_constructs, rank_by_scenario, recommend_scenario,
)
from modules.recommendation.engine import (
    enumerate_configurations, score_all_configurations,
    generate_recommendation, generate_report,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "phase5")


# ── Data loaders ────────────────────────────────────────────────────────────────

def load_phase3_data():
    """Load Phase 3 outputs: CDS, construct, degradation, regulatory, localization."""
    phase3_dir = os.path.join(PROJECT_ROOT, "outputs", "phase3")

    # CDS optimization → CAI, GC, ribosome accessibility
    cds_df = pd.read_csv(os.path.join(phase3_dir, "cds_optimization_advanced.csv"))

    # Construct analysis → construct size, restriction sites
    construct_df = pd.read_csv(os.path.join(phase3_dir, "construct_analysis.csv"))

    # Regulatory analysis → silencing scores, spatial expression
    with open(os.path.join(phase3_dir, "advanced_regulatory_analysis.json")) as f:
        regulatory = json.load(f)

    # Localization ESM
    with open(os.path.join(phase3_dir, "localization_esm_detail.json")) as f:
        localization = json.load(f)

    # Degradation detail
    with open(os.path.join(phase3_dir, "degradation_detail.json")) as f:
        degradation = json.load(f)

    return {
        "cds": cds_df,
        "construct": construct_df,
        "regulatory": regulatory,
        "localization": localization,
        "degradation": degradation,
    }


def load_phase4_data():
    """Load Phase 4 outputs: enhancer scores, gRNA, tissue, metabolic, validation."""
    phase4_dir = os.path.join(PROJECT_ROOT, "outputs", "phase4")

    with open(os.path.join(phase4_dir, "phase4_results.json")) as f:
        results = json.load(f)

    with open(os.path.join(phase4_dir, "tissue_analysis.json")) as f:
        tissue = json.load(f)

    return {
        "enhancer": results["results"]["step1_enhancer"],
        "grna": results["results"]["step2_grna"],
        "tissue": tissue,
        "yield_matrix": results["results"]["step3_tissue"]["yield_matrix"],
        "metabolic": results["results"]["step4_metabolic"],
        "validation": results["results"]["step5_validation"],
    }


def build_pipeline_data(phase3, phase4):
    """Build per-species pipeline data dict for the recommendation engine."""
    pipeline_data = {}

    # CDS data per species
    cds_by_species = {}
    for _, row in phase3["cds"].iterrows():
        sp = row["species"]
        if sp == "nicotiana_benthamiana" or sp == "nbenthamiana":
            sp = "nbenthamiana"
        cds_by_species[sp] = row.to_dict()

    # Regulatory scores per species
    reg_scores = phase3["regulatory"]["final_scores"]

    # Yield matrix for baseline yields
    yield_matrix = phase4["yield_matrix"]

    # Enhancer scores
    enhancer = phase4["enhancer"]

    for sp in ["nbenthamiana", "rice", "tomato"]:
        cds = cds_by_species.get(sp, {})
        reg = reg_scores.get(sp, {})
        enh_sp = enhancer.get(f"construct_{sp}", {})

        # Find yield estimate from tissue analysis
        tissue_scores = phase4["tissue"].get(sp, {})
        best_tissue = None
        best_yield = 0
        for tissue_name, tissue_data in tissue_scores.items():
            if isinstance(tissue_data, dict):
                y = tissue_data.get("yield_estimate", {})
                if isinstance(y, dict):
                    yval = y.get("estimated_yield_mg_kg", 0)
                    if yval > best_yield:
                        best_yield = yval
                        best_tissue = tissue_name

        # Silencing score from enhancer tissue_scores → leaf
        leaf_data = enh_sp.get("tissue_scores", {}).get("leaf", {})
        silencing_info = leaf_data.get("silencing_risk", {})
        silencing_score = silencing_info.get("overall_risk", 0.2) if isinstance(silencing_info, dict) else 0.2

        # Degradation risk from Phase 3
        deg_data = phase3["degradation"]
        degradation_score = 0.3  # default
        if isinstance(deg_data, dict):
            degradation_score = 1.0 - deg_data.get("stability_fraction", 0.7)

        pipeline_data[sp] = {
            "promoter_score": enh_sp.get("best_score", cds.get("cds_quality_score", 0.5)),
            "cai": float(cds.get("cai", 0.8)),
            "cds_gc_pct": float(cds.get("gc_pct", 40)),
            "cds_quality_score": float(cds.get("cds_quality_score", 0.7)),
            "ribosome_accessibility": float(cds.get("ribosome_accessibility", 0.7)),
            "folding_penalty": float(cds.get("folding_penalty", 0.2)),
            "silencing_score": float(silencing_score),
            "degradation_score": float(degradation_score),
            "spatial_expression_control": reg.get("spatial_expression_control", 0.5),
            "post_transcriptional_regulation": reg.get("post_transcriptional_regulation", 0.9),
            "best_tissue": best_tissue,
            "best_yield_mg_kg": best_yield,
            "metabolic_burden": phase4["metabolic"].get(sp, {}).get("tsp_5pct", {}).get("metabolic_burden", "low"),
        }

    return pipeline_data


# ── Pipeline steps ──────────────────────────────────────────────────────────────

def step1_load_data():
    """Step 1: Load all Phase 3 + Phase 4 data."""
    print("=" * 60)
    print("STEP 1: Loading Phase 3 + Phase 4 data")
    print("=" * 60)

    phase3 = load_phase3_data()
    phase4 = load_phase4_data()

    print(f"  Phase 3 modules loaded: {list(phase3.keys())}")
    print(f"  Phase 4 modules loaded: {list(phase4.keys())}")
    print()

    return phase3, phase4


def step2_train_yield_model():
    """Step 2: Train yield regression model."""
    print("=" * 60)
    print("STEP 2: Training yield regression model (GBR + quantile CI)")
    print("=" * 60)

    training_data = generate_synthetic_training_data(n_samples=2000, random_state=42)
    model_dict = train_yield_model(training_data, random_state=42)

    metrics = model_dict["metrics"]
    print(f"  R² (log10 yield): {metrics['r2']}")
    print(f"  MAE (log10):      {metrics['mae_log10']}")
    print(f"  MAE (mg/kg):      {metrics['mae_mg_kg']}")
    print(f"  Train samples:    {metrics['n_train']}")
    print(f"  Test samples:     {metrics['n_test']}")
    print()

    print("  Feature importances (top 6):")
    for fname, fimp in list(model_dict["feature_importances"].items())[:6]:
        print(f"    {fname:>25s}: {fimp:.4f}")
    print()

    return model_dict


def step3_enumerate_configurations():
    """Step 3: Enumerate construct configurations for hyaluronidase."""
    print("=" * 60)
    print("STEP 3: Enumerating construct configurations")
    print("=" * 60)

    configs = enumerate_configurations(
        protein_sequence="",
        species_preference=None,
        tissue_preference=None,
        delivery_constraints=None,
    )

    species_counts = {}
    tissue_counts = {}
    for c in configs:
        species_counts[c["species"]] = species_counts.get(c["species"], 0) + 1
        tissue_counts[c["tissue"]] = tissue_counts.get(c["tissue"], 0) + 1

    print(f"  Total valid configurations: {len(configs)}")
    print(f"  Per species: {dict(species_counts)}")
    print(f"  Per tissue:  {dict(tissue_counts)}")
    print()

    return configs


def step4_score_configurations(configs, pipeline_data, model_dict):
    """Step 4: Score all configurations."""
    print("=" * 60)
    print("STEP 4: Scoring all configurations")
    print("=" * 60)

    scored = score_all_configurations(configs, pipeline_data, model_dict)

    if scored:
        # Show yield distribution
        yields = [c.get("yield_prediction", 0) for c in scored]
        print(f"  Scored configurations: {len(scored)}")
        print(f"  Yield range: {min(yields):.1f} - {max(yields):.1f} mg/kg")
        print(f"  Yield median: {np.median(yields):.1f} mg/kg")
    print()

    return scored


def step5_pareto_and_scenarios(scored, model_dict):
    """Step 5: Pareto optimization + scenario rankings."""
    print("=" * 60)
    print("STEP 5: Pareto ranking + scenario-based optimization")
    print("=" * 60)

    # Pareto ranking (already scored, pass None for yield_model)
    pareto = pareto_rank_constructs(scored, yield_model_dict=None)

    front1 = [c for c in pareto if c.get("pareto_front", 99) <= 1]
    front2 = [c for c in pareto if c.get("pareto_front", 99) <= 2]
    print(f"  Pareto front 1 (non-dominated): {len(front1)} configs")
    print(f"  Pareto front 1+2:               {len(front2)} configs")

    # All 4 scenarios
    scenarios = {}
    for scenario_name in ["best_for_speed", "best_for_yield", "best_for_stability", "balanced"]:
        ranked = rank_by_scenario(list(pareto), scenario_name)
        scenarios[scenario_name] = ranked
        top = ranked[0] if ranked else {}
        print(f"\n  {scenario_name:>25s} — Top pick:")
        print(f"    {top.get('species', '?')} / {top.get('tissue', '?')} / "
              f"{top.get('promoter_class', '?')} / {top.get('localization', '?')}")
        print(f"    Score: {top.get('scenario_score', 0):.4f}  "
              f"Yield: {top.get('yield_prediction', 0):.1f} mg/kg")
    print()

    return pareto, scenarios


def step6_generate_recommendation(configs, pipeline_data, model_dict):
    """Step 6: Generate full recommendation with risk profiles."""
    print("=" * 60)
    print("STEP 6: Generating recommendation report")
    print("=" * 60)

    recommendation = generate_recommendation(
        protein_sequence="",
        species_preference=None,
        tissue_preference=None,
        delivery_constraints=None,
        scenario=None,  # auto-select
        yield_model_dict=model_dict,
        pipeline_data=pipeline_data,
    )

    print(f"  Auto-selected scenario: {recommendation.get('scenario', '?')}")
    print(f"  Configurations evaluated: {recommendation.get('n_configurations', 0)}")
    print(f"  Pareto-optimal: {recommendation.get('pareto_front_size', 0)}")

    top3 = recommendation.get("top3", [])
    if top3:
        best = top3[0]
        print(f"\n  TOP RECOMMENDATION:")
        print(f"    Species:      {best.get('species', '?').upper()}")
        print(f"    Tissue:       {best.get('tissue', '?')}")
        print(f"    Promoter:     {best.get('promoter_class', '?')}")
        print(f"    Localization: {best.get('localization', '?')}")
        print(f"    Delivery:     {best.get('delivery', '?')}")
        print(f"    Score:        {best.get('scenario_score', 0):.4f}")
        yield_pred = best.get("yield_prediction", 0)
        print(f"    Yield:        {yield_pred:.1f} mg/kg")

    # Risk profiles
    profiles = recommendation.get("risk_profiles", [])
    if profiles:
        rp = profiles[0]
        print(f"\n  RISK ASSESSMENT (top pick): {rp['overall_risk']}")
        for rf in rp["risk_factors"]:
            print(f"    {rf['factor']}: {rf['level']} → {rf['mitigation']}")
    print()

    return recommendation


def step7_save_outputs(recommendation, model_dict, scenarios, scored):
    """Step 7: Save all outputs to outputs/phase5/."""
    print("=" * 60)
    print("STEP 7: Saving outputs to outputs/phase5/")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Complete results JSON
    results = {
        "phase": 5,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenario": recommendation.get("scenario"),
        "n_configurations": recommendation.get("n_configurations"),
        "pareto_front_size": recommendation.get("pareto_front_size"),
        "top3": recommendation.get("top3"),
        "risk_profiles": recommendation.get("risk_profiles"),
        "yield_model_metrics": model_dict["metrics"],
        "feature_importances": model_dict["feature_importances"],
    }
    with open(os.path.join(OUTPUT_DIR, "phase5_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("  phase5_results.json")

    # 2. Summary text
    summary_lines = [
        "PHASE 5: PREDICTIVE ENGINE & RECOMMENDATION RESULTS",
        "=" * 60,
        "",
        f"Scenario: {recommendation.get('scenario', '?')}",
        f"Configurations evaluated: {recommendation.get('n_configurations', 0)}",
        f"Pareto-optimal configurations: {recommendation.get('pareto_front_size', 0)}",
        "",
    ]

    top3 = recommendation.get("top3", [])
    for i, c in enumerate(top3):
        summary_lines.append(f"  Rank {i+1}:")
        summary_lines.append(f"    Species: {c.get('species', '?')}")
        summary_lines.append(f"    Tissue: {c.get('tissue', '?')}")
        summary_lines.append(f"    Promoter: {c.get('promoter_class', '?')}")
        summary_lines.append(f"    Localization: {c.get('localization', '?')}")
        summary_lines.append(f"    Delivery: {c.get('delivery', '?')}")
        summary_lines.append(f"    Score: {c.get('scenario_score', 0):.4f}")
        summary_lines.append(f"    Yield: {c.get('yield_prediction', 0):.1f} mg/kg")
        summary_lines.append("")

    # Model metrics
    metrics = model_dict["metrics"]
    summary_lines.append(f"Yield Model Metrics:")
    summary_lines.append(f"  R²: {metrics['r2']}")
    summary_lines.append(f"  MAE (mg/kg): {metrics['mae_mg_kg']}")
    summary_lines.append("")

    with open(os.path.join(OUTPUT_DIR, "phase5_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))
    print("  phase5_summary.txt")

    # 3. Recommendation report
    report = generate_report(recommendation)
    with open(os.path.join(OUTPUT_DIR, "recommendation_report.txt"), "w") as f:
        f.write(report)
    print("  recommendation_report.txt")

    # 4. Configuration scores CSV
    if scored:
        df = pd.DataFrame(scored)
        csv_cols = [c for c in [
            "species", "tissue", "promoter_class", "localization", "delivery",
            "scenario_score", "pareto_front", "balanced_composite", "yield_prediction",
        ] if c in df.columns]
        df[csv_cols].to_csv(os.path.join(OUTPUT_DIR, "configuration_scores.csv"), index=False)
        print("  configuration_scores.csv")

    # 5. Pareto front CSV
    pareto_configs = [c for c in recommendation.get("all_ranked", []) if c.get("pareto_front", 99) <= 2]
    if pareto_configs:
        df_pareto = pd.DataFrame(pareto_configs)
        csv_cols = [c for c in [
            "species", "tissue", "promoter_class", "localization", "delivery",
            "pareto_front", "balanced_composite", "scenario_score", "yield_prediction",
        ] if c in df_pareto.columns]
        df_pareto[csv_cols].to_csv(os.path.join(OUTPUT_DIR, "pareto_front.csv"), index=False)
        print("  pareto_front.csv")

    # 6. Yield model metrics
    with open(os.path.join(OUTPUT_DIR, "yield_model_metrics.json"), "w") as f:
        json.dump({
            "metrics": model_dict["metrics"],
            "feature_importances": model_dict["feature_importances"],
        }, f, indent=2)
    print("  yield_model_metrics.json")

    # 7. Scenario rankings
    scenario_data = {}
    for sname, sranked in scenarios.items():
        top_entries = []
        for c in sranked[:5]:
            entry = {k: c.get(k) for k in [
                "species", "tissue", "promoter_class", "localization", "delivery",
                "scenario_score", "scenario_rank", "yield_prediction", "pareto_front",
            ]}
            entry["objectives"] = c.get("objectives", {})
            top_entries.append(entry)
        scenario_data[sname] = top_entries

    with open(os.path.join(OUTPUT_DIR, "scenario_rankings.json"), "w") as f:
        json.dump(scenario_data, f, indent=2, default=str)
    print("  scenario_rankings.json")

    # 8. Risk profiles
    with open(os.path.join(OUTPUT_DIR, "risk_profiles.json"), "w") as f:
        json.dump(recommendation.get("risk_profiles", []), f, indent=2, default=str)
    print("  risk_profiles.json")

    print(f"\n  All outputs saved to {OUTPUT_DIR}")
    print()


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print()
    print("=" * 60)
    print("PHASE 5: PREDICTIVE ENGINE & RECOMMENDATION PIPELINE")
    print("=" * 60)
    print()

    # Step 1: Load data
    phase3, phase4 = step1_load_data()
    pipeline_data = build_pipeline_data(phase3, phase4)

    # Step 2: Train yield model
    model_dict = step2_train_yield_model()

    # Step 3: Enumerate configurations
    configs = step3_enumerate_configurations()

    # Step 4: Score all configurations
    scored = step4_score_configurations(configs, pipeline_data, model_dict)

    # Step 5: Pareto + scenario rankings
    pareto, scenarios = step5_pareto_and_scenarios(scored, model_dict)

    # Step 6: Generate recommendation
    recommendation = step6_generate_recommendation(configs, pipeline_data, model_dict)

    # Step 7: Save outputs
    step7_save_outputs(recommendation, model_dict, scenarios, scored)

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"PHASE 5 COMPLETE — {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
