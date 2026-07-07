#!/usr/bin/env python3
"""
STEP 5: Integrated System Re-Evaluation.

Combines all updated analyses into a revised system assessment.
Does NOT change previous decisions — adds "updated_assessment" column.

Combines:
  - Promoter score (existing)
  - CDS validation (existing)
  - Localization (updated from Step 2)
  - Degradation (updated from Step 3)
  - Genome context (from Step 4)

OUTPUTS:
  outputs/phase2/system_re_evaluation.csv
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "modules"))

OUTPUT_DIR = BASE_DIR / "outputs" / "phase2"


def load_json(path):
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return {}


def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 5: Integrated System Re-Evaluation")
    print("=" * 60)

    # Load existing pipeline results
    species_list = ["nbenthamiana", "rice", "tomato"]
    existing_results = {}

    for species in species_list:
        report_path = BASE_DIR / "outputs" / f"final_report_{species}.json"
        report = load_json(report_path)
        if report:
            existing_results[species] = report

    # Load Phase 2 updates
    localization_df = load_csv(OUTPUT_DIR / "localization_ml_comparison.csv")
    stability_json = load_json(OUTPUT_DIR / "protein_stability_detail.json")
    safe_harbor_df = load_csv(OUTPUT_DIR / "safe_harbor_candidates.csv")
    grounding_df = load_csv(OUTPUT_DIR / "expression_grounding.csv")

    print(f"  Loaded {len(existing_results)} existing reports")
    print(f"  Localization data: {len(localization_df)} rows")
    print(f"  Protein stability: {'loaded' if stability_json else 'NOT AVAILABLE'}")
    print(f"  Safe harbor data: {len(safe_harbor_df)} rows")

    # Process each species
    rows = []

    for species in species_list:
        if species not in existing_results:
            print(f"\n  {species}: No existing report found")
            continue

        report = existing_results[species]
        bp = report.get("best_promoter", {})
        loc_data = report.get("localization", {})
        deg_data = report.get("degradation_risk", {})
        cds_data = report.get("optimized_cds", {})
        conf_data = report.get("confidence", {})

        # Extract existing metrics
        promoter_score = bp.get("composite_score", 0)
        promoter_weighted = bp.get("weighted_score", 0)
        silencing_risk = bp.get("silencing_risk", 0)
        expression_class = bp.get("expression_class", "UNKNOWN")
        cai = cds_data.get("cai", 0)
        cds_gc = cds_data.get("gc_pct", 0)

        # Existing localization
        existing_loc = loc_data.get("predicted_localization", "unknown")
        existing_loc_method = loc_data.get("status", "unknown")
        existing_loc_conf = loc_data.get("confidence", 0)

        # Existing degradation
        existing_deg_score = deg_data.get("risk_score", 0)
        existing_deg_class = deg_data.get("risk_class", "unknown")

        # ── Updated localization ────────────────────────────────────────
        if not localization_df.empty:
            loc_row = localization_df[localization_df["method"] == "enhanced_heuristic"]
            if not loc_row.empty:
                updated_loc = loc_row.iloc[0]["prediction"]
                updated_loc_conf = loc_row.iloc[0]["confidence"]
                updated_loc_consensus = loc_row.iloc[0]["consensus"]
                updated_loc_consensus_conf = loc_row.iloc[0]["consensus_confidence"]
            else:
                updated_loc = existing_loc
                updated_loc_conf = existing_loc_conf
                updated_loc_consensus = "unchanged"
                updated_loc_consensus_conf = "NOT ASSESSED"
        else:
            updated_loc = existing_loc
            updated_loc_conf = existing_loc_conf
            updated_loc_consensus = "unchanged"
            updated_loc_consensus_conf = "NOT ASSESSED"

        # ── Updated degradation ─────────────────────────────────────────
        if stability_json:
            primary_comp = stability_json.get("primary_compartment", "unknown")
            deg_by_comp = stability_json.get("degradation_by_compartment", {})

            # Match degradation to updated localization
            loc_key = None
            for key in deg_by_comp:
                if key in updated_loc.lower() or updated_loc.lower() in key:
                    loc_key = key
                    break

            if loc_key and loc_key in deg_by_comp:
                updated_deg = deg_by_comp[loc_key]
                updated_deg_score = updated_deg["degradation_score"]
                updated_deg_class = updated_deg["risk_class"]
            else:
                updated_deg_score = existing_deg_score
                updated_deg_class = existing_deg_class
        else:
            primary_comp = "unknown"
            updated_deg_score = existing_deg_score
            updated_deg_class = existing_deg_class

        # ── Genome context ──────────────────────────────────────────────
        has_safe_harbor = False
        top_sh_score = 0
        sh_count = 0

        if not safe_harbor_df.empty:
            # Map species names to genome species
            species_genome_map = {
                "nbenthamiana": None,  # no genome data
                "rice": "rice",
                "tomato": "tomato",
            }
            genome_species = species_genome_map.get(species)
            if genome_species:
                sh_for_species = safe_harbor_df[safe_harbor_df["species"] == genome_species]
                if not sh_for_species.empty:
                    has_safe_harbor = True
                    top_sh_score = sh_for_species["safe_harbor_score"].max()
                    sh_count = len(sh_for_species)

        # ── Recompute system risk ───────────────────────────────────────
        # Components: promoter, CDS, localization, degradation, genome

        # Promoter risk (inverse of score)
        promoter_risk = 1 - min(1, promoter_score)

        # CDS risk (inverse of CAI)
        cds_risk = 1 - cai if cai > 0 else 0.5

        # Localization risk
        if isinstance(updated_loc_consensus_conf, str):
            loc_risk = 0.5 if "HIGH" == updated_loc_consensus_conf else (
                0.3 if "MEDIUM" == updated_loc_consensus_conf else 0.7
            )
        else:
            loc_risk = 1 - float(updated_loc_consensus_conf) if updated_loc_consensus_conf else 0.5

        # Degradation risk
        deg_risk = updated_deg_score

        # Genome risk
        genome_risk = 0.3 if has_safe_harbor else 0.7

        # Weighted system risk
        system_risk = (
            0.20 * promoter_risk +
            0.15 * cds_risk +
            0.25 * loc_risk +
            0.25 * deg_risk +
            0.15 * genome_risk
        )

        # Risk classification
        if system_risk >= 0.66:
            risk_class = "HIGH"
        elif system_risk >= 0.33:
            risk_class = "MEDIUM"
        else:
            risk_class = "LOW"

        # Updated deployment assessment
        blockers = []
        if deg_risk >= 0.66:
            blockers.append(f"HIGH degradation risk ({updated_deg_class})")
        if loc_risk >= 0.5:
            blockers.append("localization uncertainty")
        if silencing_risk >= 0.5:
            blockers.append("silencing risk")

        if blockers:
            updated_assessment = f"NOT DEPLOYABLE — {'; '.join(blockers)}"
            deploy_status = "NO"
        else:
            updated_assessment = "CONDITIONAL — requires experimental validation"
            deploy_status = "CONDITIONAL"

        rows.append({
            "species": species,
            # Existing scores
            "promoter_score": round(promoter_score, 4),
            "promoter_weighted_score": round(promoter_weighted, 4),
            "silencing_risk": round(silencing_risk, 4),
            "expression_class": expression_class,
            "cds_cai": round(cai, 4),
            "cds_gc_pct": round(cds_gc, 2),
            # Existing localization
            "localization_original": existing_loc,
            "localization_original_method": existing_loc_method,
            "localization_original_confidence": round(existing_loc_conf, 4),
            # Existing degradation
            "degradation_original_score": round(existing_deg_score, 4),
            "degradation_original_class": existing_deg_class,
            # Updated localization
            "localization_updated": updated_loc,
            "localization_updated_confidence": round(updated_loc_conf, 4) if isinstance(updated_loc_conf, (int, float)) else updated_loc_conf,
            "localization_consensus": updated_loc_consensus,
            "localization_consensus_confidence": updated_loc_consensus_conf,
            # Updated degradation
            "degradation_updated_score": round(updated_deg_score, 4),
            "degradation_updated_class": updated_deg_class,
            "degradation_dominant_protease": stability_json.get("dominant_protease", "unknown") if stability_json else "unknown",
            # Genome context
            "safe_harbor_available": has_safe_harbor,
            "safe_harbor_count": sh_count,
            "safe_harbor_top_score": round(top_sh_score, 2),
            # Recomputed system
            "system_risk_promoter": round(promoter_risk, 4),
            "system_risk_cds": round(cds_risk, 4),
            "system_risk_localization": round(loc_risk, 4),
            "system_risk_degradation": round(deg_risk, 4),
            "system_risk_genome": round(genome_risk, 4),
            "system_risk_total": round(system_risk, 4),
            "system_risk_class": risk_class,
            # Decision
            "previous_decision": conf_data.get("label", "NOT RECORDED"),
            "updated_assessment": updated_assessment,
            "deployment_status": deploy_status,
        })

        print(f"\n  {species}:")
        print(f"    Promoter score: {promoter_score:.4f}")
        print(f"    Localization: {existing_loc} → {updated_loc} (consensus: {updated_loc_consensus_conf})")
        print(f"    Degradation: {existing_deg_class} → {updated_deg_class} ({updated_deg_score:.4f})")
        print(f"    Safe harbor: {'available' if has_safe_harbor else 'NOT AVAILABLE'}")
        print(f"    System risk: {system_risk:.4f} ({risk_class})")
        print(f"    Updated assessment: {updated_assessment}")

    if not rows:
        print("\n  No species results to evaluate.")
        return

    out_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "system_re_evaluation.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
