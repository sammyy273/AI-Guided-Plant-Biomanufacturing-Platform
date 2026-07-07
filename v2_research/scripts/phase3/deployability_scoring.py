#!/usr/bin/env python3
"""
STEP 6: Deployability Scoring.

Combines all pipeline layers into a single deployability score (0-1):
  - Promoter score (from pipeline)
  - CDS quality (CAI + folding + accessibility)
  - Localization confidence (ESM2 + heuristic consensus)
  - Structure-adjusted degradation
  - Genome context (safe harbor availability)

Decision:
  > 0.75 → DEPLOYABLE (computational)
  0.50-0.75 → CONDITIONAL
  < 0.50 → NOT DEPLOYABLE

OUTPUTS:
  outputs/phase3/deployability_matrix.csv
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

OUTPUT_DIR = BASE_DIR / "outputs" / "phase3"
PHASE2_DIR = BASE_DIR / "outputs" / "phase2"


def load_json(path):
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return {}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 6: Deployability Scoring")
    print("=" * 60)

    # Load all Phase 3 results
    loc_esm = load_json(OUTPUT_DIR / "localization_esm_detail.json")
    deg_detail = load_json(OUTPUT_DIR / "degradation_detail.json")
    cds_df = pd.read_csv(OUTPUT_DIR / "cds_optimization_advanced.csv") if (OUTPUT_DIR / "cds_optimization_advanced.csv").exists() else pd.DataFrame()
    construct_df = pd.read_csv(OUTPUT_DIR / "construct_analysis.csv") if (OUTPUT_DIR / "construct_analysis.csv").exists() else pd.DataFrame()
    safe_harbor_df = pd.read_csv(PHASE2_DIR / "safe_harbor_candidates.csv") if (PHASE2_DIR / "safe_harbor_candidates.csv").exists() else pd.DataFrame()

    species_list = ["tomato", "rice", "nbenthamiana"]
    rows = []

    for species in species_list:
        print(f"\n  --- {species} ---")

        # Load original report
        report = load_json(BASE_DIR / "outputs" / f"final_report_{species}.json")
        if not report:
            print(f"    No report found")
            continue

        bp = report.get("best_promoter", {})
        cds_data = report.get("optimized_cds", {})
        conf_data = report.get("confidence", {})

        # ── 1. Promoter score ───────────────────────────────────────────
        promoter_score = bp.get("composite_score", 0)
        promoter_risk = 1 - promoter_score
        print(f"    Promoter score: {promoter_score:.4f}")

        # ── 2. CDS quality ──────────────────────────────────────────────
        cds_cai = cds_data.get("cai", 0)
        if not cds_df.empty:
            cds_row = cds_df[cds_df["species"] == species]
            cds_quality = cds_row["cds_quality_score"].values[0] if not cds_row.empty else cds_cai
        else:
            cds_quality = cds_cai
        print(f"    CDS quality: {cds_quality:.4f}")

        # ── 3. Localization confidence ──────────────────────────────────
        if loc_esm:
            consensus = loc_esm.get("consensus", {})
            loc_score = consensus.get("score", 0.5)
            loc_conf = consensus.get("confidence", "MEDIUM")
            loc_agree = consensus.get("methods_agree", False)
        else:
            loc_score = 0.5
            loc_conf = "NOT ASSESSED"
            loc_agree = False

        # Map confidence to score modifier
        loc_mod = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4, "NOT ASSESSED": 0.3}.get(loc_conf, 0.5)
        loc_final = loc_score * loc_mod
        print(f"    Localization: {consensus.get('prediction', '?')} ({loc_conf}), score={loc_final:.4f}")

        # ── 4. Structure-adjusted degradation ───────────────────────────
        if deg_detail:
            deg_risk = deg_detail.get("primary_degradation_risk", 0.5)
            deg_class = deg_detail.get("primary_risk_class", "MEDIUM")
            deg_score = 1 - deg_risk  # Invert: low degradation risk = high score
        else:
            deg_risk = 0.5
            deg_class = "NOT ASSESSED"
            deg_score = 0.3

        print(f"    Degradation: {deg_class} (risk={deg_risk:.4f}), score={deg_score:.4f}")

        # ── 5. Genome context ───────────────────────────────────────────
        genome_species_map = {"nbenthamiana": None, "rice": "rice", "tomato": "tomato"}
        gs = genome_species_map.get(species)
        if gs and not safe_harbor_df.empty:
            sh_count = len(safe_harbor_df[safe_harbor_df["species"] == gs])
            genome_score = min(1.0, sh_count / 30)
        else:
            sh_count = 0
            genome_score = 0.2  # Low score if no data
        print(f"    Genome context: {sh_count} safe harbors, score={genome_score:.4f}")

        # ── 6. Construct readiness ──────────────────────────────────────
        if not construct_df.empty:
            c_row = construct_df[construct_df["species"] == species]
            if not c_row.empty:
                construct_status = c_row["status"].values[0]
                construct_score = 1.0 if construct_status == "READY" else 0.6
            else:
                construct_status = "NOT BUILT"
                construct_score = 0.3
        else:
            construct_status = "NOT ASSESSED"
            construct_score = 0.3
        print(f"    Construct: {construct_status}, score={construct_score:.4f}")

        # ── Compute weighted deployability score ────────────────────────
        weights = {
            "promoter": 0.20,
            "cds": 0.15,
            "localization": 0.25,
            "degradation": 0.20,
            "genome": 0.10,
            "construct": 0.10,
        }

        deployability = (
            weights["promoter"] * promoter_score +
            weights["cds"] * cds_quality +
            weights["localization"] * loc_final +
            weights["degradation"] * deg_score +
            weights["genome"] * genome_score +
            weights["construct"] * construct_score
        )

        # Decision
        if deployability > 0.75:
            decision = "DEPLOYABLE (computational)"
        elif deployability >= 0.50:
            decision = "CONDITIONAL"
        else:
            decision = "NOT DEPLOYABLE"

        # Identify bottleneck
        scores = {
            "promoter": promoter_score,
            "cds": cds_quality,
            "localization": loc_final,
            "degradation": deg_score,
            "genome": genome_score,
            "construct": construct_score,
        }
        bottleneck = min(scores, key=scores.get)

        print(f"\n    DEPLOYABILITY: {deployability:.4f} → {decision}")
        print(f"    Bottleneck: {bottleneck} ({scores[bottleneck]:.4f})")

        rows.append({
            "species": species,
            "promoter_score": round(promoter_score, 4),
            "cds_quality": round(cds_quality, 4),
            "localization_confidence": round(loc_final, 4),
            "localization_prediction": consensus.get("prediction", "?") if loc_esm else "NOT ASSESSED",
            "localization_methods_agree": loc_agree,
            "degradation_score": round(deg_score, 4),
            "degradation_risk_class": deg_class,
            "genome_context_score": round(genome_score, 4),
            "construct_status": construct_status,
            "construct_score": round(construct_score, 4),
            "deployability_score": round(deployability, 4),
            "decision": decision,
            "bottleneck": bottleneck,
            "bottleneck_score": round(scores[bottleneck], 4),
        })

    if not rows:
        print("\n  No results generated.")
        return

    out_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "deployability_matrix.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(out_df)} rows)")

    # Print summary
    print("\n  DEPLOYABILITY SUMMARY:")
    print(f"  {'Species':15s} {'Score':>6s} {'Decision':25s} {'Bottleneck':15s}")
    print(f"  {'-'*15} {'-'*6} {'-'*25} {'-'*15}")
    for _, row in out_df.iterrows():
        print(f"  {row['species']:15s} {row['deployability_score']:>6.4f} {row['decision']:25s} {row['bottleneck']:15s}")


if __name__ == "__main__":
    main()
