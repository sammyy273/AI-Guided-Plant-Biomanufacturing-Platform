"""
Train First Learned Model — From Heuristic Rules to Statistical Learning.

Aggregates all scored promoter candidates, computes sequence-derived features,
trains sklearn models with LOO-CV, and reports honest performance metrics.

This is the transition from:
    IF motif present AND hydrophobicity high THEN score favorable
To:
    Model learned latent relationships from sequence-score data

Targets:
    composite_score  — composite construct quality (regression)
    expression_score — expression component (regression)

Usage:
    python scripts/train_first_model.py
    python scripts/train_first_model.py --target expression_score
    python scripts/train_first_model.py --save-table
"""

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    cross_val_score,
    LeaveOneGroupOut,
    GroupKFold,
    KFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "learned_models")


def load_iteration_data():
    """Load all iteration scored CSVs into a single DataFrame."""
    pattern1 = os.path.join(PROJECT_ROOT, "outputs", "*", "iter*_scored.csv")
    pattern2 = os.path.join(PROJECT_ROOT, "outputs", "*", "*", "iter*_scored.csv")
    files = glob.glob(pattern1) + glob.glob(pattern2)
    print(f"Found {len(files)} iteration CSVs")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, index_col=0)
            parts = f.replace("\\", "/").split("/")
            species = "unknown"
            for p in parts:
                if any(s in p for s in ["arabidopsis", "nbenthamiana", "tomato",
                                         "rice", "wheat", "maize", "soybean",
                                         "ntobacum", "by2"]):
                    species = p.split("_2026")[0].replace("outer_", "")
                    break
            df["_species"] = species
            df["_source"] = f
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: {f}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["sequence"])
    print(f"  {len(combined)} unique sequences after dedup")
    return combined


def compute_features(df):
    """Compute sequence features for all rows."""
    from modules.statistical.sequence_features import build_feature_matrix

    sequences = df["sequence"].tolist()
    feature_names, X = build_feature_matrix(sequences)
    print(f"  Computed {len(feature_names)} features per sequence")
    return feature_names, X


def train_and_evaluate(X, y, groups, target_name, feature_names):
    """Train multiple models with cross-validation and report results."""
    print(f"\n{'=' * 60}")
    print(f"TARGET: {target_name}")
    print(f"  n_samples: {len(y)}")
    print(f"  target range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  target mean: {y.mean():.3f} ± {y.std():.3f}")
    print(f"  n_features: {X.shape[1]}")
    print(f"  n_groups (species): {len(np.unique(groups))}")
    print(f"{'=' * 60}")

    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        ),
    }

    results = {}

    # ── K-Fold CV (random split) ──
    print(f"\n  K-Fold CV (5-fold, random):")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        scores_r2 = cross_val_score(model, X, y, cv=kf, scoring="r2")
        scores_neg_mae = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
        mae = -scores_neg_mae

        results[name] = {
            "r2_mean": scores_r2.mean(),
            "r2_std": scores_r2.std(),
            "mae_mean": mae.mean(),
            "mae_std": mae.std(),
        }

        print(f"    {name:20s}  R²={scores_r2.mean():.3f} ± {scores_r2.std():.3f}"
              f"  MAE={mae.mean():.4f} ± {mae.std():.4f}")

    # ── Group K-Fold CV (by species) — tests generalization across species ──
    print(f"\n  Group K-Fold CV (by species — tests cross-species generalization):")
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    if n_groups >= 3:
        gkf = GroupKFold(n_splits=min(5, n_groups))
        for name, model in models.items():
            scores_r2 = cross_val_score(model, X, y, cv=gkf,
                                         scoring="r2", groups=groups)
            scores_neg_mae = cross_val_score(model, X, y, cv=gkf,
                                              scoring="neg_mean_absolute_error",
                                              groups=groups)
            mae = -scores_neg_mae
            results[f"{name}_grouped"] = {
                "r2_mean": scores_r2.mean(),
                "r2_std": scores_r2.std(),
                "mae_mean": mae.mean(),
                "mae_std": mae.std(),
            }
            print(f"    {name:20s}  R²={scores_r2.mean():.3f} ± {scores_r2.std():.3f}"
                  f"  MAE={mae.mean():.4f} ± {mae.std():.4f}")
    else:
        print(f"    Skipped: only {n_groups} groups")

    # ── Feature importances (from best tree model) ──
    print(f"\n  Top 20 features (GradientBoosting):")
    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=5, subsample=0.8, random_state=42,
    )
    gb.fit(X, y)
    importances = gb.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    top_features = []
    for i in range(min(20, len(feature_names))):
        idx = sorted_idx[i]
        fname = feature_names[idx]
        imp = importances[idx]
        print(f"    {i+1:2d}. {fname:30s}  importance={imp:.4f}")
        top_features.append({"feature": fname, "importance": float(imp)})

    results["feature_importances"] = top_features

    # ── Null baseline: predict mean ──
    null_mae = np.mean(np.abs(y - y.mean()))
    null_r2 = 0.0
    print(f"\n  Null baseline (predict mean): R²={null_r2:.3f}  MAE={null_mae:.4f}")

    return results, gb


def analyze_by_species(df, feature_names, X, target_name):
    """Break down model performance by species."""
    print(f"\n  Per-species target statistics:")
    species_stats = df.groupby("_species")[target_name].agg(["count", "mean", "std"])
    for sp, row in species_stats.iterrows():
        print(f"    {sp:20s}  n={int(row['count']):4d}"
              f"  mean={row['mean']:.3f} ± {row['std']:.3f}")


def save_training_table(df, feature_names, X, output_dir):
    """Save the full training table as CSV."""
    os.makedirs(output_dir, exist_ok=True)
    feat_df = pd.DataFrame(X, columns=feature_names)
    combined = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    path = os.path.join(output_dir, "training_table.csv")
    combined.to_csv(path, index=False)
    print(f"\n  Training table saved: {path} ({len(combined)} rows, {len(combined.columns)} cols)")
    return path


def main():
    t0 = time.time()
    target = "composite_score"
    if "--target" in sys.argv:
        idx = sys.argv.index("--target")
        target = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "composite_score"
    save_table = "--save-table" in sys.argv

    print()
    print("=" * 60)
    print("LEARNED MODEL TRAINING — Depth Over Breadth")
    print("=" * 60)
    print()

    # Step 1: Load data
    print("Step 1: Loading iteration data...")
    df = load_iteration_data()

    # Step 2: Compute features
    print("\nStep 2: Computing sequence features...")
    feature_names, X = compute_features(df)

    # Step 3: Prepare targets
    targets_available = {
        "composite_score": "Composite construct quality (heuristic composite)",
        "expression_score": "Expression component",
        "silencing_risk": "Silencing risk",
        "gc_pct": "GC percentage (trivial — should be near-perfect)",
    }

    print("\nStep 3: Available targets:")
    for t_name, t_desc in targets_available.items():
        if t_name in df.columns:
            vals = df[t_name].dropna()
            print(f"  {t_name:25s}  {t_desc}")
            print(f"    range=[{vals.min():.3f}, {vals.max():.3f}]"
                  f"  mean={vals.mean():.3f}")

    # Step 4: Train and evaluate
    print(f"\nStep 4: Training models (target: {target})")
    y = df[target].values.astype(float)
    groups = df["_species"].values

    results, best_model = train_and_evaluate(X, y, groups, target, feature_names)

    # Per-species analysis
    analyze_by_species(df, feature_names, X, target)

    # Also train on expression_score for comparison
    if target == "composite_score" and "expression_score" in df.columns:
        print("\n" + "=" * 60)
        print("BONUS: Also training on expression_score")
        y_expr = df["expression_score"].values.astype(float)
        expr_results, _ = train_and_evaluate(X, y_expr, groups,
                                              "expression_score", feature_names)
        results["expression_score_results"] = expr_results

    # Step 5: Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if save_table:
        save_training_table(df, feature_names, X, OUTPUT_DIR)

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "first_model_results.json")
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\n  Results saved: {results_path}")

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"TRAINING COMPLETE — {elapsed:.1f}s")
    print()
    print("INTERPRETATION GUIDE:")
    print("  R² > 0.5: model captures meaningful sequence-score relationships")
    print("  R² 0.2-0.5: weak signal, features partially informative")
    print("  R² < 0.2: sequence features alone cannot predict this target")
    print("  (Low R² is EXPECTED for heuristic composite targets)")
    print()
    print("  Group CV R² << K-Fold R² means the model overfits to species patterns")
    print("  and won't generalize to new species.")
    print()
    print("NEXT STEPS IF R² IS LOW:")
    print("  1. Get experimental expression data (labeled by wet-lab)")
    print("  2. Add sequence embeddings (AgroNT/PlantBERT)")
    print("  3. Increase dataset size (>1000 labeled examples)")
    print("=" * 60)


if __name__ == "__main__":
    main()
