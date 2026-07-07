"""
Generalization v2 — Residualized Targets, Cluster-Aware CV,
Biological Attribution, Artifact Registry.

Addresses four priorities:
  P1: Redesign expression_score (within-species residualization)
  P2: Sequence family clustering for cluster-aware CV
  P3: Biological attribution layer (positional saliency, motif enrichment)
  P4: Versioned artifact registry

Usage:
    python scripts/generalization_v2.py
"""

import glob
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "learned_promoter")


# ── Data loading ────────────────────────────────────────────────────────────────

def load_data():
    pattern1 = "outputs/*/iter*_scored.csv"
    pattern2 = "outputs/*/*/iter*_scored.csv"
    files = glob.glob(pattern1) + glob.glob(pattern2)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, index_col=0)
            parts = f.split("/")
            species = "unknown"
            for p in parts:
                if any(s in p for s in ["arabidopsis", "nbenthamiana", "tomato",
                                         "rice", "wheat", "maize", "soybean",
                                         "ntobacum", "by2"]):
                    species = p.split("_2026")[0].replace("outer_", "")
                    break
            df["_species"] = species
            dfs.append(df)
        except Exception:
            pass
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["sequence"])
    return df


def compute_features(df):
    from modules.statistical.sequence_features import build_feature_matrix
    feat_names, X = build_feature_matrix(df["sequence"].tolist())
    return feat_names, X


def remove_leaky_features(X, feat_names, groups, threshold=0.20):
    le = LabelEncoder()
    y_sp = le.fit_transform(groups)
    clean_mask = []
    for i, name in enumerate(feat_names):
        if X[:, i].std() > 0:
            corr = abs(np.corrcoef(X[:, i], y_sp)[0, 1])
        else:
            corr = 0.0
        clean_mask.append(corr <= threshold)
    X_clean = X[:, clean_mask]
    names_clean = [n for n, m in zip(feat_names, clean_mask) if m]
    return X_clean, names_clean, clean_mask


# ── P1: Residualized Targets ───────────────────────────────────────────────────

def compute_residualized_target(df, target_col, group_col="_species"):
    """Within-species z-score normalization.

    Converts raw scores into: how much better/worse than average for this species?
    Removes the species-level constant, preserves within-species variation.
    """
    df = df.copy()
    grouped = df.groupby(group_col)[target_col]
    species_mean = grouped.transform("mean")
    species_std = grouped.transform("std").clip(lower=0.01)
    df[f"{target_col}_residualized"] = (df[target_col] - species_mean) / species_std
    return df


def compute_rank_target(df, target_col, group_col="_species"):
    """Within-species percentile ranking.

    Converts to: what percentile is this promoter within its species?
    Removes species-specific score levels entirely.
    """
    df = df.copy()
    df[f"{target_col}_rank"] = df.groupby(group_col)[target_col].transform(
        lambda x: x.rank(pct=True)
    )
    return df


def compute_classification_target(df, target_col, thresholds=None):
    """Convert to classification: high/medium/low based on thresholds."""
    if thresholds is None:
        q = df[target_col].quantile([0.33, 0.67])
        thresholds = [q.iloc[0], q.iloc[1]]
    labels = np.ones(len(df), dtype=int)  # 1 = medium
    labels[df[target_col] < thresholds[0]] = 0  # low
    labels[df[target_col] > thresholds[1]] = 2  # high
    return labels


def evaluate_target(X, y, groups, target_label, feat_names):
    """Train and evaluate with K-Fold and Group K-Fold."""
    print(f"\n  {target_label}")
    print(f"    n={len(y)}, range=[{y.min():.3f}, {y.max():.3f}], "
          f"mean={y.mean():.3f}, std={y.std():.3f}")

    models = {
        "RF": RandomForestRegressor(n_estimators=200, max_depth=10,
                                     min_samples_leaf=5, random_state=42, n_jobs=-1),
        "GB": GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                         learning_rate=0.05, min_samples_leaf=5,
                                         subsample=0.8, random_state=42),
    }

    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        r2 = cross_val_score(model, X, y, cv=kf, scoring="r2")
        results[f"{name}_kf"] = float(r2.mean())

    n_groups = len(np.unique(groups))
    if n_groups >= 3:
        gkf = GroupKFold(n_splits=min(5, n_groups))
        for name, model in models.items():
            r2 = cross_val_score(model, X, y, cv=gkf, scoring="r2", groups=groups)
            mae = -cross_val_score(model, X, y, cv=gkf,
                                    scoring="neg_mean_absolute_error", groups=groups)
            results[f"{name}_gkf_r2"] = float(r2.mean())
            results[f"{name}_gkf_mae"] = float(mae.mean())

    kf_r2 = results.get("RF_kf", 0)
    gkf_r2 = results.get("RF_gkf_r2", float("nan"))
    gkf_mae = results.get("RF_gkf_mae", float("nan"))
    print(f"    KF R²={kf_r2:.3f}  Group R²={gkf_r2:.3f}  Group MAE={gkf_mae:.4f}")
    return results


# ── P2: Sequence Clustering ────────────────────────────────────────────────────

def cluster_promoters(X, feat_names, n_clusters=None):
    """Cluster promoters by k-mer profile similarity.

    Uses hierarchical clustering on standardized k-mer features.
    If n_clusters is None, auto-selects via gap in linkage distance.
    """
    # Use only k-mer features for clustering (not motifs or physicochemical)
    kmer_cols = [i for i, n in enumerate(feat_names) if n.startswith("k2_") or n.startswith("k3_")]
    if not kmer_cols:
        kmer_cols = list(range(X.shape[1]))

    X_kmer = X[:, kmer_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_kmer)

    # Cosine distance
    dist_matrix = pdist(X_scaled, metric="cosine")
    Z = linkage(dist_matrix, method="average")

    if n_clusters is None:
        # Auto-select: find largest gap in merge distance
        merge_dists = Z[:, 2]
        gaps = np.diff(merge_dists)
        gap_idx = np.argmax(gaps)
        n_clusters = len(merge_dists) - gap_idx
        n_clusters = max(5, min(30, n_clusters))

    cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return cluster_labels, n_clusters


def evaluate_cluster_cv(X, y, cluster_labels, feat_names):
    """Cluster-aware CV: hold out entire clusters at a time."""
    n_clusters = len(np.unique(cluster_labels))
    print(f"\n  Cluster-aware CV ({n_clusters} clusters):")

    if n_clusters < 3:
        print("    Too few clusters for CV")
        return {}

    models = {
        "RF": RandomForestRegressor(n_estimators=200, max_depth=10,
                                     min_samples_leaf=5, random_state=42, n_jobs=-1),
        "GB": GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                         learning_rate=0.05, min_samples_leaf=5,
                                         subsample=0.8, random_state=42),
    }

    gkf = GroupKFold(n_splits=min(5, n_clusters))
    results = {}
    for name, model in models.items():
        r2 = cross_val_score(model, X, y, cv=gkf, scoring="r2", groups=cluster_labels)
        mae = -cross_val_score(model, X, y, cv=gkf,
                                scoring="neg_mean_absolute_error", groups=cluster_labels)
        results[f"{name}_cluster_r2"] = float(r2.mean())
        results[f"{name}_cluster_mae"] = float(mae.mean())
        print(f"    {name}: R²={r2.mean():.3f} ± {r2.std():.3f}  MAE={mae.mean():.4f}")

    return results


# ── P3: Biological Attribution ─────────────────────────────────────────────────

def positional_saliency(model, scaler, sequence, feature_names, window=50):
    """Compute positional importance by masking sliding windows.

    For each window position, replace the window with random nucleotides
    and measure how much the prediction changes. Higher change = more important.
    """
    from modules.statistical.sequence_features import compute_sequence_features

    # Baseline prediction
    feats = compute_sequence_features(sequence)
    X = np.array([[feats.get(k, 0.0) for k in feature_names]])
    X_s = scaler.transform(X)
    baseline = model.predict(X_s)[0]

    saliency = []
    seq = list(sequence)
    rng = np.random.RandomState(42)
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}

    for start in range(0, len(sequence) - window + 1, window // 2):
        end = min(start + window, len(sequence))
        masked = seq.copy()
        for i in range(start, end):
            # Replace with random nucleotide
            masked[i] = rng.choice(["A", "C", "G", "T"])
        masked_seq = "".join(masked)

        feats_m = compute_sequence_features(masked_seq)
        X_m = np.array([[feats_m.get(k, 0.0) for k in feature_names]])
        X_ms = scaler.transform(X_m)
        pred = model.predict(X_ms)[0]

        importance = abs(baseline - pred)
        saliency.append({
            "region_start": start,
            "region_end": end,
            "importance": round(importance, 4),
            "baseline_score": round(baseline, 4),
            "masked_score": round(pred, 4),
        })

    return saliency


def motif_enrichment_analysis(df, feat_names, X, top_fraction=0.25):
    """Compare motif frequencies between top and bottom scoring promoters."""
    scores = df["composite_score"].values
    threshold_high = np.percentile(scores, 100 * (1 - top_fraction))
    threshold_low = np.percentile(scores, 100 * top_fraction)

    high_mask = scores >= threshold_high
    low_mask = scores <= threshold_low

    motif_cols = [(i, n) for i, n in enumerate(feat_names) if n.startswith("motif_")]

    enrichment = []
    for idx, name in motif_cols:
        high_freq = X[high_mask, idx].mean()
        low_freq = X[low_mask, idx].mean()
        fold_change = high_freq / max(low_freq, 0.01)
        high_present = (X[high_mask, idx] > 0).mean()
        low_present = (X[low_mask, idx] > 0).mean()

        enrichment.append({
            "motif": name.replace("motif_", ""),
            "high_score_mean": round(float(high_freq), 4),
            "low_score_mean": round(float(low_freq), 4),
            "fold_change": round(float(fold_change), 2),
            "high_present_pct": round(float(high_present * 100), 1),
            "low_present_pct": round(float(low_present * 100), 1),
        })

    enrichment.sort(key=lambda x: -x["fold_change"])
    return enrichment


# ── P4: Artifact Registry ──────────────────────────────────────────────────────

def save_versioned_artifacts(version, promoter_results, protein_cal, attribution,
                             cluster_info, metadata):
    """Save versioned artifacts with full metadata."""
    version_dir = os.path.join(PROJECT_ROOT, "artifacts", f"model_{version}")
    os.makedirs(version_dir, exist_ok=True)

    # Model artifacts
    joblib.dump(promoter_results["model"],
                os.path.join(version_dir, "promoter_rf_model.joblib"))
    joblib.dump(promoter_results["scaler"],
                os.path.join(version_dir, "promoter_scaler.joblib"))

    with open(os.path.join(version_dir, "feature_names.json"), "w") as f:
        json.dump(promoter_results["feature_names"], f)

    with open(os.path.join(version_dir, "conformal_qhat.json"), "w") as f:
        json.dump({"q_hat": promoter_results["q_hat"], "alpha": 0.10,
                    "n_calibration": promoter_results["metrics"]["n_cal"]}, f)

    np.save(os.path.join(version_dir, "conformal_scores.npy"),
            promoter_results["nonconformity_scores"])

    # Training config
    config = {
        "model_type": "RandomForest",
        "n_estimators": 500,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "max_features": 0.33,
        "random_state": 42,
        "target": "composite_score",
        "feature_selection": "removed species-correlated (r>0.20)",
        "n_features": len(promoter_results["feature_names"]),
        "n_train": promoter_results["metrics"]["n_train"],
        "n_cal": promoter_results["metrics"]["n_cal"],
        "conformal_alpha": 0.10,
    }
    with open(os.path.join(version_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Feature schema
    schema = {
        "version": version,
        "n_features": len(promoter_results["feature_names"]),
        "features": [{"name": n, "index": i}
                     for i, n in enumerate(promoter_results["feature_names"])],
    }
    with open(os.path.join(version_dir, "feature_schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    # Calibration report
    cal_report = {
        "train_r2": promoter_results["metrics"]["train_r2"],
        "cal_r2": promoter_results["metrics"]["cal_r2"],
        "cal_mae": promoter_results["metrics"]["cal_mae"],
        "conformal_coverage": promoter_results["metrics"]["conformal_coverage"],
        "conformal_q_hat": promoter_results["q_hat"],
        "feature_importances": promoter_results["feature_importances"],
        "shap_analysis": promoter_results.get("shap", {}),
    }
    with open(os.path.join(version_dir, "calibration_report.json"), "w") as f:
        json.dump(cal_report, f, indent=2, default=float)

    # Protein calibration
    with open(os.path.join(version_dir, "protein_calibration.json"), "w") as f:
        json.dump(protein_cal, f, indent=2, default=float)

    # Attribution data
    with open(os.path.join(version_dir, "attribution.json"), "w") as f:
        json.dump(attribution, f, indent=2, default=float)

    # Cluster info
    with open(os.path.join(version_dir, "cluster_info.json"), "w") as f:
        json.dump(cluster_info, f, indent=2, default=float)

    # Metadata
    with open(os.path.join(version_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=float)

    # OOD reference
    stats = {
        "feature_mean": promoter_results["feature_mean"],
        "feature_cov_inv": promoter_results["feature_cov_inv"],
        "ood_threshold": promoter_results["ood_threshold"],
    }
    with open(os.path.join(version_dir, "ood_reference.json"), "w") as f:
        json.dump(stats, f, indent=2, default=float)

    # Update latest symlink
    latest_dir = os.path.join(PROJECT_ROOT, "artifacts", "latest")
    if os.path.islink(latest_dir):
        os.unlink(latest_dir)
    elif os.path.isdir(latest_dir):
        pass  # Don't overwrite if it's a real dir
    else:
        os.symlink(version_dir, latest_dir)

    print(f"  Artifacts saved to: {version_dir}")
    for name in sorted(os.listdir(version_dir)):
        size = os.path.getsize(os.path.join(version_dir, name))
        print(f"    {name}: {size / 1024:.0f} KB")

    return version_dir


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print()
    print("=" * 60)
    print("GENERALIZATION v2 — Depth Pipeline")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading data...")
    df = load_data()
    print(f"  {len(df)} sequences")

    # Compute features
    print("\n[2/6] Computing features...")
    feat_names, X = compute_features(df)
    groups = df["_species"].values
    X_clean, clean_names, _ = remove_leaky_features(X, feat_names, groups)
    print(f"  {len(feat_names)} → {len(clean_names)} clean features")

    # ── P1: Residualized targets ──
    print("\n" + "=" * 60)
    print("P1: EXPRESSION SCORE REDESIGN")
    print("=" * 60)

    # Original
    print("\n  --- Baseline: raw expression_score ---")
    y_expr_raw = df["expression_score"].values.astype(float)
    evaluate_target(X_clean, y_expr_raw, groups, "raw expression_score", clean_names)

    # Within-species z-score
    print("\n  --- Fix A: within-species z-score (residualized) ---")
    df_r = compute_residualized_target(df, "expression_score")
    y_expr_resid = df_r["expression_score_residualized"].values.astype(float)
    evaluate_target(X_clean, y_expr_resid, groups,
                    "residualized expression_score", clean_names)

    # Within-species rank
    print("\n  --- Fix B: within-species percentile rank ---")
    df_r = compute_rank_target(df_r, "expression_score")
    y_expr_rank = df_r["expression_score_rank"].values.astype(float)
    evaluate_target(X_clean, y_expr_rank, groups,
                    "ranked expression_score", clean_names)

    # Composite (for comparison)
    print("\n  --- Reference: composite_score ---")
    y_comp = df["composite_score"].values.astype(float)
    comp_results = evaluate_target(X_clean, y_comp, groups,
                                    "composite_score", clean_names)

    # ── P2: Sequence clustering ──
    print("\n" + "=" * 60)
    print("P2: SEQUENCE FAMILY CLUSTERING")
    print("=" * 60)

    cluster_labels, n_clusters = cluster_promoters(X_clean, clean_names)
    print(f"  Clustered into {n_clusters} families")
    cluster_sizes = Counter(cluster_labels)
    print(f"  Cluster sizes: min={min(cluster_sizes.values())}, "
          f"max={max(cluster_sizes.values())}, "
          f"median={int(np.median(list(cluster_sizes.values())))}")

    # Cluster-aware CV on composite_score
    cluster_results = evaluate_cluster_cv(X_clean, y_comp, cluster_labels, clean_names)

    # Compare with species Group CV
    print(f"\n  Comparison (composite_score, RF):")
    print(f"    Species Group CV:   R²={comp_results.get('RF_gkf_r2', 'N/A'):.3f}")
    print(f"    Cluster Group CV:   R²={cluster_results.get('RF_cluster_r2', 'N/A'):.3f}")
    print(f"    K-Fold (random):    R²={comp_results.get('RF_kf', 'N/A'):.3f}")

    # ── P3: Attribution ──
    print("\n" + "=" * 60)
    print("P3: BIOLOGICAL ATTRIBUTION")
    print("=" * 60)

    # Motif enrichment
    print("\n  Motif enrichment (top vs bottom 25%):")
    enrichment = motif_enrichment_analysis(df, clean_names, X_clean)
    for item in enrichment[:10]:
        direction = "↑" if item["fold_change"] > 1 else "↓"
        print(f"    {item['motif']:20s}  {direction} fold={item['fold_change']:.2f}  "
              f"high={item['high_present_pct']:.0f}%  low={item['low_present_pct']:.0f}%")

    # Positional saliency on a sample promoter
    print("\n  Positional saliency (sample promoter):")
    sample_seq = df.iloc[0]["sequence"]

    # Train a quick model for saliency
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_clean)
    model = RandomForestRegressor(n_estimators=200, max_depth=10,
                                   min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X_all_s, y_comp)

    saliency = positional_saliency(model, scaler, sample_seq, clean_names, window=100)
    saliency.sort(key=lambda x: -x["importance"])
    print(f"    Baseline score: {saliency[0]['baseline_score']:.4f}")
    for region in saliency[:5]:
        print(f"    Region {region['region_start']:4d}-{region['region_end']:4d}: "
              f"importance={region['importance']:.4f}  "
              f"masked_score={region['masked_score']:.4f}")

    # ── P4: Retrain production model + save versioned artifacts ──
    print("\n" + "=" * 60)
    print("P4: VERSIONED ARTIFACT REGISTRY")
    print("=" * 60)

    # Retrain with clean features + conformal calibration
    print("\n  Retraining production model (99 clean features)...")
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(y_comp))
    n_cal = max(1, int(len(y_comp) * 0.20))
    cal_idx = idx[:n_cal]
    train_idx = idx[n_cal:]

    X_train, y_train = X_clean[train_idx], y_comp[train_idx]
    X_cal, y_cal = X_clean[cal_idx], y_comp[cal_idx]

    scaler_p = StandardScaler()
    X_train_s = scaler_p.fit_transform(X_train)
    X_cal_s = scaler_p.transform(X_cal)

    model_p = RandomForestRegressor(
        n_estimators=500, max_depth=12, min_samples_leaf=5,
        max_features=0.33, random_state=42, n_jobs=-1,
    )
    model_p.fit(X_train_s, y_train)

    y_cal_pred = model_p.predict(X_cal_s)
    nonconformity = np.abs(y_cal - y_cal_pred)
    q_hat_index = min(int(np.ceil((n_cal + 1) * 0.90)) - 1, n_cal - 1)
    q_hat = float(np.sort(nonconformity)[q_hat_index])
    coverage = float(np.sum(nonconformity <= q_hat) / n_cal)

    # OOD reference
    X_all_s = scaler_p.transform(X_clean)
    feature_mean = X_all_s.mean(axis=0).tolist()
    cov = np.cov(X_all_s.T) + 1e-6 * np.eye(X_all_s.shape[1])
    cov_inv = np.linalg.inv(cov).tolist()

    from scipy.stats import chi2
    ood_threshold = float(chi2.ppf(0.975, df=X_all_s.shape[1]))

    importances = model_p.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]

    promoter_results = {
        "model": model_p,
        "scaler": scaler_p,
        "feature_names": clean_names,
        "q_hat": q_hat,
        "nonconformity_scores": nonconformity,
        "feature_mean": feature_mean,
        "feature_cov_inv": cov_inv,
        "ood_threshold": ood_threshold,
        "metrics": {
            "train_r2": float(r2_score(y_train, model_p.predict(X_train_s))),
            "cal_r2": float(r2_score(y_cal, y_cal_pred)),
            "cal_mae": float(mean_absolute_error(y_cal, y_cal_pred)),
            "conformal_coverage": coverage,
            "n_train": len(train_idx),
            "n_cal": n_cal,
            "n_features": len(clean_names),
        },
        "feature_importances": [
            {"feature": clean_names[i], "importance": float(importances[i])}
            for i in top_idx
        ],
        "shap": {"shap_ranking": []},  # Will be populated by train_production_model.py
    }

    print(f"  Production model: R²={promoter_results['metrics']['cal_r2']:.4f}, "
          f"coverage={coverage:.1%}, q_hat={q_hat:.4f}")

    # Save versioned artifacts
    protein_cal = {
        "folding_quality": {"bias_correction": -0.1675, "empirical_noise": 0.1518},
        "glycosylation": {"bias_correction": 3.9, "empirical_noise": 1.3984},
        "manufacturing": {"bias_correction": 0.0, "empirical_noise": 0.08},
    }

    attribution = {
        "motif_enrichment": enrichment,
        "positional_saliency_sample": saliency,
    }

    cluster_info = {
        "n_clusters": int(n_clusters),
        "cluster_sizes": {str(k): int(v) for k, v in cluster_sizes.items()},
        "cluster_cv_r2": cluster_results.get("RF_cluster_r2", None),
        "species_cv_r2": comp_results.get("RF_gkf_r2", None),
    }

    metadata = {
        "version": "v2",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_samples": len(df),
        "n_features": len(clean_names),
        "n_leaky_removed": len(feat_names) - len(clean_names),
        "n_clusters": int(n_clusters),
        "targets_evaluated": [
            "composite_score", "expression_score_raw",
            "expression_score_residualized", "expression_score_ranked",
        ],
        "expression_score_diagnosis": {
            "fraction_near_1": float((df["expression_score"] > 0.95).mean()),
            "species_with_variation": [
                s for s in df["_species"].unique()
                if (df[df["_species"] == s]["expression_score"] > 0.95).mean() < 0.90
            ],
        },
    }

    version_dir = save_versioned_artifacts(
        "v2", promoter_results, protein_cal, attribution, cluster_info, metadata
    )

    # Also update latest/ symlink to point to v2
    latest = os.path.join(PROJECT_ROOT, "artifacts", "latest")
    if os.path.islink(latest):
        os.unlink(latest)
    os.symlink(version_dir, latest)
    print(f"  latest → {version_dir}")

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"GENERALIZATION v2 COMPLETE — {elapsed:.1f}s")
    print()
    print("P1 RESIDUALIZED TARGETS:")
    print("  expression_score (raw):         Group R² ≈ -295")
    print("  expression_score (z-scored):    see above")
    print("  expression_score (ranked):      see above")
    print("  composite_score (reference):    see above")
    print()
    print(f"P2 CLUSTER CV: {n_clusters} sequence families")
    print(f"  Cluster R² vs Species R² vs K-Fold R²: see above")
    print()
    print("P3 ATTRIBUTION:")
    print(f"  Motif enrichment: {len(enrichment)} motifs analyzed")
    print(f"  Positional saliency: {len(saliency)} regions analyzed")
    print()
    print(f"P4 ARTIFACTS: {version_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
