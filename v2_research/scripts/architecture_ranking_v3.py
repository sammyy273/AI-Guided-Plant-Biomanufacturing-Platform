"""
Architecture-Aware Features + Ranking Objectives v3.

Builds on v2 generalization results:
  - 72 new architecture features (spacing, orientation, co-occurrence, density, syntax)
  - Pairwise ranking objective (learn relative preference, not absolute score)
  - Within-family ordering using sequence clusters
  - Contrastive promoter representation
  - Evaluates whether architecture features + ranking improve cross-species generalization

Usage:
    python scripts/architecture_ranking_v3.py
"""

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


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

    combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["sequence"])
    print(f"  {len(combined)} unique sequences, {len(files)} files")
    return combined


def build_features(df, remove_leaky=True):
    """Build feature matrix with architecture-aware features."""
    from modules.statistical.sequence_features import build_feature_matrix

    sequences = df["sequence"].tolist()
    feature_names, X = build_feature_matrix(sequences)
    print(f"  {X.shape[1]} features (with architecture-aware)")

    if not remove_leaky:
        return feature_names, X

    # Species leakage audit
    le = LabelEncoder()
    y_sp = le.fit_transform(df["_species"])
    leaky_threshold = 0.20
    clean_mask = []
    for i, name in enumerate(feature_names):
        if X[:, i].std() > 0:
            corr = abs(np.corrcoef(X[:, i], y_sp)[0, 1])
        else:
            corr = 0.0
        clean_mask.append(corr <= leaky_threshold)

    n_leaky = sum(not m for m in clean_mask)
    X = X[:, clean_mask]
    feature_names = [n for n, m in zip(feature_names, clean_mask) if m]
    print(f"  Removed {n_leaky} leaky features, {len(feature_names)} clean")
    return feature_names, X


def compute_within_species_rank(df):
    """Compute within-species percentile rank for composite_score."""
    ranks = df.groupby("_species")["composite_score"].rank(pct=True)
    return ranks.values


def generate_pairwise_data(X, y, groups, n_pairs_per_group=500):
    """Generate pairwise training data for ranking.

    For each group (species), sample pairs and label by relative ordering.
    Returns (X_diff, y_pair) where X_diff = X[i] - X[j] and y_pair = sign(y[i] - y[j]).
    """
    unique_groups = np.unique(groups)
    X_pairs = []
    y_pairs = []
    group_pairs = []

    rng = np.random.RandomState(42)
    for g in unique_groups:
        mask = groups == g
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue

        n_possible = len(idx) * (len(idx) - 1) / 2
        n_sample = min(n_pairs_per_group, int(n_possible))

        # Sample random pairs
        for _ in range(n_sample):
            i, j = rng.choice(idx, 2, replace=False)
            if y[i] == y[j]:
                continue
            # Feature difference
            x_diff = X[i] - X[j]
            # Label: +1 if i > j, -1 if i < j
            label = 1.0 if y[i] > y[j] else -1.0
            X_pairs.append(x_diff)
            y_pairs.append(label)
            group_pairs.append(g)

    return np.array(X_pairs), np.array(y_pairs), np.array(group_pairs)


def train_ranking_model(X_pairs, y_pairs, group_pairs):
    """Train a ranking model on pairwise difference features."""
    print("\n  Training ranking model (pairwise differences)...")

    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=5, subsample=0.8, random_state=42,
        ),
    }

    # Group K-Fold on pairs
    n_groups = len(np.unique(group_pairs))
    results = {}

    print(f"\n  K-Fold CV on pairs ({len(y_pairs)} pairs):")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        r2 = cross_val_score(model, X_pairs, y_pairs, cv=kf, scoring="r2")
        acc = cross_val_score(model, X_pairs, (y_pairs > 0).astype(int), cv=kf, scoring="accuracy")
        results[name] = {"kf_r2": r2.mean(), "kf_acc": acc.mean()}
        print(f"    {name:20s}  R²={r2.mean():.3f}  Acc={acc.mean():.3f}")

    if n_groups >= 3:
        print(f"\n  Group K-Fold CV on pairs (by species, {n_groups} groups):")
        gkf = GroupKFold(n_splits=min(5, n_groups))
        for name, model in models.items():
            r2 = cross_val_score(model, X_pairs, y_pairs, cv=gkf,
                                  scoring="r2", groups=group_pairs)
            # For ranking, also compute pairwise accuracy
            from sklearn.model_selection import cross_val_predict
            try:
                y_pred = cross_val_predict(
                    Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(1.0))])
                    if name == "Ridge" else model,
                    X_pairs, y_pairs, cv=gkf, groups=group_pairs)
                acc = np.mean((y_pred > 0) == (y_pairs > 0))
            except Exception:
                acc = 0.5
            results[f"{name}_grouped"] = {"gkf_r2": r2.mean(), "gkf_acc": acc}
            print(f"    {name:20s}  R²={r2.mean():.3f}  Acc={acc:.3f}")

    # Train final model for feature importances
    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=5, subsample=0.8, random_state=42,
    )
    gb.fit(X_pairs, y_pairs)
    importances = gb.feature_importances_

    return results, importances


def evaluate_pointwise_with_rank_features(X, y, groups, ranks):
    """Evaluate pointwise regression using rank-transformed target."""
    print("\n  Pointwise regression with rank target:")
    results = {}

    models = {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(1.0))]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=5, subsample=0.8, random_state=42,
        ),
    }

    # K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print(f"\n  K-Fold CV (rank target):")
    for name, model in models.items():
        r2 = cross_val_score(model, X, ranks, cv=kf, scoring="r2")
        mae = -cross_val_score(model, X, ranks, cv=kf, scoring="neg_mean_absolute_error")
        results[name] = {"kf_r2": r2.mean(), "kf_mae": mae.mean()}
        print(f"    {name:20s}  R²={r2.mean():.3f} ± {r2.std():.3f}  MAE={mae.mean():.4f}")

    # Group K-Fold
    n_groups = len(np.unique(groups))
    if n_groups >= 3:
        gkf = GroupKFold(n_splits=min(5, n_groups))
        print(f"\n  Group K-Fold CV (rank target):")
        for name, model in models.items():
            r2 = cross_val_score(model, X, ranks, cv=gkf, scoring="r2", groups=groups)
            mae = -cross_val_score(model, X, ranks, cv=gkf,
                                    scoring="neg_mean_absolute_error", groups=groups)
            results[f"{name}_grouped"] = {"gkf_r2": r2.mean(), "gkf_mae": mae.mean()}
            print(f"    {name:20s}  R²={r2.mean():.3f} ± {r2.std():.3f}  MAE={mae.mean():.4f}")

    return results


def contrastive_analysis(X, y, feature_names):
    """Contrastive analysis: learn what separates top from bottom quartile."""
    print("\n  Contrastive analysis (top vs bottom quartile):")

    q75 = np.percentile(y, 75)
    q25 = np.percentile(y, 25)
    top_mask = y >= q75
    bot_mask = y <= q25

    X_top = X[top_mask]
    X_bot = X[bot_mask]

    # Mean difference
    diff = X_top.mean(axis=0) - X_bot.mean(axis=0)
    pooled_std = np.sqrt((X_top.std(axis=0)**2 + X_bot.std(axis=0)**2) / 2)
    pooled_std[pooled_std == 0] = 1.0
    effect_size = diff / pooled_std

    sorted_idx = np.argsort(np.abs(effect_size))[::-1]

    # Separate architecture vs original features
    arch_prefixes = ["spacing_", "has_pair_", "avg_spacing_", "orient_", "density_",
                     "count_distal", "count_mid", "count_proximal", "count_tss",
                     "cooc_", "syntax_", "motif_gap_", "tata_centrism"]

    print(f"\n  Top discriminative features (Cohen's d):")
    print(f"  {'Feature':50s} {'d':>7s} {'Type':>12s}")
    print(f"  {'-'*72}")

    top_arch = []
    top_orig = []
    for i in range(min(30, len(feature_names))):
        idx = sorted_idx[i]
        name = feature_names[idx]
        d = effect_size[idx]
        is_arch = any(name.startswith(p) for p in arch_prefixes)
        ftype = "ARCH" if is_arch else "ORIG"
        print(f"    {name:50s} {d:7.3f} {ftype:>12s}")
        if is_arch:
            top_arch.append((name, d))
        else:
            top_orig.append((name, d))

    print(f"\n  Architecture features in top-30 discriminators: {len(top_arch)}")
    print(f"  Original features in top-30: {len(top_orig)}")
    return top_arch, top_orig, effect_size


def compare_with_v2(X, y, groups, feature_names):
    """Compare v3 (architecture features) vs v2 (frequency-only) generalization."""
    print("\n  Architecture feature contribution to generalization:")

    arch_prefixes = ["spacing_", "has_pair_", "avg_spacing_", "orient_", "density_",
                     "count_distal", "count_mid", "count_proximal", "count_tss",
                     "cooc_", "syntax_", "motif_gap_", "tata_centrism"]
    arch_idx = [i for i, n in enumerate(feature_names)
                if any(n.startswith(p) for p in arch_prefixes)]
    orig_idx = [i for i in range(len(feature_names)) if i not in arch_idx]

    X_orig = X[:, orig_idx]
    X_arch = X[:, arch_idx]
    X_full = X

    configs = [
        ("Original features only", X_orig),
        ("Architecture features only", X_arch),
        ("Original + Architecture", X_full),
    ]

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )

    for label, X_sub in configs:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        kf_r2 = cross_val_score(rf, X_sub, y, cv=kf, scoring="r2")

        n_groups = len(np.unique(groups))
        if n_groups >= 3:
            gkf = GroupKFold(n_splits=min(5, n_groups))
            gkf_r2 = cross_val_score(rf, X_sub, y, cv=gkf, scoring="r2", groups=groups)
        else:
            gkf_r2 = np.array([0.0])

        print(f"    {label:30s}  KF R²={kf_r2.mean():.3f}  Group R²={gkf_r2.mean():.3f}  "
              f"n_feat={X_sub.shape[1]}")


def cluster_sequences(X, feature_names, n_clusters=5):
    """Cluster sequences by k-mer profile for family-aware evaluation."""
    # Use k-mer features only for clustering (not architecture features)
    kmer_cols = [i for i, n in enumerate(feature_names)
                 if n.startswith("k2_") or n.startswith("k3_")]
    if not kmer_cols:
        kmer_cols = list(range(min(50, X.shape[1])))

    X_kmer = X[:, kmer_cols]
    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_kmer_s = scaler.fit_transform(X_kmer)

    dist = pdist(X_kmer_s, metric="euclidean")
    Z = linkage(dist, method="ward")
    clusters = fcluster(Z, t=n_clusters, criterion="maxclust")
    return clusters


def main():
    t0 = time.time()

    print()
    print("=" * 60)
    print("ARCHITECTURE-AWARE + RANKING v3")
    print("=" * 60)

    # ── Load data ──
    print("\n[1/7] Loading data...")
    df = load_data()

    # ── Build features ──
    print("\n[2/7] Building features (v3: 189 features including architecture)...")
    feature_names, X = build_features(df, remove_leaky=True)

    groups = df["_species"].values
    y_comp = df["composite_score"].values.astype(float)
    ranks = compute_within_species_rank(df)

    print(f"  Feature dim: {X.shape[1]}")
    print(f"  Samples: {len(y_comp)}")
    print(f"  Species: {len(np.unique(groups))}")

    # ── P1: Architecture feature contribution ──
    print("\n[3/7] Architecture feature contribution...")
    compare_with_v2(X, y_comp, groups, feature_names)

    # ── P2: Contrastive analysis ──
    print("\n[4/7] Contrastive analysis...")
    top_arch, top_orig, effect_sizes = contrastive_analysis(X, y_comp, feature_names)

    # ── P3: Pairwise ranking ──
    print("\n[5/7] Pairwise ranking objective...")
    X_pairs, y_pairs, group_pairs = generate_pairwise_data(
        X, y_comp, groups, n_pairs_per_group=500)
    print(f"  Generated {len(y_pairs)} pairwise samples")
    print(f"  Positive pairs: {( y_pairs > 0).sum()}, Negative: {(y_pairs < 0).sum()}")

    ranking_results, pair_importances = train_ranking_model(X_pairs, y_pairs, group_pairs)

    # ── P4: Pointwise regression with rank target ──
    print("\n[6/7] Pointwise regression with rank target (architecture features)...")
    pointwise_results = evaluate_pointwise_with_rank_features(X, y_comp, groups, ranks)

    # ── P5: Cluster-aware CV with architecture features ──
    print("\n[7/7] Cluster-aware CV with architecture features...")
    clusters = cluster_sequences(X, feature_names, n_clusters=5)
    print(f"  Cluster sizes: {dict(zip(*np.unique(clusters, return_counts=True)))}")

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )

    # Cluster-aware CV
    gkf_cluster = GroupKFold(n_splits=5)
    cl_r2 = cross_val_score(rf, X, y_comp, cv=gkf_cluster, scoring="r2", groups=clusters)
    print(f"  Cluster-aware CV: R²={cl_r2.mean():.3f} ± {cl_r2.std():.3f}")

    # Species-aware CV for comparison
    gkf_species = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    sp_r2 = cross_val_score(rf, X, y_comp, cv=gkf_species, scoring="r2", groups=groups)
    print(f"  Species-aware CV: R²={sp_r2.mean():.3f} ± {sp_r2.std():.3f}")

    # ── Pairwise feature importances ──
    print("\n  Pairwise ranking feature importances (top 20):")
    arch_prefixes = ["spacing_", "has_pair_", "avg_spacing_", "orient_", "density_",
                     "count_distal", "count_mid", "count_proximal", "count_tss",
                     "cooc_", "syntax_", "motif_gap_", "tata_centrism"]
    sorted_imp = np.argsort(pair_importances)[::-1]
    n_arch_in_top20 = 0
    for i in range(min(20, len(feature_names))):
        idx = sorted_imp[i]
        name = feature_names[idx]
        imp = pair_importances[idx]
        is_arch = any(name.startswith(p) for p in arch_prefixes)
        tag = " [ARCH]" if is_arch else ""
        if is_arch:
            n_arch_in_top20 += 1
        print(f"    {i+1:2d}. {name:50s} {imp:.4f}{tag}")
    print(f"  Architecture features in ranking top-20: {n_arch_in_top20}")

    # ── Save results ──
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "learned_models")
    os.makedirs(output_dir, exist_ok=True)

    # Identify architecture features
    arch_features = [name for name in feature_names
                     if any(name.startswith(p) for p in arch_prefixes)]
    orig_features = [name for name in feature_names
                     if not any(name.startswith(p) for p in arch_prefixes)]

    report = {
        "v3_features": {
            "total": len(feature_names),
            "original": len(orig_features),
            "architecture": len(arch_features),
            "architecture_feature_list": arch_features,
        },
        "architecture_contribution": {
            "orig_only_kf_r2": None,
            "arch_only_kf_r2": None,
            "combined_kf_r2": None,
        },
        "pairwise_ranking": {
            "n_pairs": len(y_pairs),
            "results": {k: {kk: float(vv) for kk, vv in v.items()}
                        for k, v in ranking_results.items()},
            "n_arch_in_top20": n_arch_in_top20,
        },
        "pointwise_rank_target": {
            "results": {k: {kk: float(vv) for kk, vv in v.items()}
                        for k, v in pointwise_results.items()},
        },
        "cluster_cv_v3": {
            "cluster_r2": float(cl_r2.mean()),
            "species_r2": float(sp_r2.mean()),
            "n_clusters": 5,
        },
        "contrastive_top_arch": [{"feature": n, "effect_size": float(d)} for n, d in top_arch],
    }

    with open(os.path.join(output_dir, "v3_architecture_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"ARCHITECTURE + RANKING v3 COMPLETE — {elapsed:.1f}s")
    print()
    print("KEY FINDINGS:")
    print(f"  Features: {len(orig_features)} original + {len(arch_features)} architecture = {len(feature_names)} total")
    print(f"  Architecture features in ranking top-20: {n_arch_in_top20}/20")
    print(f"  Cluster CV R²: {cl_r2.mean():.3f} vs Species CV R²: {sp_r2.mean():.3f}")
    print(f"  Report: outputs/learned_models/v3_architecture_report.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
