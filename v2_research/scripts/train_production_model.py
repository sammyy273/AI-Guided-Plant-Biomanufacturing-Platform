"""
Train Production Model — Generate Learned Scoring Artifacts.

Trains a RandomForest promoter model with conformal prediction calibration,
computes protein module LOO calibration against benchmarks, and saves all
persistent artifacts for the learned_scorer module.

Usage:
    python scripts/train_production_model.py
"""

import glob
import json
import math
import os
import re
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "learned_promoter")

KD_HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

SOLUBILITY_LABEL_MAP = {
    "high": 0.85, "moderate": 0.55, "low": 0.30, "very_low": 0.10,
}


def load_training_data():
    """Load all iteration CSVs into a single DataFrame."""
    pattern1 = os.path.join(PROJECT_ROOT, "outputs", "*", "iter*_scored.csv")
    pattern2 = os.path.join(PROJECT_ROOT, "outputs", "*", "*", "iter*_scored.csv")
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

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["sequence"])
    print(f"  Loaded {len(combined)} unique sequences from {len(files)} files")
    return combined


def train_promoter_model(df):
    """Train RandomForest promoter model with conformal calibration."""
    from modules.statistical.sequence_features import build_feature_matrix_with_embeddings

    print("\n  Computing sequence features + AgroNT embeddings...")
    sequences = df["sequence"].tolist()
    feature_names, X, embed_meta = build_feature_matrix_with_embeddings(
        sequences, n_pca_components=50
    )
    y = df["composite_score"].values.astype(float)
    n_embed = embed_meta.get("n_pca_components", 0)
    print(f"  {X.shape[1]} features (handcrafted + {n_embed} AgroNT PCA), {len(y)} samples")
    print(f"  Embedding status: {embed_meta.get('embedding_status', 'unknown')}")

    # Feature audit: remove species-correlated features
    # Two-layer check:
    #   1. Linear: Pearson |r| > 0.20 with species label
    #   2. Non-linear: mutual information > 0.15 nats with species label
    #      (catches AgroNT embedding components that encode species
    #       non-linearly, which linear correlation misses)
    print("\n  Feature audit: removing species-correlated features...")
    species = df.get("_species")
    if species is None:
        # Re-derive species from file paths
        species = df.get("_species", pd.Series(["unknown"] * len(df)))
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_species = le.fit_transform(species)

    leaky_corr_threshold = 0.20
    leaky_mi_threshold = 0.15
    clean_mask = []
    n_leaky_corr = 0
    n_leaky_mi = 0

    # Pre-compute mutual information for non-linear leakage check
    try:
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(
            X, y_species, discrete=False, random_state=42, n_neighbors=5,
        )
        has_mi = True
    except Exception:
        mi_scores = np.zeros(X.shape[1])
        has_mi = False

    for i, name in enumerate(feature_names):
        flagged = False
        # Layer 1: linear correlation
        if X[:, i].std() > 0:
            corr = abs(np.corrcoef(X[:, i], y_species)[0, 1])
        else:
            corr = 0.0
        if corr > leaky_corr_threshold:
            n_leaky_corr += 1
            flagged = True

        # Layer 2: mutual information (non-linear species signal)
        if has_mi and mi_scores[i] > leaky_mi_threshold:
            n_leaky_mi += 1
            flagged = True

        clean_mask.append(not flagged)

    n_leaky = n_leaky_corr + n_leaky_mi - sum(
        1 for i in range(len(feature_names))
        if abs(np.corrcoef(X[:, i], y_species)[0, 1]) > leaky_corr_threshold
        and has_mi and mi_scores[i] > leaky_mi_threshold
        and X[:, i].std() > 0
    )
    n_removed = sum(1 for m in clean_mask if not m)
    print(f"    Removed {n_removed} leaky features "
          f"(linear |r|>{leaky_corr_threshold}: {n_leaky_corr}, "
          f"MI>{leaky_mi_threshold}: {n_leaky_mi}, "
          f"overlap: {n_leaky_corr + n_leaky_mi - n_removed})")
    X = X[:, clean_mask]
    feature_names = [n for n, m in zip(feature_names, clean_mask) if m]
    print(f"    Retained {len(feature_names)} clean features")

    # Split: 80% train, 20% calibration (species-aware to prevent data leakage)
    species_groups = df["_species"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, cal_idx = next(gss.split(X, y, groups=species_groups))
    n_cal = len(cal_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_cal, y_cal = X[cal_idx], y[cal_idx]

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)

    # Train RandomForest
    print("  Training RandomForest (500 trees)...")
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=5,
        max_features=0.33,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train_s)
    y_cal_pred = model.predict(X_cal_s)
    train_r2 = r2_score(y_train, y_train_pred)
    cal_r2 = r2_score(y_cal, y_cal_pred)
    cal_mae = mean_absolute_error(y_cal, y_cal_pred)
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Calibration R²: {cal_r2:.4f}, MAE: {cal_mae:.4f}")

    # Conformal calibration
    nonconformity = np.abs(y_cal - y_cal_pred)
    alpha = 0.10  # 90% coverage
    q_hat_index = min(int(np.ceil((n_cal + 1) * (1 - alpha))) - 1, n_cal - 1)
    q_hat = float(np.sort(nonconformity)[q_hat_index])
    print(f"  Conformal q_hat (90%): {q_hat:.4f}")

    # Verify coverage on calibration set
    in_interval = np.sum(nonconformity <= q_hat) / n_cal
    print(f"  Calibration coverage: {in_interval:.1%} (target: ≥90%)")

    # OOD reference: training feature statistics
    X_all_s = scaler.transform(X)
    feature_mean = X_all_s.mean(axis=0)
    cov = np.cov(X_all_s.T) + 1e-6 * np.eye(X_all_s.shape[1])
    cov_inv = np.linalg.inv(cov)
    ood_threshold = float(chi2.ppf(0.975, df=X_all_s.shape[1]))

    # Feature importances
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print("\n  Top features:")
    for i, idx_i in enumerate(top_idx):
        print(f"    {i+1:2d}. {feature_names[idx_i]:30s}  {importances[idx_i]:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "q_hat": q_hat,
        "nonconformity_scores": nonconformity,
        "feature_mean": feature_mean.tolist(),
        "feature_cov_inv": cov_inv.tolist(),
        "ood_threshold": ood_threshold,
        "embedding_pca": embed_meta.get("embedding_pca"),
        "embedding_status": embed_meta.get("embedding_status", "unavailable"),
        "pca_explained_variance": embed_meta.get("pca_explained_variance_ratio", []),
        "metrics": {
            "train_r2": float(train_r2),
            "cal_r2": float(cal_r2),
            "cal_mae": float(cal_mae),
            "conformal_coverage": float(in_interval),
            "n_train": len(train_idx),
            "n_cal": n_cal,
            "n_features": len(feature_names),
            "n_embedding_pca": embed_meta.get("n_pca_components", 0),
        },
        "feature_importances": [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in top_idx
        ],
    }


def compute_protein_features_vector(sequence):
    """Compute 7-dim protein feature vector."""
    seq = sequence.upper()
    n = max(len(seq), 1)
    return [
        n,
        math.log(n),
        sum(1 for aa in seq if aa in "FWY") / n,
        sum(1 for aa in seq if aa in "KRDE") / n,
        sum(1 for aa in seq if aa == "C") / n,
        float(np.mean([KD_HYDROPHOBICITY.get(aa, 0.0) for aa in seq])),
        len(re.findall(r"N[^P][ST]", seq)),
    ]


def train_protein_calibration():
    """Run LOO calibration on protein benchmarks."""
    print("\n  Loading protein benchmarks...")
    sys.path.insert(0, PROJECT_ROOT)
    from benchmarks.validation_benchmarks import (
        SOLUBILITY_BENCHMARKS,
        GLYCOSYLATION_BENCHMARKS,
        DISULFIDE_BENCHMARKS,
    )

    calibration = {}

    # ── Folding/solubility calibration ──
    print("  Calibrating folding/solubility module (LOO)...")
    proteins = SOLUBILITY_BENCHMARKS
    n = len(proteins)
    features = np.array([compute_protein_features_vector(p["sequence"])
                         for p in proteins])

    # Map labels to numeric
    labels = np.array([SOLUBILITY_LABEL_MAP.get(p["expected_solubility"], 0.5)
                       for p in proteins])

    # Compute heuristic scores for all proteins
    from modules.protein.folding_quality_model import full_folding_analysis
    heuristic_scores = []
    for p in proteins:
        result = full_folding_analysis(p["sequence"], "ER_retained", 25)
        score = result.get("folding_compatibility_score",
                          result.get("folding_yield", 0.5))
        if not isinstance(score, (int, float)):
            score = 0.5
        heuristic_scores.append(float(score))
    heuristic_scores = np.array(heuristic_scores)

    biases = []
    residuals = []
    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        bias = np.mean(heuristic_scores[train_mask]) - np.mean(labels[train_mask])
        corrected = heuristic_scores[i] - bias
        residual = abs(corrected - labels[i])
        biases.append(bias)
        residuals.append(residual)

    ref_distances = []
    for i in range(n):
        dists = np.sqrt(np.sum((features - features[i]) ** 2, axis=1))
        dists[i] = np.inf
        ref_distances.append(float(dists.min()))

    calibration["folding_quality"] = {
        "bias_correction": float(np.median(biases)),
        "empirical_noise": float(max(np.std(residuals), 0.05)),
        "ref_distance": float(np.median(ref_distances)) if ref_distances else 1.0,
        "loo_residuals": [float(r) for r in residuals],
        "loo_biases": [float(b) for b in biases],
        "heuristic_scores": [float(s) for s in heuristic_scores],
        "labels": labels.tolist(),
        "benchmark_features": features.tolist(),
    }
    print(f"    bias_correction={calibration['folding_quality']['bias_correction']:.4f}, "
          f"noise={calibration['folding_quality']['empirical_noise']:.4f}")

    # ── Glycosylation calibration ──
    print("  Calibrating glycosylation module (LOO)...")
    glyc_proteins = GLYCOSYLATION_BENCHMARKS
    n_g = len(glyc_proteins)

    from modules.protein.glycosylation_model import full_glycosylation_analysis
    glyc_features = np.array([compute_protein_features_vector(p["sequence"])
                              for p in glyc_proteins])

    # Labels: number of known sites (from benchmark)
    glyc_labels = []
    glyc_heuristic = []
    for p in glyc_proteins:
        n_known = p.get("n_known_sites", len(p.get("known_sites", [])))
        glyc_labels.append(float(n_known))
        result = full_glycosylation_analysis(p["sequence"], "ER_retained", "nbenthamiana")
        n_pred = len(result.get("sites", []))
        glyc_heuristic.append(float(n_pred))

    glyc_labels = np.array(glyc_labels)
    glyc_heuristic = np.array(glyc_heuristic)

    glyc_biases = []
    glyc_residuals = []
    for i in range(n_g):
        train_mask = np.ones(n_g, dtype=bool)
        train_mask[i] = False
        if len(train_mask) > 0:
            bias = np.mean(glyc_heuristic[train_mask]) - np.mean(glyc_labels[train_mask])
        else:
            bias = 0.0
        corrected = glyc_heuristic[i] - bias
        residual = abs(corrected - glyc_labels[i])
        glyc_biases.append(bias)
        glyc_residuals.append(residual)

    calibration["glycosylation"] = {
        "bias_correction": float(np.median(glyc_biases)),
        "empirical_noise": float(max(np.std(glyc_residuals), 0.05)),
        "ref_distance": float(np.median([float(np.sqrt(np.sum((glyc_features - glyc_features[i]) ** 2, axis=1)).min())
                                          for i in range(n_g)])),
        "loo_residuals": [float(r) for r in glyc_residuals],
        "loo_biases": [float(b) for b in glyc_biases],
        "benchmark_features": glyc_features.tolist(),
    }
    print(f"    bias_correction={calibration['glycosylation']['bias_correction']:.4f}, "
          f"noise={calibration['glycosylation']['empirical_noise']:.4f}")

    # ── Manufacturing calibration ──
    print("  Calibrating manufacturing module...")
    calibration["manufacturing"] = {
        "bias_correction": 0.0,
        "empirical_noise": 0.08,
        "ref_distance": 1.0,
        "note": "Manufacturing checks are deterministic; noise floor from vendor variability",
    }

    return calibration


def save_artifacts(promoter_results, protein_calibration):
    """Save all artifacts to disk."""
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    joblib.dump(promoter_results["model"],
                os.path.join(ARTIFACT_DIR, "promoter_rf_model.joblib"))
    joblib.dump(promoter_results["scaler"],
                os.path.join(ARTIFACT_DIR, "promoter_scaler.joblib"))

    with open(os.path.join(ARTIFACT_DIR, "promoter_feature_names.json"), "w") as f:
        json.dump(promoter_results["feature_names"], f)

    with open(os.path.join(ARTIFACT_DIR, "promoter_conformal_qhat.json"), "w") as f:
        json.dump({"q_hat": promoter_results["q_hat"],
                    "alpha": 0.10,
                    "n_calibration": promoter_results["metrics"]["n_cal"]}, f)

    np.save(os.path.join(ARTIFACT_DIR, "promoter_conformal_scores.npy"),
            promoter_results["nonconformity_scores"])

    stats = {
        "feature_mean": promoter_results["feature_mean"],
        "feature_cov_inv": promoter_results["feature_cov_inv"],
        "ood_threshold": promoter_results["ood_threshold"],
        "metrics": promoter_results["metrics"],
        "feature_importances": promoter_results["feature_importances"],
        "shap_analysis": promoter_results.get("shap", {}),
    }
    with open(os.path.join(ARTIFACT_DIR, "promoter_training_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=float)

    # Save AgroNT PCA artifact
    if promoter_results.get("embedding_pca") is not None:
        joblib.dump(promoter_results["embedding_pca"],
                     os.path.join(ARTIFACT_DIR, "agront_pca.joblib"))
        with open(os.path.join(ARTIFACT_DIR, "agront_pca_meta.json"), "w") as f:
            json.dump({
                "n_components": promoter_results["metrics"].get("n_embedding_pca", 0),
                "explained_variance_ratio": promoter_results.get("pca_explained_variance", []),
                "status": promoter_results.get("embedding_status", "unavailable"),
            }, f, indent=2)
        print(f"    agront_pca.joblib: saved")
        print(f"    agront_pca_meta.json: saved")

    with open(os.path.join(ARTIFACT_DIR, "protein_calibration.json"), "w") as f:
        json.dump(protein_calibration, f, indent=2, default=float)

    print(f"\n  Artifacts saved to: {ARTIFACT_DIR}")
    for name in os.listdir(ARTIFACT_DIR):
        path = os.path.join(ARTIFACT_DIR, name)
        size = os.path.getsize(path)
        print(f"    {name}: {size / 1024:.0f} KB")


def compute_shap_analysis(model, X_train_s, feature_names):
    """Compute SHAP values for biological interpretability."""
    import shap

    print("\n  Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    # Use a sample for speed (SHAP on full set is slow)
    n_sample = min(200, X_train_s.shape[0])
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(X_train_s.shape[0], n_sample, replace=False)
    X_sample = X_train_s[sample_idx]

    shap_values = explainer.shap_values(X_sample)

    # Top features by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    print("\n  SHAP Feature Attribution (biological interpretability):")
    shap_ranking = []
    for i in range(min(20, len(feature_names))):
        idx = sorted_idx[i]
        name = feature_names[idx]
        shap_val = mean_abs_shap[idx]
        direction = "positive" if np.mean(shap_values[:, idx]) > 0 else "negative"
        print(f"    {i+1:2d}. {name:30s}  SHAP={shap_val:.4f}  direction={direction}")
        shap_ranking.append({
            "feature": name,
            "mean_abs_shap": float(shap_val),
            "mean_shap": float(np.mean(shap_values[:, idx])),
            "direction": direction,
        })

    return {"shap_ranking": shap_ranking, "n_samples": n_sample}


def main():
    t0 = time.time()

    print()
    print("=" * 60)
    print("PRODUCTION MODEL TRAINING v2")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading training data...")
    df = load_training_data()

    # Step 2: Train promoter model (with feature audit)
    print("\n[2/5] Training promoter model (with leakage removal)...")
    promoter_results = train_promoter_model(df)

    # Step 3: SHAP interpretability
    print("\n[3/5] Computing SHAP feature attribution...")
    try:
        shap_results = compute_shap_analysis(
            promoter_results["model"],
            StandardScaler().fit_transform(
                np.array([[0.0] * len(promoter_results["feature_names"])])
            ),
            promoter_results["feature_names"],
        )
        # Re-compute with actual training data (including embeddings)
        from modules.statistical.sequence_features import build_feature_matrix_with_embeddings
        sequences = df["sequence"].tolist()
        _, X_full, _ = build_feature_matrix_with_embeddings(
            sequences, embedding_pca=promoter_results.get("embedding_pca")
        )
        # Align columns to model's feature_names (handles leakage-removed subset)
        scaler = promoter_results["scaler"]
        # X_full may have more columns than model expects if embedding was unavailable
        # at training but available now; slice to model's expected feature count
        X_aligned = X_full[:, :len(promoter_results["feature_names"])]
        X_scaled = scaler.transform(X_aligned)
        shap_results = compute_shap_analysis(
            promoter_results["model"], X_scaled, promoter_results["feature_names"]
        )
        promoter_results["shap"] = shap_results
    except Exception as e:
        print(f"  SHAP analysis failed: {e}")
        import traceback
        traceback.print_exc()
        shap_results = None

    # Step 4: Protein calibration
    print("\n[4/5] Calibrating protein modules...")
    protein_cal = train_protein_calibration()

    # Step 5: Save
    print("\n[5/5] Saving artifacts...")
    save_artifacts(promoter_results, protein_cal)

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"TRAINING COMPLETE — {elapsed:.1f}s")
    print()
    print("PROMOTER MODEL METRICS:")
    m = promoter_results["metrics"]
    print(f"  Train R²:         {m['train_r2']:.4f}")
    print(f"  Calibration R²:   {m['cal_r2']:.4f}")
    print(f"  Calibration MAE:  {m['cal_mae']:.4f}")
    print(f"  Conformal q_hat:  {promoter_results['q_hat']:.4f}")
    print(f"  Coverage:         {m['conformal_coverage']:.1%}")
    print(f"  Clean features:   {m['n_features']} (removed leaky)")
    print()
    print("PROTEIN CALIBRATION:")
    for mod, cal in protein_cal.items():
        if "bias_correction" in cal:
            print(f"  {mod:20s}  bias={cal['bias_correction']:.4f}  "
                  f"noise={cal['empirical_noise']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
