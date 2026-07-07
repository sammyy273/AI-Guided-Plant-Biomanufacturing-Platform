#!/usr/bin/env python3
"""
STEP 4: Integrate simple ML regressor for expression potential estimation.

Trains a RandomForestRegressor on computed promoter features to predict
the internal composite score (NOT real expression). Honest about limitations:
the model learns to recapitulate the heuristic scoring, not biological reality.

INPUT FEATURES:
  - motif counts (TATA, CAAT, GC-box, as-1, DOF)
  - GC content
  - sequence length
  - architecture features (TATA+CAAT present, positioning)
  - k-mer frequencies (4-mers, top variables)

TARGET: internal composite score

OUTPUTS:
  outputs/phase1/trained_model.pkl
  outputs/phase1/ml_predictions.csv
  outputs/phase1/ml_model_metrics.txt
"""

import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "modules"))

OUTPUT_DIR = BASE_DIR / "outputs" / "phase1"
PROMOTER_DIR = BASE_DIR / "data" / "promoters"

# ── Motif patterns ─────────────────────────────────────────────────────────

MOTIF_PATTERNS = {
    "TATA_box": ["TATAAA", "TATATA", "TATACA", "TATAGA"],
    "CAAT_box": ["CCAAT", "CAAAT"],
    "GC_box": ["GGGCGG", "CCGCCC"],
    "as1_element": ["TGACG", "CGTCA"],
    "DOF_site": ["AAAG", "CTTT"],
    "ocs_like": ["TGACGTAAG", "CTTACGTCA"],
    "W_box": ["TTGAC", "TTGACC"],
    "MYB_site": ["CAACTG", "CAGTTG"],
}


def count_motif(seq, patterns):
    return sum(seq.upper().count(p) for p in patterns)


def compute_4mer_frequencies(seq, top_k=20):
    """Compute top-k most variable 4-mer frequencies."""
    s = seq.upper()
    counts = Counter()
    for i in range(len(s) - 3):
        kmer = s[i:i+4]
        if all(c in "ACGT" for c in kmer):
            counts[kmer] += 1

    total = sum(counts.values()) if counts else 1
    freqs = {k: v / total for k, v in counts.items()}
    return freqs


def extract_features(seq):
    """Extract all features for ML input."""
    s = seq.upper()
    length = len(s)
    gc = (s.count("G") + s.count("C")) / length if length > 0 else 0

    # Motif counts
    motif_features = {}
    for name, patterns in MOTIF_PATTERNS.items():
        motif_features[f"motif_{name}"] = count_motif(s, patterns)

    # Architecture
    proximal = s[-50:] if length >= 50 else s
    upstream = s[-120:] if length >= 120 else s

    has_tata = int(any(p in proximal for p in MOTIF_PATTERNS["TATA_box"]))
    has_caat = int(any(p in upstream for p in MOTIF_PATTERNS["CAAT_box"]))
    has_both = has_tata and has_caat

    # Spacing: distance between first TATA and first CAAT in last 120bp
    spacing = 0
    if has_tata and has_caat:
        region = s[-120:] if length >= 120 else s
        tata_pos = None
        caat_pos = None
        for p in MOTIF_PATTERNS["TATA_box"]:
            idx = region.find(p)
            if idx >= 0:
                tata_pos = idx
                break
        for p in MOTIF_PATTERNS["CAAT_box"]:
            idx = region.find(p)
            if idx >= 0:
                caat_pos = idx
                break
        if tata_pos is not None and caat_pos is not None:
            spacing = abs(tata_pos - caat_pos)

    # CG density
    cg_density = s.count("CG") / length if length > 0 else 0

    # Mono/tri nucleotide stats
    a_frac = s.count("A") / length
    t_frac = s.count("T") / length

    features = {
        "length": length,
        "gc_content": gc,
        "has_tata": has_tata,
        "has_caat": has_caat,
        "has_both_arch": int(has_both),
        "tata_caat_spacing": spacing,
        "cg_density": cg_density,
        "a_frac": a_frac,
        "t_frac": t_frac,
        "at_content": a_frac + t_frac,
        **motif_features,
    }

    return features


def compute_target_score(features):
    """Compute the target composite score (same formula as benchmark_classification.py)."""
    arch_bonus = 0.5 if features["has_both_arch"] else (
        0.25 if features["has_tata"] else (
            0.15 if features["has_caat"] else 0.0
        )
    )

    total_motifs = sum(v for k, v in features.items() if k.startswith("motif_"))
    motif_score = min(1.0, total_motifs / 30)
    gc_balance = max(0, 1 - abs(features["gc_content"] - 0.38) / 0.25)

    composite = (
        0.30 * arch_bonus +
        0.25 * motif_score +
        0.20 * gc_balance +
        0.15 * features["has_tata"] +
        0.10 * (1 - min(1, features["cg_density"] * 3))
    )

    return round(composite, 4)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 4: ML Regressor for Expression Potential")
    print("=" * 60)

    species_list = ["arabidopsis", "rice", "tomato"]
    n_per_species = 600
    all_records = []

    for species in species_list:
        from Bio import SeqIO

        fasta_path = PROMOTER_DIR / f"{species}_promoters_1kb.fasta"
        if not fasta_path.exists():
            print(f"  {species}: FASTA not found, skipping")
            continue

        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        print(f"  {species}: loaded {len(records)} promoters")

        rng = np.random.RandomState(42)
        indices = rng.choice(len(records), min(n_per_species, len(records)), replace=False)

        for idx in indices:
            seq = str(records[idx].seq)
            features = extract_features(seq)
            target = compute_target_score(features)
            all_records.append({
                "species": species,
                "sequence_id": records[idx].id,
                **features,
                "composite_score": target,
            })

    if not all_records:
        print("ERROR: No data processed.")
        return

    df = pd.DataFrame(all_records)
    print(f"\n  Total dataset: {len(df)} promoters")
    print(f"  Features: {len([c for c in df.columns if c not in ['species', 'sequence_id', 'composite_score']])}")

    # Prepare ML data
    feature_cols = [c for c in df.columns if c not in ["species", "sequence_id", "composite_score"]]
    X = df[feature_cols].fillna(0).values
    y = df["composite_score"].values

    # Split: 70% train, 30% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"  Train: {len(X_train)}, Validation: {len(X_val)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # ── Model 1: Random Forest ─────────────────────────────────────────
    print("\n  Training RandomForestRegressor...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    rf_pred_train = rf.predict(X_train)
    rf_pred_val = rf.predict(X_val)

    rf_r2_train = r2_score(y_train, rf_pred_train)
    rf_r2_val = r2_score(y_val, rf_pred_val)
    rf_mae_train = mean_absolute_error(y_train, rf_pred_train)
    rf_mae_val = mean_absolute_error(y_val, rf_pred_val)

    print(f"    RF Train: R²={rf_r2_train:.4f}, MAE={rf_mae_train:.4f}")
    print(f"    RF Val:   R²={rf_r2_val:.4f}, MAE={rf_mae_val:.4f}")

    # ── Model 2: Ridge Regression (baseline) ───────────────────────────
    print("  Training Ridge regression (baseline)...")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)

    ridge_pred_val = ridge.predict(X_val_scaled)
    ridge_r2_val = r2_score(y_val, ridge_pred_val)
    ridge_mae_val = mean_absolute_error(y_val, ridge_pred_val)

    print(f"    Ridge Val: R²={ridge_r2_val:.4f}, MAE={ridge_mae_val:.4f}")

    # ── Select best model ──────────────────────────────────────────────
    best_model = rf if rf_r2_val >= ridge_r2_val else ridge
    best_name = "RandomForest" if rf_r2_val >= ridge_r2_val else "Ridge"
    best_r2 = max(rf_r2_val, ridge_r2_val)
    best_mae = rf_mae_val if best_name == "RandomForest" else ridge_mae_val

    print(f"\n  Best model: {best_name} (R²={best_r2:.4f})")

    # ── Feature importance ──────────────────────────────────────────────
    if best_name == "RandomForest":
        importances = rf.feature_importances_
        feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
        print("  Top 10 features:")
        for fname, imp in feat_imp[:10]:
            print(f"    {fname:30s}: {imp:.4f}")

    # ── Save model ─────────────────────────────────────────────────────
    model_path = OUTPUT_DIR / "trained_model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump({
            "model": best_model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "model_type": best_name,
            "train_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, fh)
    print(f"\n  Saved model: {model_path}")

    # ── Generate predictions on full dataset ───────────────────────────
    if best_name == "Ridge":
        X_full = scaler.transform(X)
        predictions = best_model.predict(X_full)
    else:
        predictions = best_model.predict(X)

    df["ml_expression_estimate"] = np.round(predictions, 4)
    df["ml_note"] = "ML-based expression potential estimate (trained on internal proxy targets)"

    pred_path = OUTPUT_DIR / "ml_predictions.csv"
    df.to_csv(pred_path, index=False)
    print(f"  Saved predictions: {pred_path} ({len(df)} rows)")

    # ── Metrics report ─────────────────────────────────────────────────
    metrics = [
        "ML MODEL METRICS REPORT",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "MODEL: " + best_name,
        "",
        "RANDOM FOREST HYPERPARAMETERS:" if best_name == "RandomForest" else "RIDGE HYPERPARAMETERS:",
    ]

    if best_name == "RandomForest":
        metrics.extend([
            f"  n_estimators: 200",
            f"  max_depth: 15",
            f"  min_samples_leaf: 5",
            f"  random_state: 42",
        ])
    else:
        metrics.extend([
            f"  alpha: 1.0",
            f"  random_state: 42",
        ])

    metrics.extend([
        "",
        "DATASET:",
        f"  Total samples: {len(df)}",
        f"  Train split: {len(X_train)} (70%)",
        f"  Validation split: {len(X_val)} (30%)",
        f"  Features: {len(feature_cols)}",
        f"  Species: {species_list}",
        "",
        "RESULTS:",
        f"  RandomForest — Train R²: {rf_r2_train:.4f}, Train MAE: {rf_mae_train:.4f}",
        f"  RandomForest — Val   R²: {rf_r2_val:.4f}, Val   MAE: {rf_mae_val:.4f}",
        f"  Ridge       — Val   R²: {ridge_r2_val:.4f}, Val   MAE: {ridge_mae_val:.4f}",
        f"  Best model: {best_name} (Val R²={best_r2:.4f})",
        "",
        "FEATURE IMPORTANCE (Random Forest):",
    ])

    if best_name == "RandomForest":
        for fname, imp in feat_imp:
            metrics.append(f"  {fname:30s}: {imp:.4f}")

    metrics.extend([
        "",
        "IMPORTANT DISCLAIMERS:",
        "  1. The ML model is trained to predict the INTERNAL composite score.",
        "     This is a proxy target derived from heuristic cis-element analysis,",
        "     NOT real gene expression data.",
        "  2. High R² indicates the model learns the heuristic scoring function well,",
        "     NOT that it predicts biological expression.",
        "  3. No experimental validation has been performed.",
        "  4. The model should be interpreted as an automated scoring approximation,",
        "     not a biological predictor.",
        "",
        "REPRODUCIBILITY:",
        "  Random seeds: fixed (random_state=42 for all components)",
        "  Data source: curated TSS-anchored promoter datasets (Step 1)",
    ])

    metrics_path = OUTPUT_DIR / "ml_model_metrics.txt"
    with open(metrics_path, "w") as fh:
        fh.write("\n".join(metrics))
    print(f"  Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
