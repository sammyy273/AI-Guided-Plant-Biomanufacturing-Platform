#!/usr/bin/env python3
"""
STEP 2: Localization Upgrade with ML Comparison.

Attempts DeepLoc 2.0 ML prediction, falls back to enhanced heuristic with
multiple signal detection. Creates consensus between ML and heuristic.

For the target protein (Hyaluronidase/SPAM1), detects:
- Signal peptide
- Transmembrane regions
- ER-retention signals (KDEL/HDEL)
- GPI-anchor signals
- Nuclear localization signals

OUTPUTS:
  outputs/phase2/localization_ml_comparison.csv
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "modules"))

OUTPUT_DIR = BASE_DIR / "outputs" / "phase2"
PROTEIN_DIR = BASE_DIR / "data" / "protein"


def load_target_protein():
    """Load the target protein sequence."""
    from Bio import SeqIO
    fasta_files = list(PROTEIN_DIR.glob("*.fasta"))
    if not fasta_files:
        return None, None

    for fasta_file in fasta_files:
        for record in SeqIO.parse(str(fasta_file), "fasta"):
            return str(record.seq), record.id
    return None, None


def predict_deeploc(sequence):
    """Attempt DeepLoc 2.0 prediction via HuggingFace."""
    result = {
        "method": "DeepLoc_2.0",
        "available": False,
        "prediction": None,
        "scores": {},
        "error": None,
    }

    # Try known model paths
    model_paths = [
        "Jensen-hub/DeepLoc-2",
        "jeppejorgensen/deeploc-2",
        "JensenLab/DeepLoc-2",
    ]

    for model_path in model_paths:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()

            inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).squeeze()

            compartments = [
                "Nucleus", "Cytoplasm", "Extracellular", "Mitochondrion",
                "Cell membrane", "ER", "Chloroplast", "Golgi apparatus",
                "Lysosome/Vacuole", "Peroxisome"
            ]

            scores = {comp: float(probs[i]) for i, comp in enumerate(compartments)}
            top_idx = int(probs.argmax())
            top_compartment = compartments[top_idx]
            top_score = float(probs[top_idx])

            result["available"] = True
            result["prediction"] = top_compartment
            result["top_score"] = top_score
            result["scores"] = scores
            result["model_path"] = model_path
            return result

        except Exception as e:
            result["error"] = str(e)[:200]
            continue

    return result


def enhanced_heuristic_localization(sequence):
    """
    Enhanced heuristic localization prediction.
    Detects: signal peptide, transmembrane, ER-retention, GPI-anchor, NLS.
    """
    seq = sequence.upper()
    length = len(seq)
    signals = {}

    # 1. Signal peptide detection (SignalP-like logic)
    n_terminal = seq[:50]
    hydrophobic_aas = set("AILMFWV")
    pos_charged = set("KR")

    # N-region: positive charges in first 5 residues
    n_region_pos = sum(1 for aa in n_terminal[:5] if aa in pos_charged)

    # H-region: hydrophobic core in residues 5-25
    h_region = n_terminal[5:25] if len(n_terminal) >= 25 else n_terminal[5:]
    h_hydrophobic = sum(1 for aa in h_region if aa in hydrophobic_aas)

    # Signal peptide score
    has_signal_peptide = n_region_pos >= 1 and h_hydrophobic >= 8

    # Predict cleavage site (AXA motif search in positions 15-30)
    cleavage_site = None
    for i in range(15, min(35, len(seq) - 2)):
        if seq[i] in "ASG" and seq[i+2] in "ASG":
            cleavage_site = i + 1
            break

    signals["signal_peptide"] = {
        "detected": has_signal_peptide,
        "n_region_positive": n_region_pos,
        "h_region_hydrophobic_count": h_hydrophobic,
        "cleavage_site": cleavage_site,
    }

    # 2. Transmembrane region detection
    tm_regions = []
    window = 19
    for i in range(0, length - window + 1, 3):
        segment = seq[i:i+window]
        hydrophobic_frac = sum(1 for aa in segment if aa in hydrophobic_aas) / window
        if hydrophobic_frac >= 0.65:
            tm_regions.append({
                "start": i + 1,
                "end": i + window,
                "hydrophobic_fraction": round(hydrophobic_frac, 3),
                "sequence": segment,
            })

    signals["transmembrane_regions"] = tm_regions

    # 3. ER-retention signals
    c_terminal_4 = seq[-4:] if length >= 4 else ""
    c_terminal_6 = seq[-6:] if length >= 6 else ""

    has_kdel = "KDEL" in c_terminal_4 or "HDEL" in c_terminal_4
    signals["er_retention"] = {
        "detected": has_kdel,
        "c_terminal_4": c_terminal_4,
    }

    # 4. GPI-anchor signal (hydrophobic C-terminus)
    c_terminal_20 = seq[-20:] if length >= 20 else seq
    c_hydrophobic = sum(1 for aa in c_terminal_20 if aa in hydrophobic_aas)
    has_gpi_signal = c_hydrophobic >= 12 and len(tm_regions) > 0

    signals["gpi_anchor"] = {
        "detected": has_gpi_signal,
        "c_terminal_hydrophobic_count": c_hydrophobic,
    }

    # 5. Nuclear localization signals (NLS)
    nls_patterns = ["KKKR", "KKRK", "PKKKRKV", "KRPAATKKAGQAKKK", "KKKK", "RRRR"]
    nls_hits = []
    for pattern in nls_patterns:
        idx = seq.find(pattern)
        while idx >= 0:
            nls_hits.append({"pattern": pattern, "position": idx + 1})
            idx = seq.find(pattern, idx + 1)

    signals["nuclear_localization"] = {
        "detected": len(nls_hits) > 0,
        "hits": nls_hits[:5],  # cap at 5
    }

    # 6. Vacuolar sorting signals
    vacuolar_motifs = ["NPIR", "NPIRP", "LQRK", "NPIL"]
    vacuolar_hits = [m for m in vacuolar_motifs if m in seq[:100]]
    signals["vacuolar_sorting"] = {
        "detected": len(vacuolar_hits) > 0,
        "hits": vacuolar_hits,
    }

    # ── Decision logic ─────────────────────────────────────────────────

    prediction = "cytoplasm"
    confidence = 0.5
    reasoning = []

    if has_signal_peptide:
        reasoning.append(f"Signal peptide detected (H-region hydrophobic={h_hydrophobic}, cleavage@{cleavage_site})")

        if has_kdel:
            prediction = "ER"
            confidence = 0.92
            reasoning.append("ER-retention signal (KDEL/HDEL) detected")
        elif has_gpi_signal and len(tm_regions) >= 1:
            prediction = "extracellular"
            confidence = 0.82
            reasoning.append("GPI-anchor signal + C-terminal TM region → cell surface/extracellular")
        elif len(tm_regions) >= 2:
            prediction = "membrane"
            confidence = 0.78
            reasoning.append(f"{len(tm_regions)} TM regions → integral membrane protein")
        elif len(tm_regions) == 1 and tm_regions[0]["start"] > length * 0.6:
            prediction = "extracellular"
            confidence = 0.80
            reasoning.append("Signal peptide + single C-terminal TM → secreted/GPI-anchored")
        else:
            prediction = "extracellular"
            confidence = 0.75
            reasoning.append("Signal peptide → secretory pathway (default extracellular)")
    else:
        reasoning.append("No signal peptide detected")

        if nls_hits:
            prediction = "nucleus"
            confidence = 0.65
            reasoning.append(f"NLS detected ({len(nls_hits)} hits)")
        elif len(tm_regions) >= 2:
            prediction = "membrane"
            confidence = 0.70
            reasoning.append(f"{len(tm_regions)} TM regions without signal peptide")
        else:
            prediction = "cytoplasm"
            confidence = 0.60
            reasoning.append("Default: cytosolic (no targeting signals)")

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "method": "enhanced_heuristic",
        "reasoning": "; ".join(reasoning),
        "signals": signals,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 2: Localization Upgrade with ML Comparison")
    print("=" * 60)

    # Load target protein
    sequence, seq_id = load_target_protein()
    if sequence is None:
        print("  ERROR: No target protein found in data/protein/")
        return

    print(f"  Target protein: {seq_id}")
    print(f"  Sequence length: {len(sequence)} aa")
    print()

    # 1. Try DeepLoc ML prediction
    print("  Attempting DeepLoc 2.0 ML prediction...")
    deeploc_result = predict_deeploc(sequence)

    if deeploc_result["available"]:
        print(f"    SUCCESS: {deeploc_result['prediction']} (score={deeploc_result['top_score']:.4f})")
        print(f"    Model: {deeploc_result['model_path']}")
    else:
        print(f"    FAILED: {deeploc_result['error']}")
        print("    Status: DeepLoc NOT AVAILABLE — using enhanced heuristic only")

    # 2. Run enhanced heuristic
    print("\n  Running enhanced heuristic localization...")
    heuristic_result = enhanced_heuristic_localization(sequence)

    print(f"    Prediction: {heuristic_result['prediction']}")
    print(f"    Confidence: {heuristic_result['confidence']}")
    print(f"    Reasoning: {heuristic_result['reasoning']}")

    # Print detected signals
    signals = heuristic_result["signals"]
    for sig_name, sig_data in signals.items():
        if sig_name == "transmembrane_regions":
            print(f"    TM regions: {len(sig_data)} detected")
            for tm in sig_data:
                print(f"      {tm['start']}-{tm['end']}: hydro={tm['hydrophobic_fraction']:.2f} ({tm['sequence']})")
        else:
            print(f"    {sig_name}: {sig_data}")

    # 3. Load existing pipeline predictions for comparison
    existing_predictions = {}
    for species in ["nbenthamiana", "rice", "tomato"]:
        report_path = BASE_DIR / "outputs" / f"final_report_{species}.json"
        if report_path.exists():
            with open(report_path) as fh:
                report = json.load(fh)
            if "protein_analysis" in report:
                pa = report["protein_analysis"]
                existing_predictions[species] = {
                    "localization": pa.get("localization", "unknown"),
                    "localization_method": pa.get("localization_method", "unknown"),
                    "localization_confidence": pa.get("localization_confidence", 0),
                }

    # 4. Build consensus
    predictions_list = []

    # Heuristic prediction
    predictions_list.append({
        "method": "enhanced_heuristic",
        "prediction": heuristic_result["prediction"],
        "confidence": heuristic_result["confidence"],
    })

    # DeepLoc prediction (if available)
    if deeploc_result["available"]:
        predictions_list.append({
            "method": "DeepLoc_2.0",
            "prediction": deeploc_result["prediction"],
            "confidence": deeploc_result["top_score"],
        })

    # Existing pipeline prediction
    for species, pred in existing_predictions.items():
        predictions_list.append({
            "method": f"pipeline_original ({species})",
            "prediction": pred["localization"],
            "confidence": pred["localization_confidence"],
        })

    # Consensus logic
    all_preds = [p["prediction"] for p in predictions_list]
    pred_set = set(all_preds)

    if len(pred_set) == 1:
        consensus = all_preds[0]
        consensus_confidence = "HIGH"
        consensus_note = "All methods agree"
    elif len(pred_set) <= 2:
        # Check if predictions are in same compartment family
        secretory = {"extracellular", "ER", "membrane", "cell membrane", "Golgi apparatus"}
        if pred_set.issubset(secretory):
            consensus = "secretory_pathway"
            consensus_confidence = "MEDIUM"
            consensus_note = "Methods agree on secretory pathway but differ on final compartment"
        else:
            consensus = "ambiguous"
            consensus_confidence = "LOW"
            consensus_note = f"Methods disagree: {', '.join(pred_set)}"
    else:
        consensus = "ambiguous"
        consensus_confidence = "LOW"
        consensus_note = f"Multiple conflicting predictions: {', '.join(pred_set)}"

    print(f"\n  Consensus: {consensus} (confidence={consensus_confidence})")
    print(f"  Note: {consensus_note}")

    # 5. Build output
    output_rows = []

    for p in predictions_list:
        output_rows.append({
            "target_protein": seq_id,
            "method": p["method"],
            "prediction": p["prediction"],
            "confidence": p["confidence"],
            "consensus": consensus,
            "consensus_confidence": consensus_confidence,
        })

    # Add detailed signal analysis
    output_rows.append({
        "target_protein": seq_id,
        "method": "signal_analysis",
        "prediction": f"signal_peptide={signals['signal_peptide']['detected']}, "
                      f"TM_regions={len(signals['transmembrane_regions'])}, "
                      f"ER_retention={signals['er_retention']['detected']}, "
                      f"GPI_anchor={signals['gpi_anchor']['detected']}, "
                      f"NLS={signals['nuclear_localization']['detected']}",
        "confidence": heuristic_result["confidence"],
        "consensus": consensus,
        "consensus_confidence": consensus_confidence,
    })

    out_df = pd.DataFrame(output_rows)
    out_path = OUTPUT_DIR / "localization_ml_comparison.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(out_df)} rows)")

    # Save detailed signal analysis as JSON
    detail_path = OUTPUT_DIR / "localization_signals_detail.json"
    detail = {
        "protein_id": seq_id,
        "sequence_length": len(sequence),
        "deeploc": deeploc_result,
        "enhanced_heuristic": heuristic_result,
        "consensus": {
            "prediction": consensus,
            "confidence": consensus_confidence,
            "note": consensus_note,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # Remove non-serializable items
    if "signals" in detail["enhanced_heuristic"]:
        # Keep signals but ensure serializable
        pass

    with open(detail_path, "w") as fh:
        json.dump(detail, fh, indent=2, default=str)
    print(f"  Saved: {detail_path}")


if __name__ == "__main__":
    main()
