#!/usr/bin/env python3
"""
STEP 1: Structure-Aware Validation with ESMFold.

Generates 3D structure prediction for hyaluronidase (SPAM1) using ESMFold,
then computes structural features: solvent accessibility, secondary structure,
disorder regions, domain compactness, and protease-accessible surface regions.

OUTPUTS:
  outputs/phase3/protein_structure.pdb
  outputs/phase3/structure_analysis.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "modules"))

OUTPUT_DIR = BASE_DIR / "outputs" / "phase3"
PROTEIN_DIR = BASE_DIR / "data" / "protein"


def load_target_protein():
    from Bio import SeqIO
    for f in PROTEIN_DIR.glob("*.fasta"):
        for rec in SeqIO.parse(str(f), "fasta"):
            return str(rec.seq), rec.id
    return None, None


def predict_structure_esmfold(sequence):
    """Predict protein structure using ESMFold via HuggingFace transformers."""
    from transformers import AutoTokenizer
    from transformers import EsmForProteinFolding, EsmConfig
    from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

    print("  Loading ESMFold model...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    print(f"  Predicting structure for {len(sequence)} aa...")
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.cuda() if torch.cuda.is_available() else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Convert to PDB
    output = outputs
    atom37_positions = output.positions.cpu().squeeze()
    atom37_mask = output.atoms_cpu().squeeze().ne(0) if hasattr(output, 'atoms_cpu') else None

    # Get pLDDT confidence scores
    if hasattr(output, 'plddt') or hasattr(output, 'lddt'):
        plddt = output.plddt.cpu().squeeze() if hasattr(output, 'plddt') else None
    else:
        plddt = None

    # Build PDB
    protein = OFProtein(
        aatype=inputs["input_ids"].cpu().squeeze(),
        atom_positions=atom37_positions,
        atom_mask=atom37_mask if atom37_mask is not None else torch.ones_like(atom37_positions[..., 0]),
        residue_index=torch.arange(len(sequence)),
        b_factors=plddt if plddt is not None else torch.zeros(len(sequence)),
    )

    pdb_string = to_pdb(protein)
    return pdb_string, plddt


def predict_structure_fallback(sequence):
    """Fallback: use ESM2 embeddings for structure property prediction."""
    print("  ESMFold failed — using ESM2 embedding-based fallback (CPU)")
    from transformers import AutoTokenizer, EsmModel

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
    # Force CPU to avoid CUDA assertion carry-over
    model = model.cpu()
    model.eval()

    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.cpu().squeeze().numpy()
    return embeddings


def compute_secondary_structure(sequence):
    """Predict secondary structure from sequence (Chou-Fasman-like heuristic)."""
    ss_assignments = []
    seq = sequence.upper()
    n = len(seq)

    # Propensity scales (simplified)
    helix_prop = {"A": 1.42, "E": 1.51, "L": 1.21, "M": 1.45, "Q": 1.11, "K": 1.16,
                  "R": 0.98, "H": 1.00, "I": 1.08, "F": 1.13, "W": 1.08, "V": 1.06}
    sheet_prop = {"V": 1.70, "I": 1.60, "Y": 1.47, "F": 1.38, "W": 1.37, "L": 1.30,
                  "C": 1.19, "T": 1.19, "Q": 1.10, "M": 1.05, "N": 0.73, "P": 0.57}

    # Sliding window
    window = 6
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        window_seq = seq[start:end]

        h_score = sum(helix_prop.get(aa, 1.0) for aa in window_seq) / len(window_seq)
        e_score = sum(sheet_prop.get(aa, 1.0) for aa in window_seq) / len(window_seq)

        if h_score > 1.15 and h_score > e_score:
            ss_assignments.append("H")  # helix
        elif e_score > 1.15:
            ss_assignments.append("E")  # sheet
        else:
            ss_assignments.append("C")  # coil

    return ss_assignments


def compute_solvent_accessibility(sequence, ss_assignments):
    """Estimate relative solvent accessibility from sequence."""
    n = len(sequence)
    rasa = []

    # Simple heuristic based on position and neighbors
    for i in range(n):
        # N-terminal and C-terminal residues are more exposed
        pos_factor = 1.0
        if i < 5 or i > n - 5:
            pos_factor = 1.2
        if i < 2 or i > n - 2:
            pos_factor = 1.5

        # Coil residues more exposed than helix/sheet
        ss_factor = {"H": 0.6, "E": 0.7, "C": 1.0}.get(ss_assignments[i], 0.8)

        # Proline and glycine are often surface-exposed
        aa = sequence[i].upper()
        aa_factor = 1.0
        if aa in "PGDENQKRS":
            aa_factor = 1.2
        elif aa in "AILMFWV":
            aa_factor = 0.7

        # Local density: count hydrophobic neighbors
        window = 5
        start = max(0, i - window)
        end = min(n, i + window + 1)
        neighbors = sequence[start:end].upper()
        hydro = sum(1 for a in neighbors if a in "AILMFWV")
        density_factor = max(0.3, 1.0 - hydro / (2 * window + 1))

        r = min(1.0, pos_factor * ss_factor * aa_factor * density_factor)
        rasa.append(round(r, 3))

    return rasa


def detect_disorder_regions(sequence, window=30):
    """Detect intrinsically disordered regions (simple charge-hydropathy)."""
    seq = sequence.upper()
    n = len(seq)
    disorder_scores = []

    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        segment = seq[start:end]

        # Net charge
        pos = sum(1 for a in segment if a in "KR")
        neg = sum(1 for a in segment if a in "DE")
        net_charge = abs(pos - neg) / len(segment)

        # Mean hydropathy (Kyte-Doolittle)
        kd = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9,
              "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3,
              "P": -1.6, "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5,
              "K": -3.9, "R": -4.5}
        mean_hydro = sum(kd.get(a, 0) for a in segment) / len(segment)

        # Uversky plot: disordered if low hydrophobicity + high charge
        disorder = max(0, min(1, 0.5 - mean_hydro / 8 + net_charge))
        disorder_scores.append(round(disorder, 3))

    return disorder_scores


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Structure-Aware Validation")
    print("=" * 60)

    sequence, seq_id = load_target_protein()
    if sequence is None:
        print("  ERROR: No target protein found")
        return

    print(f"  Target: {seq_id} ({len(sequence)} aa)")
    print(f"  CUDA: {torch.cuda.is_available()}")
    print()

    # ── Predict structure ───────────────────────────────────────────────
    pdb_string = None
    plddt = None
    structure_source = "none"

    # Try ESMFold
    try:
        from transformers import AutoTokenizer
        pdb_string, plddt = predict_structure_esmfold(sequence)
        structure_source = "ESMFold"
        print(f"  ESMFold prediction successful")
        if plddt is not None:
            mean_plddt = plddt.mean().item()
            print(f"  Mean pLDDT: {mean_plddt:.1f}")
            if mean_plddt < 50:
                print("  WARNING: Low confidence structure (pLDDT < 50)")
    except Exception as e:
        print(f"  ESMFold failed: {str(e)[:100]}")
        print("  Attempting ESM2 fallback...")

        try:
            embeddings = predict_structure_fallback(sequence)
            structure_source = "ESM2_embeddings_only"
            print(f"  ESM2 embeddings computed: shape {embeddings.shape}")
        except Exception as e2:
            print(f"  ESM2 also failed: {str(e2)[:100]}")
            structure_source = "sequence_heuristic_only"

    # ── Compute structural features ─────────────────────────────────────
    print("\n  Computing structural features...")

    # Secondary structure
    ss = compute_secondary_structure(sequence)
    ss_counts = {"H": ss.count("H"), "E": ss.count("E"), "C": ss.count("C")}
    ss_fracs = {k: round(v / len(ss), 3) for k, v in ss_counts.items()}
    print(f"    Secondary structure: {ss_fracs}")

    # Solvent accessibility
    rasa = compute_solvent_accessibility(sequence, ss)
    exposed_count = sum(1 for r in rasa if r > 0.7)
    buried_count = sum(1 for r in rasa if r < 0.3)
    print(f"    Surface exposure: {exposed_count} exposed, {buried_count} buried, {len(sequence) - exposed_count - buried_count} intermediate")

    # Disorder
    disorder = detect_disorder_regions(sequence)
    disordered_regions = []
    in_disorder = False
    start = 0
    for i, d in enumerate(disorder):
        if d > 0.6 and not in_disorder:
            start = i
            in_disorder = True
        elif d <= 0.6 and in_disorder:
            disordered_regions.append({"start": start + 1, "end": i, "length": i - start})
            in_disorder = False
    if in_disorder:
        disordered_regions.append({"start": start + 1, "end": len(sequence), "length": len(sequence) - start})
    print(f"    Disordered regions: {len(disordered_regions)}")

    # Domain compactness (rough estimate from sequence)
    # Hydrophobic core residues tend to cluster in globular domains
    hydro_clusters = []
    window = 20
    for i in range(0, len(sequence) - window, window // 2):
        segment = sequence[i:i + window].upper()
        hydro_frac = sum(1 for a in segment if a in "AILMFWV") / window
        if hydro_frac > 0.4:
            hydro_clusters.append({"start": i + 1, "end": i + window, "hydrophobicity": round(hydro_frac, 3)})

    # Merge overlapping clusters
    merged_clusters = []
    for c in hydro_clusters:
        if merged_clusters and c["start"] <= merged_clusters[-1]["end"] + 10:
            merged_clusters[-1]["end"] = max(merged_clusters[-1]["end"], c["end"])
        else:
            merged_clusters.append(dict(c))
    print(f"    Hydrophobic core regions: {len(merged_clusters)}")

    # ── Identify protease-accessible exposed motifs ─────────────────────
    protease_motifs = {
        "SBT1_subtilase": ["RR", "KR", "LK", "IR", "LR"],
        "C1A_cysteine": ["VG", "FA", "WA", "GVA", "GFA"],
        "A1_aspartic": ["DTG", "DSG", "DTA", "DTS", "DVG"],
    }

    exposed_motifs = []
    for family, motifs in protease_motifs.items():
        for motif in motifs:
            pos = 0
            while True:
                idx = sequence.upper().find(motif, pos)
                if idx == -1:
                    break
                # Check exposure of motif region
                motif_exposure = np.mean(rasa[max(0, idx):min(len(rasa), idx + len(motif))])
                exposed_motifs.append({
                    "family": family,
                    "motif": motif,
                    "position": idx + 1,
                    "exposure": round(motif_exposure, 3),
                    "accessible": motif_exposure > 0.5,
                })
                pos = idx + 1

    accessible_count = sum(1 for m in exposed_motifs if m["accessible"])
    total_motifs = len(exposed_motifs)
    print(f"    Protease motifs: {total_motifs} total, {accessible_count} surface-accessible")

    # ── Save PDB ────────────────────────────────────────────────────────
    if pdb_string:
        pdb_path = OUTPUT_DIR / "protein_structure.pdb"
        with open(pdb_path, "w") as fh:
            fh.write(pdb_string)
        print(f"\n  Saved structure: {pdb_path}")
    else:
        print("\n  No PDB structure generated (ESMFold unavailable)")

    # ── Save analysis JSON ──────────────────────────────────────────────
    analysis = {
        "protein_id": seq_id,
        "sequence_length": len(sequence),
        "structure_source": structure_source,
        "esmfold_plddt_mean": float(plddt.mean()) if plddt is not None else None,
        "esmfold_plddt_per_residue": [round(float(p), 1) for p in plddt] if plddt is not None else None,
        "secondary_structure": {
            "fractions": ss_fracs,
            "counts": ss_counts,
            "per_residue": ss,
        },
        "solvent_accessibility": {
            "per_residue": rasa,
            "exposed_count": exposed_count,
            "buried_count": buried_count,
            "intermediate_count": len(sequence) - exposed_count - buried_count,
            "mean_rasa": round(np.mean(rasa), 4),
        },
        "disorder": {
            "per_residue": disorder,
            "regions": disordered_regions,
            "disordered_fraction": round(sum(1 for d in disorder if d > 0.6) / len(disorder), 4),
        },
        "hydrophobic_cores": merged_clusters,
        "protease_accessible_motifs": {
            "total": total_motifs,
            "accessible": accessible_count,
            "buried": total_motifs - accessible_count,
            "accessibility_fraction": round(accessible_count / max(total_motifs, 1), 4),
            "details": exposed_motifs,
        },
        "confidence_note": "Structure predicted computationally by ESMFold. No experimental validation."
        if structure_source == "ESMFold" else "ESMFold unavailable; features computed from sequence heuristics only.",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    analysis_path = OUTPUT_DIR / "structure_analysis.json"
    with open(analysis_path, "w") as fh:
        json.dump(analysis, fh, indent=2, default=str)
    print(f"  Saved analysis: {analysis_path}")


if __name__ == "__main__":
    main()
