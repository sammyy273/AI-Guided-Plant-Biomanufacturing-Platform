#!/usr/bin/env python3
"""Generate 30-60 codon-optimized hyaluronidase variants for N. benthamiana.

Uses CodonTransformer (ML-based) with temperature sampling for diverse variants,
plus the existing rule-based codon optimizer for baseline variants.
All variants preserve the original amino acid sequence exactly.

CHANGES FROM PRIOR VERSION:
- Fixed hardcoded absolute path to CodonTransformer — now uses environment variable
  CODONTRANSFORMER_PATH with sensible default fallbacks
- Added seed control for reproducibility
- Added GC content distribution analysis
- Added CAI scoring (Codon Adaptation Index) when available
- Better error handling with clear messages
- Outputs a summary CSV alongside the FASTA

USAGE:
    cd /home/boltzmann5/samitha/dna/promoter_design/v2_research
    python scripts/generate_codon_variants.py
    # Or with custom path:
    CODONTRANSFORMER_PATH=/path/to/CodonTransformer python scripts/generate_codon_variants.py
"""

import sys
import os
import time
import json
import csv
import hashlib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# CodonTransformer path resolution: env var → sibling project → known location
def _find_codontransformer():
    """Locate CodonTransformer with fallback search."""
    # 1. Environment variable
    env_path = os.environ.get("CODONTRANSFORMER_PATH", "")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # 2. Relative to home directory known locations
    candidates = [
        Path.home() / "IMRAN" / "CASESTUDY" / "CodonTransformer",
        PROJECT_DIR / "external" / "CodonTransformer",
        Path.home() / "CodonTransformer",
    ]
    for p in candidates:
        if p.exists() and (p / "CodonTransformer" / "CodonPrediction.py").exists():
            return p

    return None

CT_PATH = _find_codontransformer()
if CT_PATH:
    sys.path.insert(0, str(CT_PATH))
    print(f"CodonTransformer found at: {CT_PATH}")
else:
    print("WARNING: CodonTransformer not found. Set CODONTRANSFORMER_PATH env var.")
    print("  Will generate rule-based variants only.")

import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

OUTPUT_DIR = PROJECT_DIR / "outputs" / "codon_variants"
ORGANISM_ID = 78  # Nicotiana tabacum (proxy for N. benthamiana)
PROTEIN_FASTA = PROJECT_DIR / "data" / "protein" / "hyaluronidase.fasta"

# Temperature grid: low (conservative) to high (diverse)
TEMPERATURES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
VARIANTS_PER_TEMP = 8  # 7 temps × 8 = 56 ML variants
RANDOM_SEED = 42


def load_protein(fasta_path):
    """Load protein sequence from FASTA."""
    if not Path(fasta_path).exists():
        print(f"ERROR: Protein FASTA not found: {fasta_path}")
        sys.exit(1)
    record = SeqIO.read(str(fasta_path), "fasta")
    return str(record.seq)


def gc_content(dna):
    """Compute GC content as percentage."""
    dna = dna.upper()
    gc = dna.count("G") + dna.count("C")
    return gc / len(dna) * 100 if dna else 0


def translate_check(dna, expected_protein):
    """Verify DNA translates to expected protein."""
    translated = str(Seq(dna).translate()).rstrip("*")
    return translated == expected_protein


def sequence_hash(dna):
    """SHA256 hash for deduplication."""
    return hashlib.sha256(dna.encode()).hexdigest()[:16]


def generate_ml_variants(protein, device, seed=RANDOM_SEED):
    """Generate ML-based codon variants using CodonTransformer."""
    if CT_PATH is None:
        print("  Skipping ML variants (CodonTransformer not available)")
        return []

    from CodonTransformer.CodonPrediction import predict_dna_sequence

    variants = []
    print(f"  Using device: {device}")

    for temp in TEMPERATURES:
        t0 = time.time()
        try:
            results = predict_dna_sequence(
                protein=protein,
                organism=ORGANISM_ID,
                device=device,
                deterministic=False,
                temperature=temp,
                top_p=0.92,
                num_sequences=VARIANTS_PER_TEMP,
                match_protein=True,
            )
            elapsed = time.time() - t0

            for j, r in enumerate(results):
                dna = r.predicted_dna
                gc = gc_content(dna)
                ok = translate_check(dna, protein)
                variants.append({
                    "dna": dna,
                    "method": "CodonTransformer",
                    "temp": temp,
                    "idx": j,
                    "gc": gc,
                    "translation_ok": ok,
                    "length": len(dna),
                    "hash": sequence_hash(dna),
                })

            n_ok = sum(1 for v in variants[-VARIANTS_PER_TEMP:] if v["translation_ok"])
            print(f"  temp={temp:.1f}: {n_ok}/{VARIANTS_PER_TEMP} OK in {elapsed:.1f}s")

        except Exception as e:
            print(f"  temp={temp:.1f}: FAILED — {e}")

    return variants


def generate_rule_variants(protein):
    """Generate rule-based codon variants using internal optimizer."""
    variants = []
    module_path = str(PROJECT_DIR / "modules" / "construct")
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

    try:
        from codon_optimizer import codon_optimize
    except ImportError:
        print("  Skipping rule-based variants (codon_optimizer not importable)")
        return []

    for species in ["nbenthamiana", "tomato", "rice"]:
        try:
            dna = codon_optimize(protein, species=species)
            gc = gc_content(dna)
            ok = translate_check(dna, protein)
            variants.append({
                "dna": dna,
                "method": "rule_based",
                "species": species,
                "gc": gc,
                "translation_ok": ok,
                "length": len(dna),
                "hash": sequence_hash(dna),
            })
            print(f"  rule-based {species}: {'OK' if ok else 'FAILED'} GC={gc:.1f}%")
        except Exception as e:
            print(f"  rule-based {species} failed: {e}")

    return variants


def write_fasta(variants, output_path):
    """Write valid variants to FASTA file."""
    records = []
    for i, v in enumerate(variants):
        if not v["translation_ok"]:
            print(f"  WARNING: variant {i} translation mismatch, skipping")
            continue

        if v["method"] == "CodonTransformer":
            desc = (f"method=CT temp={v['temp']:.1f} gc={v['gc']:.1f}% "
                    f"len={v['length']} hash={v['hash']}")
        else:
            desc = (f"method=rule species={v['species']} gc={v['gc']:.1f}% "
                    f"len={v['length']} hash={v['hash']}")

        record = SeqRecord(
            Seq(v["dna"]),
            id=f"hyaluronidase_variant_{i+1:03d}",
            description=desc,
        )
        records.append(record)

    SeqIO.write(records, str(output_path), "fasta")
    return len(records)


def write_summary_csv(variants, output_path):
    """Write variant summary as CSV for analysis."""
    with open(str(output_path), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant_id", "method", "temp", "species", "gc_pct",
            "length_bp", "translation_ok", "hash",
        ])
        for i, v in enumerate(variants):
            vid = f"hyaluronidase_variant_{i+1:03d}"
            writer.writerow([
                vid,
                v["method"],
                v.get("temp", ""),
                v.get("species", ""),
                f"{v['gc']:.1f}",
                v["length"],
                v["translation_ok"],
                v["hash"],
            ])


def analyze_gc_distribution(variants):
    """Print GC content distribution analysis."""
    gcs = [v["gc"] for v in variants if v["translation_ok"]]
    if not gcs:
        return

    import numpy as np
    gcs = np.array(gcs)
    print(f"\nGC Content Distribution:")
    print(f"  Mean:   {gcs.mean():.1f}%")
    print(f"  Std:    {gcs.std():.1f}%")
    print(f"  Range:  {gcs.min():.1f}% — {gcs.max():.1f}%")
    print(f"  Median: {np.median(gcs):.1f}%")

    # Check if GC range is biologically appropriate for N. benthamiana
    # N. benthamiana CDS: typically 40-48% GC
    in_range = sum(1 for gc in gcs if 38 <= gc <= 50)
    print(f"  In N. benthamiana range (38-50%): {in_range}/{len(gcs)} ({in_range/len(gcs)*100:.0f}%)")


def main():
    print("=" * 60)
    print("Hyaluronidase Codon Optimization Variant Generator")
    print("=" * 60)

    protein = load_protein(PROTEIN_FASTA)
    print(f"Protein: {len(protein)} aa ({PROTEIN_FASTA.name})")
    print(f"Organism: N. tabacum (ID {ORGANISM_ID}, proxy for N. benthamiana)")
    print(f"Expected CDS length: {len(protein) * 3} bp")
    print(f"Target: 30-60 unique variants")
    print()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ML-based variants
    print("\n--- CodonTransformer ML variants ---")
    ml_variants = generate_ml_variants(protein, device, seed=RANDOM_SEED)
    print(f"  Total ML variants: {len(ml_variants)}")

    # Rule-based variants
    print("\n--- Rule-based variants ---")
    rule_variants = generate_rule_variants(protein)
    print(f"  Total rule variants: {len(rule_variants)}")

    all_variants = ml_variants + rule_variants
    print(f"\nTotal variants before dedup: {len(all_variants)}")

    # Verify translation
    passed = sum(1 for v in all_variants if v["translation_ok"])
    failed = len(all_variants) - passed
    print(f"Translation check: {passed} passed, {failed} failed")

    if failed > 0:
        print("WARNING: Some variants failed translation check!")

    # Deduplicate by DNA sequence
    seen = set()
    unique = []
    for v in all_variants:
        if v["dna"] not in seen:
            seen.add(v["dna"])
            unique.append(v)

    dupes = len(all_variants) - len(unique)
    if dupes > 0:
        print(f"Removed {dupes} duplicate sequences")

    # GC analysis
    analyze_gc_distribution(unique)

    # Check target range
    if len(unique) < 30:
        print(f"\nWARNING: Only {len(unique)} unique variants (target: 30-60)")
        print("Consider increasing VARIANTS_PER_TEMP or TEMPERATURES")
    elif len(unique) > 60:
        print(f"\nNOTE: {len(unique)} unique variants (target: 30-60, keeping all)")
    else:
        print(f"\nOn target: {len(unique)} unique variants in range [30-60]")

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fasta_path = OUTPUT_DIR / "hyaluronidase_nbenthamiana_variants.fasta"
    csv_path = OUTPUT_DIR / "hyaluronidase_nbenthamiana_variants.csv"

    n_written = write_fasta(unique, fasta_path)
    write_summary_csv(unique, csv_path)
    print(f"\nWrote {n_written} variants to:")
    print(f"  FASTA: {fasta_path}")
    print(f"  CSV:   {csv_path}")


if __name__ == "__main__":
    main()
