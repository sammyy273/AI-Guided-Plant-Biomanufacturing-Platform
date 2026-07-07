#!/usr/bin/env python3
"""
STEP 4: Genome Context — Safe Harbor Candidate Identification.

Uses real genome FASTA + GFF3 data (downloaded in Phase 1) to identify
intergenic safe harbor regions for transgene insertion.

For species with genome data:
- Load FASTA + GFF3
- Compute distance to nearest gene
- Compute distance to nearest TE (if annotated)
- Classify GC content of region
- Identify heuristic safe zones

OUTPUTS:
  outputs/phase2/safe_harbor_candidates.csv
"""

import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "modules"))

OUTPUT_DIR = BASE_DIR / "outputs" / "phase2"
GENOME_DIR = BASE_DIR / "data" / "species_genomes"

# ── Species genome data availability ───────────────────────────────────────

SPECIES_GENOMES = {
    "arabidopsis": {
        "fasta": GENOME_DIR / "arabidopsis" / "TAIR10.fa.gz",
        "gff": GENOME_DIR / "arabidopsis" / "TAIR10_GFF3_genes.gff.gz",
        "common_name": "Arabidopsis thaliana",
        "chromosomes": ["1", "2", "3", "4", "5"],
    },
    "rice": {
        "fasta": GENOME_DIR / "rice" / "IRGSP-1.0.fa.gz",
        "gff": GENOME_DIR / "rice" / "IRGSP-1.0_GFF3_genes.gff.gz",
        "common_name": "Oryza sativa",
        "chromosomes": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
    },
    "tomato": {
        "fasta": GENOME_DIR / "tomato" / "SL3.0.fa.gz",
        "gff": GENOME_DIR / "tomato" / "SL3.0_GFF3_genes.gff.gz",
        "common_name": "Solanum lycopersicum",
        "chromosomes": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
    },
}


def parse_gene_positions(gff_path):
    """Parse GFF3 to extract gene positions and transposable elements."""
    genes = []
    tes = []

    opener = gzip.open if str(gff_path).endswith(".gz") else open
    with opener(str(gff_path), "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue

            feature_type = parts[2]
            chrom = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            attrs = parts[8]

            if feature_type == "gene":
                gene_id = None
                for attr in attrs.split(";"):
                    if attr.startswith("ID=gene:"):
                        gene_id = attr.split("gene:", 1)[1]
                        break
                    elif attr.startswith("ID="):
                        gene_id = attr.split("ID=", 1)[1]
                        break

                if gene_id:
                    biotype = "protein_coding"
                    if "transposable_element" in attrs:
                        biotype = "transposable_element"
                    elif "TE" in attrs and "pseudogene" in attrs:
                        biotype = "transposable_element"

                    if biotype == "transposable_element":
                        tes.append({"chrom": chrom, "start": start, "end": end, "id": gene_id})
                    else:
                        genes.append({"chrom": chrom, "start": start, "end": end, "id": gene_id, "strand": strand})

    return genes, tes


def compute_intergenic_regions(genes, chrom_length):
    """Find intergenic gaps on a chromosome."""
    sorted_genes = sorted(genes, key=lambda g: g["start"])

    regions = []
    prev_end = 0

    for gene in sorted_genes:
        if gene["start"] > prev_end + 1:
            regions.append({
                "start": prev_end + 1,
                "end": gene["start"] - 1,
                "length": gene["start"] - prev_end - 1,
            })
        prev_end = max(prev_end, gene["end"])

    # After last gene
    if prev_end < chrom_length:
        regions.append({
            "start": prev_end + 1,
            "end": chrom_length,
            "length": chrom_length - prev_end,
        })

    return regions


def compute_gc_content_window(genome_seq, start, end):
    """Compute GC content of a genomic region."""
    region = genome_seq[start-1:end].upper()
    if len(region) == 0:
        return 0
    gc = region.count("G") + region.count("C")
    return gc / len(region)


def find_safe_harbor_candidates(species_key, config, max_candidates=50):
    """Identify safe harbor candidates for one species."""
    print(f"\n  Processing: {config['common_name']} ({species_key})")

    fasta_path = config["fasta"]
    gff_path = config["gff"]

    if not fasta_path.exists() or not gff_path.exists():
        print(f"    Genome data NOT AVAILABLE")
        return []

    # Parse genes and TEs
    genes, tes = parse_gene_positions(gff_path)
    print(f"    Genes: {len(genes)}, TEs: {len(tes)}")

    # Group by chromosome
    genes_by_chrom = defaultdict(list)
    for g in genes:
        genes_by_chrom[g["chrom"]].append(g)

    tes_by_chrom = defaultdict(list)
    for t in tes:
        tes_by_chrom[t["chrom"]].append(t)

    # Load genome
    print("    Loading genome...")
    genome = {}
    opener = gzip.open if str(fasta_path).endswith(".gz") else open
    with opener(str(fasta_path), "rt") as fh:
        from Bio import SeqIO
        for record in SeqIO.parse(fh, "fasta"):
            genome[record.id] = str(record.seq).upper()

    chrom_lengths = {k: len(v) for k, v in genome.items()}
    print(f"    Loaded {len(genome)} sequences")

    # Find safe harbor candidates
    candidates = []

    for chrom in config["chromosomes"]:
        if chrom not in chrom_lengths:
            # Try with prefix matching
            matching = [c for c in chrom_lengths if c.endswith(chrom) or chrom in c]
            if matching:
                chrom_actual = matching[0]
            else:
                continue
        else:
            chrom_actual = chrom

        chrom_genes = genes_by_chrom.get(chrom_actual, genes_by_chrom.get(chrom, []))
        chrom_tes = tes_by_chrom.get(chrom_actual, tes_by_chrom.get(chrom, []))
        chrom_len = chrom_lengths[chrom_actual]

        # Find intergenic regions
        intergenic = compute_intergenic_regions(chrom_genes, chrom_len)

        for region in intergenic:
            # Filter: must be >2kb from nearest gene
            if region["length"] < 2000:
                continue

            mid = (region["start"] + region["end"]) // 2

            # Distance to nearest gene
            min_dist_gene = float("inf")
            for g in chrom_genes:
                dist = min(abs(mid - g["start"]), abs(mid - g["end"]))
                min_dist_gene = min(min_dist_gene, dist)

            # Distance to nearest TE
            min_dist_te = float("inf")
            te_count_nearby = 0
            for t in chrom_tes:
                dist = min(abs(mid - t["start"]), abs(mid - t["end"]))
                min_dist_te = min(min_dist_te, dist)
                if dist < 10000:  # within 10kb
                    te_count_nearby += 1

            # GC content of region
            gc = compute_gc_content_window(genome[chrom_actual], region["start"], region["end"])

            # Classify GC
            if gc < 0.30:
                gc_class = "AT_rich"
            elif gc < 0.40:
                gc_class = "moderate"
            elif gc < 0.50:
                gc_class = "GC_rich"
            else:
                gc_class = "very_GC_rich"

            # TE density
            region_len = region["end"] - region["start"]
            te_density = te_count_nearby / (region_len / 10000) if region_len > 0 else 0

            # Score the region
            score = 0
            score += min(20, min_dist_gene / 1000) * 2       # Distance from genes
            score += min(20, min_dist_te / 1000) * 1.5       # Distance from TEs
            score += 10 if 0.30 <= gc <= 0.45 else 0         # Optimal GC
            score -= min(15, te_density * 5)                   # Penalize TE density
            score += min(10, region["length"] / 5000)         # Prefer longer regions

            candidates.append({
                "species": species_key,
                "chromosome": chrom_actual,
                "region_start": region["start"],
                "region_end": region["end"],
                "region_length": region["length"],
                "midpoint": mid,
                "distance_nearest_gene": round(min_dist_gene, 0),
                "distance_nearest_te": round(min_dist_te, 0) if min_dist_te != float("inf") else "NOT ANNOTATED",
                "te_count_nearby_10kb": te_count_nearby,
                "te_density": round(te_density, 4),
                "gc_content": round(gc, 4),
                "gc_class": gc_class,
                "safe_harbor_score": round(score, 2),
            })

    # Sort by score and return top candidates
    candidates.sort(key=lambda x: -x["safe_harbor_score"])
    return candidates[:max_candidates]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 4: Genome Context — Safe Harbor Candidates")
    print("=" * 60)

    all_candidates = []
    species_status = {}

    for species_key, config in SPECIES_GENOMES.items():
        candidates = find_safe_harbor_candidates(species_key, config)
        if candidates:
            all_candidates.extend(candidates)
            species_status[species_key] = f"{len(candidates)} candidates found"
            print(f"    → {len(candidates)} safe harbor candidates")
        else:
            species_status[species_key] = "NOT AVAILABLE"

    if not all_candidates:
        print("\n  No safe harbor candidates found for any species.")
        # Still create output with NOT AVAILABLE labels
        for species_key in SPECIES_GENOMES:
            all_candidates.append({
                "species": species_key,
                "chromosome": "NOT AVAILABLE",
                "region_start": "NOT AVAILABLE",
                "region_end": "NOT AVAILABLE",
                "region_length": "NOT AVAILABLE",
                "midpoint": "NOT AVAILABLE",
                "distance_nearest_gene": "NOT AVAILABLE",
                "distance_nearest_te": "NOT AVAILABLE",
                "te_count_nearby_10kb": "NOT AVAILABLE",
                "te_density": "NOT AVAILABLE",
                "gc_content": "NOT AVAILABLE",
                "gc_class": "NOT AVAILABLE",
                "safe_harbor_score": "NOT AVAILABLE",
            })

    out_df = pd.DataFrame(all_candidates)
    out_path = OUTPUT_DIR / "safe_harbor_candidates.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(out_df)} rows)")

    # Print top candidates per species
    print("\n  Top safe harbor candidates per species:")
    for species_key in SPECIES_GENOMES:
        sp_df = out_df[out_df["species"] == species_key]
        if sp_df.empty or str(sp_df.iloc[0]["chromosome"]) == "NOT AVAILABLE":
            print(f"    {species_key}: NOT AVAILABLE")
            continue
        top = sp_df.head(3)
        for _, row in top.iterrows():
            print(f"    {species_key} chr{row['chromosome']}:{row['midpoint']:,} "
                  f"(score={row['safe_harbor_score']:.1f}, "
                  f"nearest_gene={row['distance_nearest_gene']:.0f}bp, "
                  f"GC={row['gc_content']:.2f}, "
                  f"length={row['region_length']:,}bp)")


if __name__ == "__main__":
    main()
