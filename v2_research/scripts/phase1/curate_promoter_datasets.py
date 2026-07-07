#!/usr/bin/env python3
"""
STEP 1: Curate real TSS-anchored promoter datasets from authoritative genome sources.

For each species, extracts ~1 kb upstream of annotated transcription start sites (TSS)
from GFF3 gene models. Only uses real genome sequences — no fabrication.

Species:
  - Arabidopsis thaliana (TAIR10 — already local)
  - Oryza sativa (IRGSP-1.0 from Ensembl Plants)
  - Solanum lycopersicum (SL4.0 from Ensembl Plants)
  - Nicotiana benthamiana (v1.0.1 from Sol Genomics / NB1 annotation)

OUTPUTS:
  data/promoters/{species}_promoters_1kb.fasta
  data/promoters/{species}_metadata.csv
  data/promoters/data_curation_report.txt
"""

import csv
import gzip
import hashlib
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# ── Configuration ──────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parents[2]  # v2_research/
DATA_DIR = BASE_DIR / "data"
PROMOTER_DIR = DATA_DIR / "promoters"
GENOME_DIR = DATA_DIR / "species_genomes"

UPSTREAM_LENGTH = 1000
MIN_LENGTH = 800
MAX_LENGTH = 1200
MAX_AMBIGUOUS_FRACTION = 0.05

SPECIES_CONFIG = {
    "arabidopsis": {
        "common_name": "Arabidopsis thaliana",
        "genome_fasta": GENOME_DIR / "arabidopsis" / "TAIR10.fa.gz",
        "genome_gff": GENOME_DIR / "arabidopsis" / "TAIR10_GFF3_genes.gff.gz",
        "source": "TAIR10 (Ensembl Plants)",
        "fasta_url": None,  # already local
        "gff_url": None,
        "chromosome_prefix": "",  # chromosomes named 1,2,3,4,5 in FASTA
        "notes": "High-quality TSS annotations from Araport11 gene models over TAIR10 assembly.",
    },
    "rice": {
        "common_name": "Oryza sativa",
        "genome_fasta": GENOME_DIR / "rice" / "IRGSP-1.0.fa.gz",
        "genome_gff": GENOME_DIR / "rice" / "IRGSP-1.0_GFF3_genes.gff.gz",
        "source": "IRGSP-1.0 (Ensembl Plants)",
        "fasta_url": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/oryza_sativa/dna/Oryza_sativa.IRGSP-1.0.dna.toplevel.fa.gz",
        "gff_url": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/gff3/oryza_sativa/Oryza_sativa.IRGSP-1.0.57.gff3.gz",
        "chromosome_prefix": "",
        "notes": "High-quality TSS from IRGSP-1.0 reference annotation.",
    },
    "tomato": {
        "common_name": "Solanum lycopersicum",
        "genome_fasta": GENOME_DIR / "tomato" / "SL3.0.fa.gz",
        "genome_gff": GENOME_DIR / "tomato" / "SL3.0_GFF3_genes.gff.gz",
        "source": "SL3.0 (Ensembl Plants release-57)",
        "fasta_url": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/solanum_lycopersicum/dna/Solanum_lycopersicum.SL3.0.dna.toplevel.fa.gz",
        "gff_url": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/gff3/solanum_lycopersicum/Solanum_lycopersicum.SL3.0.57.gff3.gz",
        "chromosome_prefix": "",
        "notes": "High-quality TSS from SL3.0 reference annotation (ITAG4.0 gene models).",
    },
    "nbenthamiana": {
        "common_name": "Nicotiana benthamiana",
        "genome_fasta": GENOME_DIR / "nbenthamiana" / "Nbent_v1.0.1.fa.gz",
        "genome_gff": GENOME_DIR / "nbenthamiana" / "Nbent_v1.0.1_gene_models.gff.gz",
        "source": "NB1 (Sol Genomics Network / NB1 assembly)",
        "fasta_url": None,  # Not reliably available via public FTP
        "gff_url": None,
        "chromosome_prefix": "",
        "notes": "NOT AVAILABLE — Nicotiana benthamiana genome and annotations could not be downloaded "
                 "from Sol Genomics Network or NCBI at time of execution. "
                 "APPROXIMATE upstream extraction would be required; TSS uncertainty is high for this "
                 "non-model species. Skipping until data becomes available.",
    },
}


# ── Download helpers ───────────────────────────────────────────────────────

def download_if_needed(url: str, dest: Path, label: str):
    """Download a file if it doesn't already exist."""
    if dest.exists():
        size_mb = dest.stat().st_size / 1e6
        print(f"  [{label}] Already exists ({size_mb:.1f} MB): {dest}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [{label}] Downloading from {url} ...")
    print(f"            -> {dest}")
    try:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        size_mb = dest.stat().st_size / 1e6
        print(f"  [{label}] Downloaded ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [{label}] DOWNLOAD FAILED: {e}")
        # Clean up partial download
        if tmp.exists():
            tmp.unlink()
        return False


# ── GFF3 parsing ───────────────────────────────────────────────────────────

def parse_gff3_genes(gff_path: Path):
    """
    Parse GFF3 to extract gene-level TSS positions.

    For + strand genes: TSS = gene start
    For - strand genes: TSS = gene end

    Returns list of dicts: {gene_id, chromosome, strand, tss_position, gene_start, gene_end}
    """
    genes = []
    opener = gzip.open if str(gff_path).endswith(".gz") else open

    with opener(str(gff_path), "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue

            feature_type = parts[2]
            if feature_type != "gene":
                continue

            chrom = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            attributes = parts[8]

            # Extract gene_id from ID attribute
            gene_id = None
            for attr in attributes.split(";"):
                if attr.startswith("ID=gene:"):
                    gene_id = attr.split("gene:", 1)[1]
                    break
                elif attr.startswith("ID="):
                    gene_id = attr.split("ID=", 1)[1]
                    break

            if gene_id is None:
                continue

            # Filter out transposable elements and non-standard features
            if "transposable_element" in attributes or "TE" in attributes.split("logic_name=")[-1] if "logic_name=" in attributes else False:
                continue

            # Determine TSS
            if strand == "+":
                tss = start
            else:
                tss = end

            genes.append({
                "gene_id": gene_id,
                "chromosome": chrom,
                "strand": strand,
                "tss_position": tss,
                "gene_start": start,
                "gene_end": end,
            })

    return genes


# ── Sequence extraction ────────────────────────────────────────────────────

def load_genome(fasta_path: Path):
    """Load genome sequences indexed by chromosome name."""
    print(f"  Loading genome: {fasta_path}")
    genome = {}
    opener = gzip.open if str(fasta_path).endswith(".gz") else open

    with opener(str(fasta_path), "rt") as fh:
        for record in SeqIO.parse(fh, "fasta"):
            # Store with and without potential prefixes
            genome[record.id] = str(record.seq).upper()

    print(f"  Loaded {len(genome)} chromosome(s)")
    return genome


def extract_upstream(genome, gene_info, length=UPSTREAM_LENGTH):
    """
    Extract upstream sequence from TSS.

    For + strand: upstream is (TSS - length) to (TSS - 1)
    For - strand: upstream is TSS to (TSS + length - 1), then reverse-complemented
    """
    chrom = gene_info["chromosome"]
    strand = gene_info["strand"]
    tss = gene_info["tss_position"]

    if chrom not in genome:
        return None, None, None

    chrom_seq = genome[chrom]
    chrom_len = len(chrom_seq)

    if strand == "+":
        extract_start = max(0, tss - length)
        extract_end = tss
        seq = chrom_seq[extract_start:extract_end]
    else:
        extract_start = tss - 1  # 0-based
        extract_end = min(chrom_len, tss - 1 + length)
        seq = chrom_seq[extract_start:extract_end]
        seq = str(Seq(seq).reverse_complement())

    actual_length = len(seq)
    if actual_length < MIN_LENGTH:
        return None, None, None

    return seq, extract_start + 1, extract_end  # 1-based for metadata


def filter_sequence(seq):
    """Return True if sequence passes quality filters."""
    n_count = seq.upper().count("N")
    if n_count / len(seq) > MAX_AMBIGUOUS_FRACTION:
        return False
    if len(seq) < MIN_LENGTH or len(seq) > MAX_LENGTH:
        return False
    return True


# ── Main pipeline ──────────────────────────────────────────────────────────

def curate_species(species_key: str, config: dict, report_lines: list):
    """Curate promoters for one species."""
    print(f"\n{'='*60}")
    print(f"  Curating: {config['common_name']} ({species_key})")
    print(f"{'='*60}")

    fasta_path = config["genome_fasta"]
    gff_path = config["genome_gff"]

    # Download if needed
    if config["fasta_url"] and not fasta_path.exists():
        ok = download_if_needed(config["fasta_url"], fasta_path, f"{species_key} FASTA")
        if not ok:
            report_lines.append(
                f"\n{species_key.upper()}: FAILED — genome FASTA could not be downloaded.\n"
                f"  URL: {config['fasta_url']}\n"
                f"  STATUS: NOT AVAILABLE\n"
            )
            return False

    if config["gff_url"] and not gff_path.exists():
        ok = download_if_needed(config["gff_url"], gff_path, f"{species_key} GFF3")
        if not ok:
            report_lines.append(
                f"\n{species_key.upper()}: FAILED — GFF3 annotation could not be downloaded.\n"
                f"  URL: {config['gff_url']}\n"
                f"  STATUS: NOT AVAILABLE\n"
            )
            return False

    # Verify files exist
    if not fasta_path.exists():
        report_lines.append(
            f"\n{species_key.upper()}: SKIPPED — genome FASTA not found at {fasta_path}\n"
            f"  STATUS: NOT AVAILABLE\n"
        )
        return False

    if not gff_path.exists():
        report_lines.append(
            f"\n{species_key.upper()}: SKIPPED — GFF3 not found at {gff_path}\n"
            f"  STATUS: NOT AVAILABLE\n"
        )
        return False

    # Parse genes
    genes = parse_gff3_genes(gff_path)
    print(f"  Parsed {len(genes)} gene features")

    if not genes:
        report_lines.append(
            f"\n{species_key.upper()}: FAILED — no gene features found in GFF3.\n"
        )
        return False

    # Load genome
    genome = load_genome(fasta_path)

    # Extract upstream sequences
    records = []
    metadata_rows = []
    seen_seqs = set()
    total_genes = len(genes)
    extracted = 0
    filtered_ambiguous = 0
    filtered_length = 0
    filtered_duplicate = 0
    filtered_no_chrom = 0

    for i, gene in enumerate(genes):
        if (i + 1) % 10000 == 0:
            print(f"  Processing gene {i+1}/{total_genes}...")

        seq, ext_start, ext_end = extract_upstream(genome, gene)

        if seq is None:
            filtered_no_chrom += 1
            continue

        if len(seq) < MIN_LENGTH:
            filtered_length += 1
            continue

        # Filter ambiguous
        n_frac = seq.upper().count("N") / len(seq)
        if n_frac > MAX_AMBIGUOUS_FRACTION:
            filtered_ambiguous += 1
            continue

        # Filter duplicates
        seq_hash = hashlib.md5(seq.encode()).hexdigest()
        if seq_hash in seen_seqs:
            filtered_duplicate += 1
            continue
        seen_seqs.add(seq_hash)

        # Create record
        record_id = f"{species_key}_{gene['gene_id']}"
        record = SeqRecord(
            Seq(seq),
            id=record_id,
            description=f"upstream_{len(seq)}bp chrom={gene['chromosome']} strand={gene['strand']} tss={gene['tss_position']}"
        )
        records.append(record)
        metadata_rows.append({
            "gene_id": gene["gene_id"],
            "chromosome": gene["chromosome"],
            "strand": gene["strand"],
            "tss_position": gene["tss_position"],
            "extracted_start": ext_start,
            "extracted_end": ext_end,
            "sequence_length": len(seq),
        })
        extracted += 1

    # Save FASTA
    fasta_out = PROMOTER_DIR / f"{species_key}_promoters_1kb.fasta"
    with open(fasta_out, "w") as fh:
        SeqIO.write(records, fh, "fasta")
    print(f"  Saved {len(records)} promoters to {fasta_out.name}")

    # Save metadata CSV
    meta_out = PROMOTER_DIR / f"{species_key}_metadata.csv"
    with open(meta_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "gene_id", "chromosome", "strand", "tss_position",
            "extracted_start", "extracted_end", "sequence_length"
        ])
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"  Saved metadata to {meta_out.name}")

    # Report
    lengths = [r["sequence_length"] for r in metadata_rows]
    gc_contents = []
    for r in records:
        s = str(r.seq).upper()
        gc = (s.count("G") + s.count("C")) / len(s) * 100
        gc_contents.append(gc)

    avg_len = sum(lengths) / len(lengths) if lengths else 0
    avg_gc = sum(gc_contents) / len(gc_contents) if gc_contents else 0

    report_lines.append(
        f"\n{species_key.upper()} — {config['common_name']}\n"
        f"{'─'*50}\n"
        f"Source: {config['source']}\n"
        f"Total genes parsed: {total_genes}\n"
        f"Promoters extracted: {extracted}\n"
        f"Filtered (ambiguous >5% N): {filtered_ambiguous}\n"
        f"Filtered (too short <{MIN_LENGTH}bp): {filtered_length}\n"
        f"Filtered (no chromosome match): {filtered_no_chrom}\n"
        f"Filtered (duplicate): {filtered_duplicate}\n"
        f"Avg promoter length: {avg_len:.0f} bp\n"
        f"Avg GC content: {avg_gc:.1f}%\n"
        f"Notes: {config['notes']}\n"
    )

    return True


def main():
    PROMOTER_DIR.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "PROMOTER DATASET CURATION REPORT",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Pipeline: phase1/curate_promoter_datasets.py",
        f"Upstream extraction length: ~{UPSTREAM_LENGTH} bp",
        f"Length filter: {MIN_LENGTH}–{MAX_LENGTH} bp",
        f"Ambiguity filter: <{MAX_AMBIGUOUS_FRACTION*100:.0f}% N bases",
        "",
        "METHODOLOGY:",
        "  For each annotated gene in the GFF3 file:",
        "    - + strand: TSS = gene start position; upstream = [TSS-1000, TSS)",
        "    - - strand: TSS = gene end position; upstream = [TSS, TSS+1000), reverse-complemented",
        "  Sequences filtered for ambiguity, length, and duplicates.",
        "  All sequences are REAL genomic DNA — no synthetic data.",
        "",
        "=" * 60,
    ]

    success_count = 0
    for species_key, config in SPECIES_CONFIG.items():
        try:
            ok = curate_species(species_key, config, report_lines)
            if ok:
                success_count += 1
        except Exception as e:
            report_lines.append(
                f"\n{species_key.upper()}: ERROR — {e}\n"
            )
            print(f"  ERROR processing {species_key}: {e}")

    report_lines.append(
        f"\n{'='*60}\n"
        f"SUMMARY\n"
        f"{'='*60}\n"
        f"Species successfully curated: {success_count}/{len(SPECIES_CONFIG)}\n"
        f"Output directory: data/promoters/\n"
    )

    # Save report
    report_path = PROMOTER_DIR / "data_curation_report.txt"
    with open(report_path, "w") as fh:
        fh.write("\n".join(report_lines))
    print(f"\nCuration report saved to {report_path}")


if __name__ == "__main__":
    main()
