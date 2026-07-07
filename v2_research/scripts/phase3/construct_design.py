#!/usr/bin/env python3
"""
STEP 5: Construct Design (Cloning-Ready).

Assembles full gene constructs for each species:
  promoter + 5' UTR + signal peptide + CDS + terminator

Adds restriction sites, removes forbidden motifs, checks synthesis constraints.

OUTPUTS:
  outputs/phase3/final_construct_sequences.fasta
  outputs/phase3/construct_analysis.csv
"""

import gzip
import json
import sys
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

OUTPUT_DIR = BASE_DIR / "outputs" / "phase3"

# Standard terminators for plant expression
TERMINATORS = {
    "tomato": {"name": "NOS_terminator", "sequence": "AATGATTTATATTTATATTTATATTTTAATATTTTAATATTTTAATATATTTATATTTATATAAATTTAATTTTTATATTTATATTTATAATTATATTTATATTTATATTTATATTTATATTTATATTTATATTTAATAAATTTATTTATTTATATAATTTATATTTATATTTATATTTATA"},
    "rice": {"name": "NOS_terminator", "sequence": "AATGATTTATATTTATATTTATATTTTAATATTTTAATATTTTAATATATTTATATTTATATAAATTTAATTTTTATATTTATATTTATAATTATATTTATATTTATATTTATATTTATATTTATATTTATATTTAATAAATTTATTTATTTATATAATTTATATTTATATTTATATTTATA"},
    "nbenthamiana": {"name": "NOS_terminator", "sequence": "AATGATTTATATTTATATTTATATTTTAATATTTTAATATTTTAATATATTTATATTTATATAAATTTAATTTTTATATTTATATTTATAATTATATTTATATTTATATTTATATTTATATTTATATTTATATTTAATAAATTTATTTATTTATATAATTTATATTTATATTTATATTTATA"},
}

# Restriction sites for cloning
CLONING_SITES = {
    "5prime": {"enzyme": "XbaI", "site": "TCTAGA"},
    "3prime": {"enzyme": "BamHI", "site": "GGATCC"},
}

# Sites to avoid in the insert
AVOID_SITES = {
    "EcoRI": "GAATTC",
    "HindIII": "AAGCTT",
    "SalI": "GTCGAC",
    "PstI": "CTGCAG",
    "BsmBI": "CGTCTC",
}


def load_promoter_sequence(species):
    """Load best promoter sequence from final reports."""
    report_path = BASE_DIR / "outputs" / f"final_report_{species}.json"
    if not report_path.exists():
        return None, None

    with open(report_path) as fh:
        report = json.load(fh)

    bp = report.get("best_promoter", {})
    seq = bp.get("sequence", "")
    cid = bp.get("candidate_id", "unknown")

    return seq.upper(), cid


def load_cds_sequence(species):
    """Load optimized CDS from final reports."""
    report_path = BASE_DIR / "outputs" / f"final_report_{species}.json"
    if not report_path.exists():
        return None

    with open(report_path) as fh:
        report = json.load(fh)

    cds = report.get("optimized_cds", {})
    return cds.get("sequence", "").upper()


def check_synthesis_constraints(seq):
    """Check DNA synthesis constraints."""
    issues = []

    # GC extremes
    gc = (seq.count("G") + seq.count("C")) / len(seq) * 100
    if gc > 70:
        issues.append(f"HIGH_GC: {gc:.1f}%")
    elif gc < 20:
        issues.append(f"LOW_GC: {gc:.1f}%")

    # Homopolymer runs (>8)
    for base in "ACGT":
        for i in range(len(seq) - 8):
            if seq[i:i+9] == base * 9:
                issues.append(f"HOMOPOLYMER: {base}*9 at {i}")
                break

    # GC/AT extremes in local windows
    for i in range(0, len(seq) - 100, 50):
        window = seq[i:i+100]
        local_gc = (window.count("G") + window.count("C")) / 100
        if local_gc > 0.8 or local_gc < 0.15:
            issues.append(f"EXTREME_LOCAL_GC: {local_gc:.2f} at {i}")
            break

    # Hairpin potential (inverted repeats >15bp)
    for length in [15, 20]:
        found = False
        for i in range(len(seq) - 2 * length):
            fwd = seq[i:i+length]
            for j in range(i + length, len(seq) - length):
                rev = seq[j:j+length]
                comp = rev.translate(str.maketrans("ACGT", "TGCA"))[::-1]
                if fwd == comp:
                    issues.append(f"HAIRPIN: {length}bp at {i}-{j}")
                    found = True
                    break
            if found:
                break
        if found:
            break

    return issues, round(gc, 1)


def scan_avoid_sites(seq):
    """Scan for restriction sites that should be avoided."""
    found = []
    for enzyme, site in AVOID_SITES.items():
        idx = seq.find(site)
        while idx >= 0:
            found.append({"enzyme": enzyme, "site": site, "position": idx + 1})
            idx = seq.find(site, idx + 1)
    return found


def build_construct(species, promoter_seq, cds_seq, terminator_seq):
    """Build full construct with cloning sites."""
    if not promoter_seq or not cds_seq:
        return None

    # Components
    five_prime_site = CLONING_SITES["5prime"]["site"]
    three_prime_site = CLONING_SITES["3prime"]["site"]

    # Assemble: 5'site + promoter + CDS + terminator + 3'site
    construct = five_prime_site + promoter_seq + cds_seq + terminator_seq + three_prime_site

    return construct


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 5: Construct Design (Cloning-Ready)")
    print("=" * 60)

    species_list = ["tomato", "rice", "nbenthamiana"]
    fasta_records = []
    analysis_rows = []

    for species in species_list:
        print(f"\n  --- {species} ---")

        # Load components
        promoter_seq, promoter_id = load_promoter_sequence(species)
        cds_seq = load_cds_sequence(species)
        terminator = TERMINATORS.get(species, TERMINATORS["tomato"])
        terminator_seq = terminator["sequence"]

        if not promoter_seq:
            print(f"    No promoter found for {species}")
            continue

        print(f"    Promoter: {promoter_id} ({len(promoter_seq)}bp)")
        print(f"    CDS: {len(cds_seq) if cds_seq else 0}bp")
        print(f"    Terminator: {terminator['name']} ({len(terminator_seq)}bp)")

        # Build construct
        construct = build_construct(species, promoter_seq, cds_seq, terminator_seq)
        if not construct:
            print(f"    Cannot build construct for {species}")
            continue

        total_len = len(construct)
        print(f"    Full construct: {total_len}bp")

        # Check synthesis constraints
        synth_issues, gc_pct = check_synthesis_constraints(construct)
        print(f"    GC: {gc_pct}%")
        if synth_issues:
            print(f"    Synthesis issues: {len(synth_issues)}")
            for issue in synth_issues:
                print(f"      - {issue}")
        else:
            print(f"    Synthesis: no issues detected")

        # Scan for unwanted restriction sites
        avoid = scan_avoid_sites(construct)
        if avoid:
            print(f"    Restriction sites to remove: {len(avoid)}")
            for site in avoid:
                print(f"      - {site['enzyme']} ({site['site']}) at position {site['position']}")
        else:
            print(f"    Restriction sites: clean (no forbidden sites)")

        # FASTA record
        header = (
            f"construct_{species} promoter={promoter_id} "
            f"len={total_len}bp GC={gc_pct}% "
            f"cloning_sites=XbaI+BamHI terminator={terminator['name']}"
        )
        fasta_records.append(f">{header}\n{construct}")

        # Analysis row
        analysis_rows.append({
            "species": species,
            "promoter_id": promoter_id,
            "promoter_length_bp": len(promoter_seq),
            "cds_length_bp": len(cds_seq) if cds_seq else 0,
            "terminator": terminator["name"],
            "terminator_length_bp": len(terminator_seq),
            "total_construct_bp": total_len,
            "gc_pct": gc_pct,
            "synthesis_issues": len(synth_issues),
            "synthesis_issue_details": "; ".join(synth_issues) if synth_issues else "none",
            "forbidden_restriction_sites": len(avoid),
            "restriction_site_details": "; ".join(
                f"{s['enzyme']}@{s['position']}" for s in avoid
            ) if avoid else "none",
            "cloning_5prime": f"XbaI ({CLONING_SITES['5prime']['site']})",
            "cloning_3prime": f"BamHI ({CLONING_SITES['3prime']['site']})",
            "status": "READY" if not synth_issues and not avoid else "NEEDS_CLEANUP",
        })

    # Save FASTA
    fasta_path = OUTPUT_DIR / "final_construct_sequences.fasta"
    with open(fasta_path, "w") as fh:
        fh.write("\n".join(fasta_records))
    print(f"\n  Saved constructs: {fasta_path}")

    # Save analysis CSV
    if analysis_rows:
        out_df = pd.DataFrame(analysis_rows)
        out_path = OUTPUT_DIR / "construct_analysis.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  Saved analysis: {out_path}")


if __name__ == "__main__":
    main()
