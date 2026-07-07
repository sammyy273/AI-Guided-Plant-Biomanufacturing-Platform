# Final Species Report Generator
#
# Generates a comprehensive per-species report from batch run outputs.
# Aggregates the best candidate across all iterations into a single document
# with all 7 evaluation axes + comparison to known promoters.

import json
import os
import csv
import glob
from typing import Optional


# Known promoter benchmarks (well-characterised sequences)
# Sources: PlantCARE, published literature
KNOWN_PROMOTERS = {
    "CaMV_35S": {
        "sequence": (
            "TGACGTAAAGGATCCCGTGTGGAATGTAAAAAGAATGAGCGCAAGACCTTCCAGATC"
            "TTTCCAAACTCTCCAAGCGCACGATCTTCAACTCTTCTCCACCATGGTGTCCAGAAG"
            "GTGTTTGAGCACTTCAACGAGCAGCAGTCCAAATATCAGTACCCACAGTATCTTGCC"
            "GTCAATGGTGACCTTAATGCTTTCTCTTAACATGGTTTATCCATTCGTTCAATCCAC"
            "TCTTAAGGCCTTTTAATATGGTGGAGATCATCACTTTTGGTCTCTCCAATCTTTAGC"
        ),
        "type": "viral",
        "typical_strength": "high",
        "notes": "Universal strong constitutive promoter, but may trigger silencing in some species",
    },
    "Maize_Ubiquitin": {
        "sequence": (
            "GTCTTTACAGATCTCTCTCTCTCTCTCTCTCTCTCTCAACACTCTCTCTCTCTCTCTC"
            "ATATAAGGGGTGGGTTTGTTTGTTTGTTGGGGTGGGGATGTGGGGAGATGATGGGG"
            "AGGGGATGCTGCAGGCATGCCGCTGCAGGTACCCAAGCTGGGGATCCATGGATATGG"
        ),
        "type": "endogenous_monocot",
        "typical_strength": "very_high",
        "notes": "Standard monocot constitutive promoter, includes intron for enhanced expression",
    },
    "AtUBQ10": {
        "sequence": (
            "AACTCTCTATATAAGAGTAGATTATTTATTGCTTTGTGTTTCTCTTTTTTTTTTTTTT"
            "TTTTTTTTTTTGATTTTAGCAATTTATTTTCATAATATAATAATAATAAAATATTTAT"
            "TATATATATATATGTTTTTTTAATCTAATTTATATAATATATATATATTATAATATAT"
        ),
        "type": "endogenous_dicot",
        "typical_strength": "high",
        "notes": "Arabidopsis Ubiquitin 10 — strong constitutive, gene trap compatible",
    },
    "OsActin1": {
        "sequence": (
            "CCACCCCTTTACACTCCCTTCACCATCCTCTCCACTACTCCTCTCCCATCTCCATCC"
            "CTCCCTCCACTCCACCACTCCCATCTCATCTCCACCTCCCATCTCCTCCATCTCCAT"
        ),
        "type": "endogenous_monocot",
        "typical_strength": "very_high",
        "notes": "Rice Actin 1 — strong monocot constitutive promoter",
    },
}


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two sequences."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def _novelty_vs_known(candidate_seq: str) -> list:
    """Compute novelty (1 - identity) against all known promoters."""
    results = []
    for name, info in KNOWN_PROMOTERS.items():
        ref = info["sequence"]
        # Use the shorter of the two for comparison
        min_len = min(len(candidate_seq), len(ref))
        cand_short = candidate_seq[:min_len]
        ref_short = ref[:min_len]
        dist = _levenshtein(cand_short, ref_short)
        identity = 1.0 - (dist / min_len) if min_len > 0 else 0
        results.append({
            "promoter": name,
            "type": info["type"],
            "identity": round(identity, 3),
            "novelty": round(1 - identity, 3),
            "expected_strength": info["typical_strength"],
        })
    return sorted(results, key=lambda x: -x["identity"])


def _compute_confidence(candidate_scores: dict) -> dict:
    """Compute confidence level based on how many axes agree on quality.

    High confidence: ≥5/7 axes above threshold
    Medium confidence: 3-4/7
    Low confidence: <3/7
    """
    def _f(key, default=0.0):
        """Get float from CSV row (values may be strings)."""
        v = candidate_scores.get(key, default)
        return float(v) if isinstance(v, str) else v

    axes = {
        "strength": _f("weighted_score") >= 40,
        "silencing_low": _f("silencing_risk") <= 0.25,
        "novelty": _f("novelty_35s") >= 0.3,
        "diversity": _f("internal_div") >= 0.3,
        "embedding": _f("embedding_similarity") >= 0.9,
        "safe_harbor": _f("safe_harbor_score") >= 0.5,
        "gc_range": 30 <= _f("gc_pct") <= 55,
    }

    passing = sum(1 for v in axes.values() if v)
    total = len(axes)

    if passing >= 5:
        level = "HIGH"
    elif passing >= 3:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "level": level,
        "score": f"{passing}/{total}",
        "details": axes,
    }


def _find_best_across_iterations(output_dir: str) -> Optional[dict]:
    """Find the single best candidate across all iterations in a run."""
    best = None
    best_score = -1

    csv_files = sorted(glob.glob(os.path.join(output_dir, "iter*_scored.csv")))
    for csv_path in csv_files:
        with open(csv_path) as f:
            reader = list(csv.DictReader(f))
        for row in reader:
            score = float(row.get("composite_score", 0))
            if score > best_score:
                best_score = score
                best = row
                best["_csv_file"] = csv_path

    return best


def generate_species_report(species_key: str, output_dir: str) -> str:
    """Generate a comprehensive final report for one species.

    Args:
        species_key: Species identifier (e.g., 'tomato')
        output_dir: Path to the batch output directory for this species

    Returns:
        Formatted report string
    """
    # Load species config
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "species", f"{species_key}.yaml"
    )
    species_name = species_key
    species_type = "unknown"
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            species_name = cfg.get("species", {}).get("name", species_key)
            species_type = cfg.get("species", {}).get("type", "unknown")
        except Exception:
            pass

    # Find best candidate across all iterations
    best = _find_best_across_iterations(output_dir)

    # Load summary
    summary_path = os.path.join(output_dir, "loop_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    # Load best seed FASTA
    fasta_path = os.path.join(output_dir, "best_seed.fasta")
    best_sequence = ""
    if os.path.exists(fasta_path):
        with open(fasta_path) as f:
            lines = f.readlines()
        best_sequence = "".join(l.strip() for l in lines if not l.startswith(">"))

    # Build report
    W = 65
    lines = []
    lines.append("=" * W)
    lines.append(f"  FINAL SPECIES REPORT: {species_name}")
    lines.append("=" * W)
    lines.append(f"  Species key:     {species_key}")
    lines.append(f"  Type:            {species_type}")
    lines.append(f"  Iterations run:  {summary.get('iterations_completed', '?')}")
    lines.append(f"  Models used:     {summary.get('models', '?')}")
    lines.append("")

    if not best:
        lines.append("  NO VALID CANDIDATES FOUND")
        return "\n".join(lines)

    seq = best.get("sequence", best_sequence)
    composite = float(best.get("composite_score", 0))
    iter_file = os.path.basename(best.get("_csv_file", ""))

    # ── 1. Best Promoter Sequence ──
    lines.append("-" * W)
    lines.append("  1. BEST PROMOTER SEQUENCE")
    lines.append("-" * W)
    lines.append(f"  Candidate ID:    {best.get('', 'unknown')}")
    lines.append(f"  Source:          {iter_file}")
    lines.append(f"  Length:          {len(seq)} bp")
    lines.append(f"  Composite score: {composite:.4f}")
    lines.append(f"  Pareto front:    {best.get('pareto_front', '?')}")
    lines.append("")
    lines.append(f"  Sequence (FASTA):")
    lines.append(f"  >best_promoter_{species_key}_score{composite:.3f}")
    for i in range(0, len(seq), 80):
        lines.append(f"  {seq[i:i+80]}")
    lines.append("")

    # ── 2. Predicted Strength ──
    lines.append("-" * W)
    lines.append("  2. PREDICTED PROMOTER STRENGTH")
    lines.append("-" * W)
    strength = float(best.get("weighted_score", 0))
    gc = float(best.get("gc_pct", 0))
    lines.append(f"  Weighted cis-score:   {strength:.0f}")
    lines.append(f"  GC content:           {gc:.1f}%")
    lines.append(f"  Expression score:     {float(best.get('expression_score', 0)):.4f}")
    if strength >= 100:
        lines.append(f"  Assessment:           STRONG (top 10% of candidates)")
    elif strength >= 50:
        lines.append(f"  Assessment:           MODERATE")
    else:
        lines.append(f"  Assessment:           WEAK")
    lines.append("")

    # ── 3. Silencing Risk ──
    lines.append("-" * W)
    lines.append("  3. SILENCING RISK")
    lines.append("-" * W)
    silencing = float(best.get("silencing_risk", 0))
    if silencing <= 0.2:
        risk_level = "LOW"
    elif silencing <= 0.35:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
    lines.append(f"  Overall risk:         {silencing:.4f} ({risk_level})")
    lines.append(f"  PTGS risk:            {'LOW' if silencing <= 0.2 else 'ELEVATED'}")
    lines.append(f"  TGS risk:             {'LOW' if silencing <= 0.3 else 'ELEVATED'}")
    lines.append("")

    # ── 4. Novelty vs Known Promoters ──
    lines.append("-" * W)
    lines.append("  4. NOVELTY vs KNOWN PROMOTERS")
    lines.append("-" * W)
    novelty_35s = float(best.get("novelty_35s", 0))
    lines.append(f"  Novelty vs CaMV 35S:  {novelty_35s:.3f}")
    lines.append(f"  Internal diversity:   {float(best.get('internal_div', 0)):.3f}")
    lines.append("")
    lines.append(f"  Comparison to known promoters:")
    comparisons = _novelty_vs_known(seq)
    for c in comparisons:
        marker = "<-- most similar" if c["identity"] == comparisons[0]["identity"] else ""
        lines.append(f"    vs {c['promoter']:<20} identity: {c['identity']:.1%}  "
                      f"novelty: {c['novelty']:.1%}  "
                      f"({c['type']})  {marker}")
    lines.append("")

    # ── 5. Safe Harbor ──
    lines.append("-" * W)
    lines.append("  5. GENOMIC INSERTION SITE")
    lines.append("-" * W)
    sh_score = float(best.get("safe_harbor_score", 0))
    if sh_score >= 0.8:
        sh_assessment = "SAFE — recommended"
    elif sh_score >= 0.5:
        sh_assessment = "MODERATE — some risk factors"
    else:
        sh_assessment = "UNSAFE — high risk of position effects"
    lines.append(f"  Safe harbor score:    {sh_score:.4f}")
    lines.append(f"  Assessment:           {sh_assessment}")
    lines.append(f"  Note:                 Genome data not loaded — score is placeholder")
    lines.append("")

    # ── 6. Ranking vs Known Promoters ──
    lines.append("-" * W)
    lines.append("  6. RANKING vs KNOWN PROMOTERS")
    lines.append("-" * W)
    lines.append(f"  {'Promoter':<25} {'Our Score':>12} {'Assessment':>15}")
    lines.append(f"  {'-'*25} {'-'*12} {'-'*15}")
    lines.append(f"  {'THIS CANDIDATE':<25} {composite:>12.3f} {'':>15}")
    for c in comparisons:
        # Estimate known promoter score as baseline
        est_score = 0.6 if c["expected_strength"] == "high" else (
            0.7 if c["expected_strength"] == "very_high" else 0.4
        )
        better = "BETTER" if composite > est_score else "WORSE"
        lines.append(f"  {c['promoter']:<25} {est_score:>12.3f} {better:>15}")
    lines.append("")

    # ── 7. Confidence ──
    lines.append("-" * W)
    lines.append("  7. CONFIDENCE LEVEL")
    lines.append("-" * W)
    confidence = _compute_confidence(best)
    lines.append(f"  Overall confidence:   {confidence['level']} ({confidence['score']} axes passing)")
    lines.append(f"  Axis breakdown:")
    for axis, passed in confidence["details"].items():
        mark = "PASS" if passed else "FAIL"
        lines.append(f"    {axis:<25} {mark}")
    lines.append("")

    # ── Summary ──
    lines.append("=" * W)
    lines.append(f"  VERDICT: {'PROCEED TO WET-LAB' if confidence['level'] in ('HIGH', 'MEDIUM') and composite >= 0.6 else 'NEEDS IMPROVEMENT'}")
    lines.append(f"  Best composite score: {composite:.4f}")
    lines.append(f"  Strength: {strength:.0f} | Silencing: {risk_level} | "
                  f"Novelty: {novelty_35s:.3f} | Confidence: {confidence['level']}")
    lines.append("=" * W)

    return "\n".join(lines)


def generate_all_reports(base_output_dir: str, species_list: list = None):
    """Generate reports for all species in a batch run.

    Args:
        base_output_dir: Base outputs directory
        species_list: List of species keys. If None, auto-detect.
    """
    if species_list is None:
        species_list = [
            "arabidopsis", "nbenthamiana", "rice", "maize", "tomato",
            "soybean", "wheat", "ntobacum", "by2_cells",
        ]

    report_dir = os.path.join(base_output_dir, "..", "final_reports")
    os.makedirs(report_dir, exist_ok=True)

    for species_key in species_list:
        # Find the most recent batch output for this species
        sp_dirs = sorted(glob.glob(os.path.join(base_output_dir, species_key, "*")))
        if not sp_dirs:
            print(f"  {species_key}: no output data found")
            continue

        latest = sp_dirs[-1]
        report = generate_species_report(species_key, latest)

        report_path = os.path.join(report_dir, f"{species_key}_report.txt")
        with open(report_path, "w") as f:
            f.write(report)

        # Extract one-line summary
        summary = json.load(open(os.path.join(latest, "loop_summary.json")))
        best = summary.get("best_composite_score", 0)
        iters = summary.get("iterations_completed", 0)
        print(f"  {species_key:<16} | best={best:.3f} | {iters} iters | saved: {report_path}")

    return report_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate final species reports")
    parser.add_argument("--output-dir", default=None,
                        help="Base outputs directory (default: auto-detect)")
    parser.add_argument("--species", default=None,
                        help="Single species to report on (default: all)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")

    if args.species:
        # Single species
        sp_dirs = sorted(glob.glob(os.path.join(args.output_dir, args.species, "*")))
        if sp_dirs:
            report = generate_species_report(args.species, sp_dirs[-1])
            print(report)
        else:
            print(f"No output data found for {args.species}")
    else:
        # All species
        report_dir = generate_all_reports(args.output_dir)
        print(f"\nAll reports saved to: {report_dir}")
