# =============================================================================
# blast_novelty.py
#
# WHAT THIS DOES:
#   Runs each candidate promoter through NCBI BLAST (online, via Biopython)
#   to verify novelty. Records the percent identity of the top hit.
#   Candidates with >95% identity to any known sequence are flagged as
#   non-novel. Candidates with <85% identity are considered genuinely novel.
#
# WHY BLAST INSTEAD OF JUST LOCAL COMPARISON:
#   Our pipeline generates sequences using Evo2, which was trained on genomic
#   data. It may reproduce known promoter sequences from its training data.
#   BLAST against GenBank's nr database catches this — something no local
#   metric can do.
#
# PERFORMANCE:
#   Each BLAST query takes ~30-60 seconds via NCBI's online API.
#   For 60 candidates: ~30-60 min total. For overnight loops, we BLAST
#   only the top 10 candidates after filtering (not all 60).
#
# OUTPUT:
#   Adds 'blast_top_identity' and 'blast_novel' columns to scored CSV.
# =============================================================================

import io
import sys
import time
import logging
from Bio.Blast import NCBIWWW, NCBIXML

NOVELTY_THRESHOLD = 85.0   # < 85% identity = genuinely novel
KNOWN_THRESHOLD = 95.0     # > 95% identity = essentially a known sequence


def blast_single_sequence(seq: str, seq_id: str = "query",
                          logger: logging.Logger = None) -> dict:
    """
    BLAST a single DNA sequence against NCBI nt database.
    Returns dict with top hit info or 'NOVEL' if no significant hits.
    """
    result = {
        "blast_top_identity": None,
        "blast_top_hit": None,
        "blast_evalue": None,
        "blast_novel": None,
    }

    try:
        if logger:
            logger.info(f"       BLASTing {seq_id} ({len(seq)} bp)...")

        handle = NCBIWWW.qblast(
            program="blastn",
            database="nt",
            sequence=seq,
            hitlist_size=5,
            expect=0.001,
            format_type="XML",
        )

        blast_records = NCBIXML.parse(io.StringIO(handle.read()))
        for record in blast_records:
            if record.alignments:
                best = record.alignments[0]
                best_hsp = best.hsps[0]
                # Percent identity
                identity_pct = (best_hsp.identities / best_hsp.align_length) * 100
                result["blast_top_identity"] = round(identity_pct, 1)
                result["blast_top_hit"] = best.hit_def[:80]
                result["blast_evalue"] = best_hsp.expect
                result["blast_novel"] = identity_pct < NOVELTY_THRESHOLD
            else:
                # No significant hits — genuinely novel
                result["blast_top_identity"] = 0.0
                result["blast_top_hit"] = "No significant hit"
                result["blast_evalue"] = 1.0
                result["blast_novel"] = True
            break  # only first record

        handle.close()

    except Exception as e:
        if logger:
            logger.warning(f"       BLAST failed for {seq_id}: {e}")
        result["blast_top_identity"] = -1.0
        result["blast_top_hit"] = f"BLAST error: {str(e)[:60]}"
        result["blast_novel"] = None

    return result


def blast_candidates(sequences: dict, logger: logging.Logger = None,
                     max_sequences: int = 10) -> dict:
    """
    BLAST a dictionary of {id: sequence} pairs.
    Limits to max_sequences to respect time constraints.
    Returns dict of {id: blast_result_dict}.
    """
    results = {}
    to_blast = list(sequences.items())[:max_sequences]

    if len(sequences) > max_sequences and logger:
        logger.info(f"       BLASTing top {max_sequences} of {len(sequences)} candidates")

    for i, (seq_id, seq) in enumerate(to_blast):
        result = blast_single_sequence(seq, seq_id, logger)
        results[seq_id] = result
        # Rate limit: wait between queries
        if i < len(to_blast) - 1:
            time.sleep(3)

    return results


def print_blast_summary(results: dict, logger: logging.Logger):
    """Print a summary of BLAST results."""
    logger.info("")
    logger.info("       BLAST NOVELTY RESULTS:")
    logger.info("       " + "-" * 60)

    novel_count = 0
    known_count = 0
    error_count = 0

    for seq_id, r in results.items():
        identity = r["blast_top_identity"]
        if identity is None or identity < 0:
            status = "ERROR"
            error_count += 1
        elif identity < NOVELTY_THRESHOLD:
            status = f"NOVEL ({identity}% identity)"
            novel_count += 1
        elif identity < KNOWN_THRESHOLD:
            status = f"VARIANT ({identity}% identity)"
            novel_count += 1
        else:
            status = f"KNOWN ({identity}% identity)"
            known_count += 1

        hit_info = r.get("blast_top_hit", "N/A")
        logger.info(f"       {seq_id[:40]:<40} {status}")
        logger.info(f"         Top hit: {hit_info}")

    logger.info("       " + "-" * 60)
    logger.info(f"       Summary: {novel_count} novel/variant, "
                f"{known_count} known, {error_count} errors")

    if known_count > 0:
        logger.warning("       WARN: Some candidates are nearly identical to known sequences!")
        logger.warning("       These should be excluded from wet-lab testing.")

    return results
