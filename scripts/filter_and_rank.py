# =============================================================================
# 4_filter_and_rank.py
#
# WHAT THIS DOES:
#   Takes the 60 Evo2 candidates and runs four independent analyses to
#   produce a Pareto-ranked shortlist of the best synthetic promoters.
#
# FOUR ANALYSES:
#
#   1. CIS-ELEMENT SCORING (local, instant)
#      Same element scan as Step 2 with added GCN4 and weighted scoring.
#      Hard filters: TATA box present, CAAT box present, GC content 40-60%.
#      Element weights: as-1=5x, GCN4=5x, ocs-like=3x, others=1-2x.
#      (TATA/CAAT alone produce minimal transcription; as-1/GCN4 drive strength.)
#
#   2. DNABERT-2 COSINE SIMILARITY (downloads ~500 MB on first run)
#      DNABERT-2 encodes any DNA sequence as a vector (embedding).
#      We measure how similar each candidate's vector is to the 2×35S vector.
#      NOTE: This is a syntactic similarity measure, not a validated functional
#      predictor. Treat DNABERT-2 similarity as one signal among several.
#
#   3. NOVELTY vs 35S
#      Levenshtein edit distance in the last 300 bp (closest to TSS).
#      Higher = more different from 35S, potentially more novel.
#
#   4. INTERNAL DIVERSITY
#      Mean pairwise Levenshtein distance to all other candidates.
#      Higher = more unique relative to the cohort.
#
#   PARETO RANKING:
#      Three-objective Pareto front (strength, novelty, diversity).
#      Candidates on front 1 are non-dominated — best by at least one objective.
#
#   COMPOSITE SCORE:
#      50% weighted strength + 25% novelty + 25% diversity.
#      Within the same Pareto front, sort by composite score.
#
# OUTPUT:
#   outputs/all_candidates_scored.csv
#   outputs/top3_candidates.fasta       ← take these to PlantCARE
#   outputs/ranking_table.csv
# =============================================================================

import os
import re
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import (setup_logger, log_step, log_ok, log_warn,
                   log_err, log_info, save_fasta, load_fasta)
from score_references import CIS_ELEMENTS, score_sequence
from jaspar_scanner import fetch_plant_profiles, scan_sequence_jaspar
from blast_novelty import blast_candidates, print_blast_summary


# =============================================================================
# DNABERT-2
# =============================================================================
def load_dnabert2(logger):
    """
    Load DNABERT-2 from HuggingFace.
    Downloads ~500 MB on first run, cached locally after that.
    """
    import os as _os
    # Disable flash-attention triton kernel — causes "Tensor on device meta"
    # runtime error on CPU-only machines. Must be set BEFORE importing transformers.
    _os.environ["USE_TORCH"] = "1"
    _os.environ["FLASH_ATTENTION"] = "0"
    # Also disable Triton entirely to avoid import errors
    _os.environ["TRITON_DISABLE"] = "1"

    try:
        from transformers import AutoTokenizer, AutoModel, AutoConfig
    except ImportError:
        log_err(logger, "transformers not installed. Run: pip install transformers torch")
        return None, None

    model_id = "zhihan1996/DNABERT-2-117M"
    log_info(logger, f"Loading DNABERT-2 ({model_id})...")
    log_info(logger, "First run: downloads ~500 MB. This is normal. Please wait.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if getattr(tokenizer, 'pad_token_id', None) is None:
            tokenizer.pad_token_id = getattr(tokenizer, 'eos_token_id', None) or 0
            tokenizer.pad_token = getattr(tokenizer, 'eos_token', None) or '[PAD]'

        raw_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # DNABERT-2's config lacks pad_token_id — inject it before model init.
        if not hasattr(raw_config, 'pad_token_id'):
            raw_config.pad_token_id = tokenizer.pad_token_id

        # Force SDPA attention implementation to avoid flash-attn/meta tensor issues
        # on CPU-only systems with transformers >= 5.x
        if hasattr(raw_config, 'attn_implementation'):
            raw_config.attn_implementation = "sdpa"

        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=raw_config,
            attn_implementation="sdpa",  # Explicit attention impl
        )
        model.eval()

        # Verify model actually works by doing a test embedding
        test_input = tokenizer("ATCGATCG", return_tensors="pt", padding=True,
                               truncation=True, max_length=16)
        with torch.no_grad():
            test_output = model(**test_input)
        if test_output.last_hidden_state is not None:
            log_ok(logger, "DNABERT-2 loaded and verified (test embedding succeeded).")
            return tokenizer, model
        else:
            log_warn(logger, "DNABERT-2 loaded but test embedding returned None.")
            return None, None

    except ImportError as e:
        log_warn(logger, f"DNABERT-2 requires missing dependency: {e}")
        log_warn(logger, "Try: pip install triton einops")
        log_warn(logger, "Continuing with k-mer frequency similarity as fallback.")
        return None, None
    except Exception as e:
        log_warn(logger, f"DNABERT-2 failed to load: {type(e).__name__}: {e}")
        log_warn(logger, "Continuing with k-mer frequency similarity as fallback.")
        return None, None


def embed(sequence: str, tokenizer, model, max_length: int = 512) -> torch.Tensor:
    """
    Encode a DNA sequence as a mean-pooled embedding vector.
    Truncates from the 3' end if over 512 tokens (~3 kb).
    For our 800 bp candidates this is never an issue.
    """
    inputs = tokenizer(
        sequence, return_tensors="pt",
        padding=True, truncation=True, max_length=max_length
    )
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state.mean(dim=1)   # shape: (1, hidden_dim)


def cosine_sim(e1, e2) -> float:
    """Cosine similarity between two vectors (torch tensors or numpy arrays)."""
    if hasattr(e1, 'cpu'):
        return float(F.cosine_similarity(e1.cpu().flatten().unsqueeze(0),
                                           e2.cpu().flatten().unsqueeze(0)).item())
    # numpy fallback
    e1, e2 = np.array(e1).flatten(), np.array(e2).flatten()
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10))


# =============================================================================
# K-MER FREQUENCY SIMILARITY (fallback when DNABERT-2 unavailable)
# =============================================================================
def kmer_freq_vector(sequence: str, k: int = 4) -> np.ndarray:
    """
    Compute k-mer frequency vector for a DNA sequence.
    Uses 4-mer frequencies by default (256-dim vector).
    Normalised so sum = 1.0 (frequency, not count).
    """
    seq = sequence.upper()
    n = len(seq)
    vec = np.zeros(4 ** k, dtype=np.float32)
    for i in range(n - k + 1):
        km = seq[i:i+k]
        if all(c in 'ACGT' for c in km):
            idx = sum({'A': 0, 'C': 1, 'G': 2, 'T': 3}[c] * (4 ** (k - 1 - j))
                      for j, c in enumerate(km))
            vec[idx] += 1
    total = vec.sum()
    return vec / total if total > 0 else vec


def kmer_similarity(seq1: str, seq2: str, k: int = 4) -> float:
    """
    Compute cosine similarity between two DNA sequences based on k-mer frequencies.
    k=4 (4-mers) is standard for regulatory element comparison.
    Returns float in [-1, 1]; 1.0 = identical k-mer profile.
    """
    v1 = kmer_freq_vector(seq1, k)
    v2 = kmer_freq_vector(seq2, k)
    return cosine_sim(v1, v2)


# =============================================================================
# LEVENSHTEIN DISTANCE & DIVERSITY
# =============================================================================
def levenshtein(s1: str, s2: str) -> int:
    """Pure-Python Levenshtein edit distance. O(nm) time, O(n) space."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1,           # deletion
                           prev[j] + (c1 != c2),        # substitution
                           curr[j] + 1))               # insertion
        prev = curr
    return prev[-1]


def _seq_to_region(seq: str, length: int = 300) -> str:
    """Use the last `length` bp of the sequence (closest to TSS) as the
    most functionally relevant region for distance comparison."""
    return seq[-length:]


def compute_novelty(df: pd.DataFrame, ref_seq: str, ref_region: str, logger) -> pd.Series:
    """
    Compute novelty = normalised Levenshtein edit distance from 2×35S.
    Higher = more novel (more different from 35S).
    Normalised by max possible distance (length of shorter sequence).
    """
    ref_len = len(ref_region)
    max_dist = ref_len  # max possible edits = length of reference region

    novelty = {}
    for cid in df.index:
        cand_seq = df.loc[cid, "sequence"]
        cand_region = _seq_to_region(cand_seq, ref_len)
        dist = levenshtein(ref_region, cand_region)
        # Normalise: 0 = identical to 35S, 1 = maximally different
        novelty[cid] = round(min(dist, max_dist) / max_dist, 4)

    novelty_series = pd.Series(novelty, name="novelty_35s")
    log_info(logger, f"Novelty range: {novelty_series.min():.3f} – {novelty_series.max():.3f} "
                     f"(higher = more different from 35S)")
    return novelty_series


def compute_internal_diversity(df: pd.DataFrame, logger) -> pd.Series:
    """
    Compute internal diversity for each candidate = mean pairwise Levenshtein
    distance to all other candidates, normalised by median sequence length.
    Higher = more different from the rest of the cohort (more unique).
    """
    cids = list(df.index)
    seqs = [df.loc[c, "sequence"] for c in cids]
    n = len(cids)
    seq_lens = [len(s) for s in seqs]
    median_len = int(np.median(seq_lens))

    diversity = {}
    for i, cid in enumerate(cids):
        si = _seq_to_region(seqs[i], median_len)
        total_dist = 0
        count = 0
        for j, (cj, sj) in enumerate(zip(cids, seqs)):
            if i == j:
                continue
            sj_r = _seq_to_region(sj, median_len)
            total_dist += levenshtein(si, sj_r)
            count += 1
        diversity[cid] = round(total_dist / (count * median_len), 4) if count > 0 else 0.0

    div_series = pd.Series(diversity, name="internal_div")
    log_info(logger, f"Internal diversity range: {div_series.min():.3f} – {div_series.max():.3f}")
    return div_series


# =============================================================================
# PARETO RANKING
# =============================================================================
def pareto_rank(df: pd.DataFrame, objectives: list[str],
                higher_is_better: dict[str, bool]) -> pd.Series:
    """
    Assign Pareto non-dominated front ranks.
    Front 1 = non-dominated (best). Front 2 = dominated only by front 1, etc.

    Args:
        df: DataFrame with one row per candidate, one column per objective.
        objectives: list of column names to consider.
        higher_is_better: dict mapping objective name → True if higher is better.
    Returns:
        pd.Series with Pareto front number per candidate (1 = best front).
    """
    n = len(df)
    ranks = [1] * n
    remaining = list(range(n))

    # obj_vals[o][i] = value of objective o for candidate at index i
    obj_vals = {o: df[o].values.tolist() for o in objectives}

    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                all_better = all(
                    (obj_vals[o][j] > obj_vals[o][i] if higher_is_better[o]
                     else obj_vals[o][j] < obj_vals[o][i])
                    for o in objectives
                )
                any_different = any(
                    obj_vals[o][j] != obj_vals[o][i]
                    for o in objectives
                )
                if all_better and any_different:
                    dominated = True
                    break
            if not dominated:
                front.append(i)

        for i in front:
            remaining.remove(i)
        for i in remaining:
            ranks[i] += 1

    return pd.Series(ranks, index=df.index, name="pareto_front")


# =============================================================================
# NULL CONTROL CHECK
# =============================================================================
def check_null_contamination(df_ranked: pd.DataFrame, logger):
    """
    Warn if null-seed candidates appear in the top 10.
    If they do, Evo2 generation quality was poor for the real seeds.
    """
    top10_ids = list(df_ranked.head(10).index)
    null_in_top10 = [cid for cid in top10_ids if "null" in cid.lower()]
    if null_in_top10:
        log_warn(logger, f"{len(null_in_top10)} null-seed candidates appear in top 10:")
        for cid in null_in_top10:
            rank = list(df_ranked.index).index(cid) + 1
            log_warn(logger, f"  Rank {rank}: {cid}")
        log_warn(logger, "This suggests Evo2 generation quality was low.")
        log_warn(logger, "Try: EVO2_MODEL = 'evo2-40b' in config.py and rerun step 3.")
    else:
        log_ok(logger, "Null control check passed — no null-seed candidates in top 10.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    logger = setup_logger(config.LOG_FILE)
    log_step(logger, 4, "Filtering and Ranking All 60 Candidates")

    # ── Load candidates ───────────────────────────────────────────────────────
    if not os.path.exists(config.CANDIDATES_FASTA):
        log_err(logger, "Run step3_generate_candidates.py first.")
        sys.exit(1)

    candidates  = load_fasta(config.CANDIDATES_FASTA)
    references  = load_fasta(config.REF_FASTA)
    ref_35s     = references.get("2x_CaMV_35S", "")

    log_ok(logger, f"Loaded {len(candidates)} candidates")

    # ── ANALYSIS 1: Cis-element scoring ──────────────────────────────────────
    log_info(logger, "")
    log_info(logger, "Running cis-element analysis...")

    rows = {}
    for cid, seq in tqdm(candidates.items(), desc="Scoring"):
        s  = score_sequence(seq)
        gc = (seq.count("G") + seq.count("C")) / len(seq) * 100
        rows[cid] = {
            "sequence":       seq,
            "seed":           "_".join(cid.split("_")[:-1]),
            "length_bp":      len(seq),
            "gc_pct":         round(gc, 1),
            "total_elements": s["total_elements"],
            "weighted_score": s["weighted_score"],
            "TATA_box":       s["TATA_box"],
            "CAAT_box":       s["CAAT_box"],
            "GCN4_motif":     s.get("GCN4_motif", 0),
            "ocs_like":       s.get("ocs_like", 0),
            "as1_element":    s["as1_element"],
            "G_box":          s["G_box"],
            "W_box":          s["W_box"],
            "has_TATA":       s["TATA_box"] > 0,
            "has_CAAT":       s["CAAT_box"] > 0,
        }

    df = pd.DataFrame(rows).T

    # ── HARD FILTER ───────────────────────────────────────────────────────────
    # G-box is NOT in the hard filter (2×35S itself has G-box = 0)
    n_before = len(df)
    mask = (
          df["has_TATA"].astype(bool)
        & df["has_CAAT"].astype(bool)
        & df["gc_pct"].astype(float).between(config.MIN_GC_PCT, config.MAX_GC_PCT)
    )
    df_f = df[mask].copy()
    n_removed = n_before - len(df_f)

    log_ok(logger, f"Hard filter: {n_before} → {len(df_f)} candidates "
                   f"({n_removed} removed: missing TATA/CAAT or GC out of 40–60%)")

    # ── ANALYSIS 2: DNABERT-2 cosine similarity ───────────────────────────────
    log_info(logger, "")
    tokenizer, model = load_dnabert2(logger)
    use_dnabert = (tokenizer is not None and model is not None and ref_35s)

    if use_dnabert:
        ref_emb = embed(ref_35s, tokenizer, model)
        log_ok(logger, "Reference 2×35S embedded via DNABERT-2.")

        sims = {}
        for cid in tqdm(df_f.index, desc="DNABERT-2"):
            seq = df_f.loc[cid, "sequence"]
            emb = embed(seq, tokenizer, model)
            sims[cid] = round(cosine_sim(ref_emb, emb), 4)

        df_f["dnabert2_sim"] = df_f.index.map(sims)
        log_ok(logger, "DNABERT-2 cosine similarity computed.")
    else:
        log_warn(logger, "DNABERT-2 unavailable — using k-mer frequency similarity instead.")
        log_warn(logger, "NOTE: k-mer similarity is a model-free approximation.")
        log_warn(logger, "It does NOT use learned sequence representations.")
        log_warn(logger, "Computing 4-mer frequency profiles for all candidates...")
        sims = {}
        for cid in tqdm(df_f.index, desc="k-mer sim"):
            seq = df_f.loc[cid, "sequence"]
            sims[cid] = round(kmer_similarity(seq, ref_35s, k=4), 4)
        df_f["dnabert2_sim"] = df_f.index.map(sims)

        # Check for suspicious uniformity — if all values are identical,
        # something is wrong with the k-mer computation
        unique_sims = set(sims.values())
        if len(unique_sims) <= 2:
            log_warn(logger, f"WARNING: k-mer similarity has only {len(unique_sims)} unique values!")
            log_warn(logger, "  This suggests the similarity metric is not discriminating.")
            log_warn(logger, "  Consider fixing DNABERT-2 loading or using a different metric.")
        else:
            log_ok(logger, f"k-mer similarity computed (range: "
                           f"{min(sims.values()):.3f} – {max(sims.values()):.3f}, "
                           f"{len(unique_sims)} unique values).")

    # ── ANALYSIS 3: NOVELTY & DIVERSITY ───────────────────────────────────────
    log_info(logger, "")
    log_info(logger, "Computing novelty and internal diversity...")
    ref_region = _seq_to_region(ref_35s, 300)

    novelty = compute_novelty(df_f, ref_35s, ref_region, logger)
    df_f["novelty_35s"] = df_f.index.map(novelty.to_dict())

    internal_div = compute_internal_diversity(df_f, logger)
    df_f["internal_div"] = df_f.index.map(internal_div.to_dict())

    # ── PARETO RANKING ─────────────────────────────────────────────────────────
    log_info(logger, "")
    log_info(logger, "Computing Pareto front across 3 objectives:")
    log_info(logger, "  [1] strength  = weighted cis-element score")
    log_info(logger, "  [2] novelty   = edit distance from 2×35S")
    log_info(logger, "  [3] diversity = mean edit distance to other candidates")

    objectives = ["weighted_score", "novelty_35s", "internal_div"]
    higher = {"weighted_score": True, "novelty_35s": True, "internal_div": True}
    pareto = pareto_rank(df_f, objectives, higher)
    df_f["pareto_front"] = pareto

    n_pareto1 = (df_f["pareto_front"] == 1).sum()
    log_ok(logger, f"Pareto front 1 (best): {n_pareto1} candidates")
    log_info(logger, f"Remaining candidates spread across {df_f['pareto_front'].max()} fronts")

    # ── COMPOSITE SCORE (multi-objective) ─────────────────────────────────────
    # Normalise each objective to [0, 1]
    for obj in objectives:
        vals = df_f[obj].astype(float)
        mn, mx = vals.min(), vals.max()
        if mx > mn:
            df_f[f"{obj}_norm"] = (vals - mn) / (mx - mn)
        else:
            df_f[f"{obj}_norm"] = 0.5

    # Weighted composite: 50% strength + 25% novelty + 25% diversity
    df_f["composite_score"] = (
          df_f["weighted_score_norm"].astype(float)  * 0.50
        + df_f["novelty_35s_norm"].astype(float)    * 0.25
        + df_f["internal_div_norm"].astype(float)   * 0.25
    ).round(4)

    df_f = df_f.sort_values(["pareto_front", "composite_score"], ascending=[True, False])

    if len(df_f) == 0:
        log_err(logger, "No candidates passed the hard filter.")
        log_warn(logger, "Try: increase EVO2_N_VARIANTS or switch to evo2-40b in config.py")
        sys.exit(1)

    # ── NULL CONTROL CHECK ────────────────────────────────────────────────────
    log_info(logger, "")
    check_null_contamination(df_f, logger)

    # ── ANALYSIS 5: JASPAR TF BINDING PROFILE SCANNING ───────────────────────
    # Replaces/augments PlantCARE with modern JASPAR 2024 position weight matrices.
    # Scans against 1,745 plant TF profiles — far more comprehensive than regex.
    log_info(logger, "")
    log_info(logger, "Scanning against JASPAR 2024 plant TF binding profiles...")
    try:
        jaspar_profiles = fetch_plant_profiles(max_profiles=50)
        jaspar_hits = {}
        for cid in tqdm(df_f.index, desc="JASPAR scan"):
            seq = df_f.loc[cid, "sequence"]
            jaspar_hits[cid] = scan_sequence_jaspar(seq, jaspar_profiles)

        df_f["jaspar_total_hits"] = df_f.index.map(
            {c: h["jaspar_total_hits"] for c, h in jaspar_hits.items()})
        df_f["jaspar_priority_hits"] = df_f.index.map(
            {c: h["jaspar_priority_hits"] for c, h in jaspar_hits.items()})
        df_f["jaspar_weighted_score"] = df_f.index.map(
            {c: h["jaspar_weighted_score"] for c, h in jaspar_hits.items()})
        df_f["jaspar_profiles_matched"] = df_f.index.map(
            {c: h["jaspar_profiles_matched"] for c, h in jaspar_hits.items()})

        log_ok(logger, f"JASPAR scan complete: {len(jaspar_profiles)} profiles scanned.")
        log_info(logger, f"  JASPAR weighted score range: "
                 f"{df_f['jaspar_weighted_score'].min()} – "
                 f"{df_f['jaspar_weighted_score'].max()}")
    except Exception as e:
        log_warn(logger, f"JASPAR scan failed (API may be down): {e}")
        df_f["jaspar_total_hits"] = 0
        df_f["jaspar_priority_hits"] = 0
        df_f["jaspar_weighted_score"] = 0
        df_f["jaspar_profiles_matched"] = 0

    # ── ANALYSIS 6: BLAST NOVELTY CHECK (top candidates only) ────────────────
    # Verifies candidates are genuinely novel (not reproducing known GenBank seqs).
    log_info(logger, "")
    log_info(logger, "Running BLAST novelty check against GenBank nt database...")
    log_info(logger, "  (BLASTing top 10 candidates — ~30-60 seconds each)")
    try:
        top10_seqs = {cid: df_f.loc[cid, "sequence"]
                      for cid in df_f.head(10).index}
        blast_results = blast_candidates(top10_seqs, logger, max_sequences=10)
        print_blast_summary(blast_results, logger)

        # Add BLAST columns to dataframe (only for BLASTed candidates)
        blast_identity = {}
        blast_novel = {}
        for cid in df_f.index:
            if cid in blast_results:
                blast_identity[cid] = blast_results[cid]["blast_top_identity"]
                blast_novel[cid] = blast_results[cid]["blast_novel"]
            else:
                blast_identity[cid] = None
                blast_novel[cid] = None

        df_f["blast_top_identity"] = df_f.index.map(blast_identity)
        df_f["blast_novel"] = df_f.index.map(blast_novel)

        n_novel = sum(1 for v in blast_novel.values() if v is True)
        n_known = sum(1 for v in blast_novel.values() if v is False)
        log_ok(logger, f"BLAST complete: {n_novel} novel, {n_known} known "
                       f"(of {len(blast_results)} BLASTed)")

        # Flag non-novel candidates
        known_ids = [cid for cid, v in blast_novel.items()
                     if v is False]
        if known_ids:
            log_warn(logger, f"Non-novel candidates (>{95}% identity to GenBank):")
            for cid in known_ids:
                log_warn(logger, f"  {cid}: {blast_results[cid]['blast_top_identity']}% "
                                f"— {blast_results[cid]['blast_top_hit'][:60]}")

    except Exception as e:
        log_warn(logger, f"BLAST novelty check failed: {e}")
        log_warn(logger, "Continuing without BLAST scores.")
        df_f["blast_top_identity"] = None
        df_f["blast_novel"] = None

    # ── SAVE ALL SCORED ───────────────────────────────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    keep_cols = ["seed", "length_bp", "gc_pct", "total_elements", "weighted_score",
                 "TATA_box", "CAAT_box", "GCN4_motif", "ocs_like",
                 "as1_element", "G_box", "W_box",
                 "dnabert2_sim", "novelty_35s", "internal_div",
                 "jaspar_total_hits", "jaspar_priority_hits", "jaspar_weighted_score",
                 "blast_top_identity", "blast_novel",
                 "pareto_front", "composite_score"]
    df_f[keep_cols].to_csv(config.ALL_SCORED_CSV)
    log_ok(logger, f"Full scored table → {config.ALL_SCORED_CSV}")

    # ── TOP 3 (prefer real-seed candidates when DNABERT-2 is unavailable) ────
    if not use_dnabert:
        log_warn(logger, "DNABERT-2 unavailable: top-3 will prefer candidates from")
        log_warn(logger, "real seeds (35S or endogenous). Null-seed candidates are")
        log_warn(logger, "shown in the ranking CSV but excluded from top-3 FASTA.")

    # Prefer non-null-seed candidates to fill top 3
    real_cands = df_f[~df_f.index.str.contains("null", case=False)]
    null_cands = df_f[df_f.index.str.contains("null", case=False)]
    n_real = len(real_cands)
    top3_real = real_cands.head(min(3, n_real))
    remaining = 3 - len(top3_real)
    top3 = pd.concat([top3_real, null_cands.head(remaining)]) if remaining > 0 else top3_real
    top3_seqs = {}

    log_info(logger, "")
    log_info(logger, "=" * 62)
    log_info(logger, "  YOUR TOP 3 SYNTHETIC PROMOTER CANDIDATES")
    log_info(logger, "=" * 62)

    for rank, (cid, row) in enumerate(top3.iterrows(), 1):
        label = f"Rank{rank}_{cid}"
        top3_seqs[label] = row["sequence"]
        log_info(logger, "")
        log_info(logger, f"  Rank {rank}:  {cid}")
        log_info(logger, f"    Pareto front:         {int(row['pareto_front'])}")
        log_info(logger, f"    Composite score:     {row['composite_score']}")
        log_info(logger, f"    Weighted strength:   {row['weighted_score']}  (higher = stronger)")
        log_info(logger, f"    Novelty vs 35S:     {row['novelty_35s']}  (higher = more different)")
        log_info(logger, f"    Internal diversity:  {row['internal_div']}  (higher = more unique)")
        log_info(logger, f"    GC content:          {row['gc_pct']}%")
        log_info(logger, f"    as-1 elements:       {int(row['as1_element'])}  [5x weight]")
        log_info(logger, f"    GCN4 motif:         {int(row['GCN4_motif'])}  [5x weight]")
        log_info(logger, f"    ocs-like elements:   {int(row['ocs_like'])}  [3x weight]")
        log_info(logger, f"    TATA box:            {int(row['TATA_box'])}")
        log_info(logger, f"    CAAT box:            {int(row['CAAT_box'])}")
        log_info(logger, f"    G-box:               {int(row['G_box'])}")
        log_info(logger, f"    DNABERT-2 sim:       {row['dnabert2_sim']}  (vs 2×35S)")
        log_info(logger, f"    Sequence ({len(row['sequence'])} bp):")
        log_info(logger, f"    {row['sequence'][:80]}...")

    save_fasta(top3_seqs, config.TOP3_FASTA)
    log_ok(logger, f"\nTop 3 FASTA → {config.TOP3_FASTA}")

    # Save ranking table
    ranking_cols = ["seed", "gc_pct", "weighted_score", "total_elements",
                    "TATA_box", "CAAT_box", "GCN4_motif", "ocs_like",
                    "as1_element", "G_box", "dnabert2_sim",
                    "novelty_35s", "internal_div",
                    "jaspar_weighted_score", "blast_top_identity", "blast_novel",
                    "pareto_front", "composite_score"]
    df_f[ranking_cols].head(10).to_csv(config.RANKING_CSV)
    log_ok(logger, f"Top-10 ranking table → {config.RANKING_CSV}")

    # ── NEXT STEPS ────────────────────────────────────────────────────────────
    log_info(logger, "")
    log_info(logger, "NEXT STEPS:")
    log_info(logger, f"  1. Open {config.TOP3_FASTA}")
    log_info(logger, "  2. Go to: https://bioinformatics.psb.ugent.be/webtools/plantcare")
    log_info(logger, "  3. Paste each sequence (one at a time) and run analysis")
    log_info(logger, "  4. Check each result for TATA box, CAAT box, as-1 elements")
    log_info(logger, "  5. Discard any candidate where PlantCARE shows NO TATA box")
    log_info(logger, "  6. Remaining candidates go to AlphaGenome validation")


if __name__ == "__main__":
    main()
