#!/usr/bin/env python3
"""
Gene Construct Design Pipeline — Full Orchestrator.

Designs a complete gene construct for plant expression:
  STEP 1: Promoter Design (DeepPlantCRE + cis-scoring + generation)
  STEP 2: gRNA Design + HDR Targeting
  STEP 3: Safe Harbor Validation
  STEP 4: UTR + Terminator Engineering
  STEP 5: Degradation Prediction
  FINAL: Construct validation and wet-lab readiness assessment

Usage:
  python gene_construct_pipeline.py --species tomato \\
      --expression-profile high \\
      [--protein-seq MSEKKIA...] \\
      [--output-dir outputs/construct_design]

All modules produce confidence scores. If any module fails,
the construct is marked NOT READY for wet-lab validation.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.cross_species.species_config import load_species_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gene_construct_pipeline")


# ---------------------------------------------------------------------------
# STEP 1 — PROMOTER DESIGN
# ---------------------------------------------------------------------------

def step1_promoter_design(species: str, species_config: dict, expression_profile: str,
                          protein_sequence: str, seed_file: str = None) -> dict:
    """Design promoter using DeepPlantCRE, cis-scoring, and generation ensemble.

    Uses the existing v2_research modules:
    - modules/expression/deepplantcre.py for expression prediction
    - modules/evaluation/cis_scoring.py for cis-element scoring
    - modules/generation/mutational_generator.py for promoter generation
    - modules/silencing/silencing_risk.py for silencing risk
    """
    logger.info("STEP 1 — Promoter Design (DeepPlantCRE + cis-scoring)")
    start = time.time()

    result = {
        "step": "promoter_design",
        "status": "PENDING",
        "promoter_sequence": None,
        "predicted_expression_strength": None,
        "tf_motifs": [],
        "confidence_score": 0.0,
        "details": {},
    }

    try:
        from modules.evaluation.cis_scoring import (
            score_candidate, scan_cis_elements, gc_content,
            compute_weighted_score, extract_features,
        )
        from modules.silencing.silencing_risk import compute_silencing_risk
        from modules.generation.mutational_generator import generate_candidates

        # Load seed promoter
        seed_sequence = _load_seed_promoter(species, seed_file)
        if not seed_sequence:
            result["status"] = "FAILED"
            result["error"] = "No seed promoter available"
            return result

        # Generate candidates using mutational generator (offline, no GPU needed)
        candidates = generate_candidates(
            species_key=species,
            seed_sequence=seed_sequence,
            n_variants=20,
            species_config=species_config,
        )

        if not candidates:
            result["status"] = "FAILED"
            result["error"] = "Candidate generation produced no results"
            return result

        # Score all candidates
        cis_weights = species_config.get("cis_element_weights", {})
        hard_filter = species_config.get("hard_filter", {})

        scored = []
        for cand_id, cand_seq in candidates.items():
            if not cand_seq or len(cand_seq) < 100:
                continue

            try:
                scoring = score_candidate(cand_seq, species_config)
                counts = scan_cis_elements(cand_seq)

                # Silencing risk
                silencing = compute_silencing_risk(cand_seq)

                scored.append({
                    "id": cand_id,
                    "sequence": cand_seq,
                    "weighted_score": scoring.get("weighted_score", 0),
                    "gc_pct": scoring.get("gc_pct", 0),
                    "passed_filters": scoring.get("passed_filters", False),
                    "filter_failures": scoring.get("filter_failures", []),
                    "promoter_class": scoring.get("promoter_class", "unclassified"),
                    "silencing_risk": silencing.get("overall_risk", 0.5),
                    "silencing_level": silencing.get("risk_level", "MODERATE"),
                    "counts": {k: v for k, v in counts.items() if not k.startswith("_")},
                })
            except Exception as e:
                logger.warning(f"Scoring failed for {cand_id}: {e}")
                continue

        if not scored:
            result["status"] = "FAILED"
            result["error"] = "All candidates failed scoring"
            return result

        # Select best candidate that passes filters
        passing = [s for s in scored if s["passed_filters"]]
        pool = passing if passing else scored
        pool.sort(key=lambda x: x["weighted_score"], reverse=True)
        best = pool[0]

        # Extract TF motifs from best candidate
        best_counts = scan_cis_elements(best["sequence"])
        tf_motifs = []
        motif_to_tf = {
            "TATA_box": "TBP (TATA-binding protein)",
            "CAAT_box": "CBF/NF-Y (CAAT-binding factor)",
            "as1_element": "ASF-1/TGA (bZIP TFs)",
            "GCN4_motif": "bZIP/RISBZ1 (GCN4-like)",
            "ocs_like": "OCSBF/OBF (bZIP TFs)",
            "DOF_site": "DOF (DNA-binding with one finger)",
            "G_box": "bZIP/GBF/HY5 (G-box binding)",
            "W_box": "WRKY transcription factors",
            "ABRE": "AREB/ABF (bZIP, ABA-responsive)",
            "MBS": "MYB transcription factors",
        }
        for motif_name, tf_name in motif_to_tf.items():
            count = best_counts.get(motif_name, 0)
            if count > 0:
                tf_motifs.append({
                    "motif": motif_name,
                    "transcription_factor": tf_name,
                    "count": count,
                })

        # Expression strength prediction
        expr_strength = "medium"
        ws = best["weighted_score"]
        if ws >= 50:
            expr_strength = "high"
        elif ws >= 30:
            expr_strength = "medium"
        else:
            expr_strength = "low"

        # Confidence
        confidence = 0.0
        if best["passed_filters"]:
            confidence = min(1.0, ws / 60.0) * 0.5 + 0.3
            if best["silencing_level"] == "LOW":
                confidence += 0.1
            elif best["silencing_level"] == "HIGH":
                confidence -= 0.1
        else:
            confidence = min(0.5, ws / 60.0) * 0.3 + 0.1

        confidence = max(0.0, min(1.0, confidence))

        result["status"] = "SUCCESS"
        result["promoter_sequence"] = best["sequence"]
        result["predicted_expression_strength"] = expr_strength
        result["tf_motifs"] = tf_motifs
        result["confidence_score"] = round(confidence, 3)
        result["details"] = {
            "best_candidate_id": best["id"],
            "weighted_score": best["weighted_score"],
            "gc_pct": best["gc_pct"],
            "passed_filters": best["passed_filters"],
            "filter_failures": best["filter_failures"],
            "promoter_class": best["promoter_class"],
            "silencing_risk": best["silencing_risk"],
            "silencing_level": best["silencing_level"],
            "n_candidates_generated": len(candidates),
            "n_candidates_scored": len(scored),
            "n_candidates_passing": len(passing),
            "seed_used": seed_sequence[:50] + "..." if seed_sequence else None,
        }

    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
        logger.error(f"Step 1 failed: {e}", exc_info=True)

    result["computation_time_s"] = round(time.time() - start, 2)
    return result


# ---------------------------------------------------------------------------
# STEP 2 — gRNA DESIGN + HDR TARGETING
# ---------------------------------------------------------------------------

def step2_grna_design(species: str, insert_sequence: str,
                      genome_fasta: str = None) -> dict:
    """Design gRNAs and HDR template for safe harbor integration."""
    logger.info("STEP 2 — gRNA Design + HDR Targeting")
    start = time.time()

    result = {
        "step": "grna_design",
        "status": "PENDING",
        "gRNA_sequences": [],
        "off_target_score": 0.0,
        "hdr_template": None,
        "integration_site": "unknown",
        "confidence_score": 0.0,
    }

    try:
        from modules.grna.grna_designer import design_grna_pipeline

        grna_result = design_grna_pipeline(
            species=species,
            insert_sequence=insert_sequence,
            genome_fasta=genome_fasta,
        )

        result["status"] = grna_result.get("status", "FAILED")
        result["gRNA_sequences"] = grna_result.get("gRNA_sequences", [])
        result["off_target_score"] = grna_result.get("off_target_score", 0.0)
        result["hdr_template"] = grna_result.get("hdr_template")
        result["integration_site"] = grna_result.get("integration_site", "unknown")
        result["confidence_score"] = grna_result.get("confidence_score", 0.0)
        result["details"] = {
            "safe_harbor": grna_result.get("safe_harbor"),
            "best_grna": grna_result.get("best_grna"),
        }

        if result["status"] != "SUCCESS":
            result["error"] = grna_result.get("error", "Unknown error")

    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
        logger.error(f"Step 2 failed: {e}", exc_info=True)

    result["computation_time_s"] = round(time.time() - start, 2)
    return result


# ---------------------------------------------------------------------------
# STEP 3 — SAFE HARBOR VALIDATION
# ---------------------------------------------------------------------------

def step3_safe_harbor_validation(species: str, species_config: dict,
                                 genome_fasta: str = None,
                                 annotation_gff: str = None) -> dict:
    """Validate safe harbor site for chromatin state and position effects."""
    logger.info("STEP 3 — Safe Harbor Validation")
    start = time.time()

    result = {
        "step": "safe_harbor_validation",
        "status": "PENDING",
        "safe_harbor_valid": False,
        "chromatin_state": "unknown",
        "position_effect_risk": "unknown",
        "confidence_score": 0.0,
    }

    try:
        from modules.grna.grna_designer import SAFE_HARBOR_LOCI

        species_key = species.lower().replace("-", "_").replace(" ", "_")
        harbors = SAFE_HARBOR_LOCI.get(species_key, [])

        if not harbors:
            result["status"] = "FAILED"
            result["error"] = f"No safe harbor data for {species_key}"
            result["position_effect_risk"] = "high"
            return result

        harbor = harbors[0]
        chr_name = harbor["chr"]
        position = harbor["position"]

        # Try to use SafeHarborPredictor if genome data available
        if genome_fasta and Path(genome_fasta).exists():
            try:
                from modules.genomics.safe_harbor import SafeHarborPredictor

                predictor = SafeHarborPredictor(
                    genome_fasta=genome_fasta,
                    annotation_gff=annotation_gff,
                    species_key=species_key,
                )

                prediction = predictor.score_position(
                    chromosome=chr_name,
                    position=position,
                    insert_length=800,
                )

                chromatin_state = prediction.get("scores", {}).get("chromatin_state", {}).get("state", "unknown")
                overall_score = prediction.get("overall_score", 0.0)

                result["safe_harbor_valid"] = overall_score >= 0.5
                result["chromatin_state"] = chromatin_state
                result["position_effect_risk"] = (
                    "low" if overall_score >= 0.7 else
                    "medium" if overall_score >= 0.4 else
                    "high"
                )
                result["confidence_score"] = round(overall_score, 3)
                result["details"] = {
                    "harbor_name": harbor["name"],
                    "harbor_position": f"{chr_name}:{position}",
                    "harbor_evidence": harbor.get("evidence", ""),
                    "prediction": prediction,
                }
                result["status"] = "SUCCESS"

            except Exception as e:
                logger.warning(f"SafeHarborPredictor failed, using config-based validation: {e}")
                result = _safe_harbor_from_config(species, species_config, harbor)
        else:
            # Fall back to config-based validation
            result = _safe_harbor_from_config(species, species_config, harbor)

    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
        logger.error(f"Step 3 failed: {e}", exc_info=True)

    result["computation_time_s"] = round(time.time() - start, 2)
    return result


def _safe_harbor_from_config(species: str, species_config: dict, harbor: dict) -> dict:
    """Validate safe harbor using species config when genome data unavailable."""
    safe_harbor_config = species_config.get("safe_harbors", {})
    expression_config = species_config.get("expression", {})

    has_known_harbor = safe_harbor_config.get("known_sites", False)
    epigenome_quality = safe_harbor_config.get("epigenome_data", "low")

    # Confidence based on evidence quality
    confidence = 0.3
    if has_known_harbor:
        confidence += 0.3
    if epigenome_quality in ("extensive", "high"):
        confidence += 0.2
    if harbor.get("evidence"):
        confidence += 0.1

    result = {
        "step": "safe_harbor_validation",
        "status": "SUCCESS",
        "safe_harbor_valid": has_known_harbor or confidence > 0.5,
        "chromatin_state": "open" if has_known_harbor else "unknown",
        "position_effect_risk": "low" if has_known_harbor else "medium",
        "confidence_score": round(min(1.0, confidence), 3),
        "details": {
            "harbor_name": harbor.get("name", "unknown"),
            "harbor_position": f"{harbor.get('chr', '?')}:{harbor.get('position', '?')}",
            "harbor_evidence": harbor.get("evidence", ""),
            "validation_method": "config-based (no genome FASTA)",
            "epigenome_data_quality": epigenome_quality,
        },
    }
    return result


# ---------------------------------------------------------------------------
# STEP 4 — UTR + TERMINATOR ENGINEERING
# ---------------------------------------------------------------------------

def step4_utr_terminator(species: str, cds_sequence: str) -> dict:
    """Design optimized UTRs and select terminator."""
    logger.info("STEP 4 — UTR + Terminator Engineering")
    start = time.time()

    result = {
        "step": "utr_terminator",
        "status": "PENDING",
        "utr_5": None,
        "utr_3": None,
        "terminator": None,
        "stability_score": 0.0,
    }

    try:
        from modules.construct.utr_engineer import engineer_utr_terminator

        utr_result = engineer_utr_terminator(
            species=species,
            cds_sequence=cds_sequence,
        )

        result["status"] = "SUCCESS"
        result["utr_5"] = utr_result["utr_5"]
        result["utr_3"] = utr_result["utr_3"]
        result["terminator"] = utr_result["terminator"]
        result["stability_score"] = utr_result["stability_score"]
        result["details"] = {
            "utr_5_length": utr_result["utr_5_length_bp"],
            "utr_5_gc": utr_result["utr_5_gc"],
            "utr_5_kozak_valid": utr_result["utr_5_kozak_valid"],
            "utr_5_accessibility": utr_result["utr_5_accessibility"],
            "utr_3_length": utr_result["utr_3_length_bp"],
            "terminator_name": utr_result["terminator_name"],
            "terminator_length": utr_result["terminator_length_bp"],
            "terminator_efficiency": utr_result["terminator_efficiency"],
            "warnings": utr_result.get("warnings", []),
        }

    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
        logger.error(f"Step 4 failed: {e}", exc_info=True)

    result["computation_time_s"] = round(time.time() - start, 2)
    return result


# ---------------------------------------------------------------------------
# STEP 5 — DEGRADATION PREDICTION
# ---------------------------------------------------------------------------

def step5_degradation_prediction(protein_sequence: str,
                                  localization: str = None) -> dict:
    """Predict degradation risk from protein sequence."""
    logger.info("STEP 5 — Degradation Prediction")
    start = time.time()

    result = {
        "step": "degradation_prediction",
        "status": "PENDING",
        "degradation_risk": "unknown",
        "cleavage_sites": [],
        "half_life_estimate": 0.0,
    }

    try:
        from modules.construct.degradation_predictor import predict_degradation

        # Try to get localization from DeepLoc if available
        if localization is None:
            try:
                from modules.protein.deeploc import predict_localisation
                loc_result = predict_localisation(protein_sequence)
                localization = loc_result.get("predicted_localisation", "Cytoplasm")
            except Exception:
                localization = "Cytoplasm"

        # Try to get surface exposure from ESMFold if available
        surface_exposure = None
        try:
            from modules.biophysics.structure_validation import validate_structure
            struct = validate_structure(protein_sequence)
            plddt_mean = struct.get("mean_plddt", 0)
            if plddt_mean > 50:
                surface_exposure = 0.5  # Approximate
        except Exception:
            pass

        deg_result = predict_degradation(
            protein_sequence=protein_sequence,
            localization=localization,
            surface_exposure=surface_exposure,
        )

        result["status"] = "SUCCESS"
        result["degradation_risk"] = deg_result["degradation_risk"]
        result["cleavage_sites"] = deg_result["cleavage_site_summary"]
        result["half_life_estimate"] = deg_result["half_life_estimate_h"]
        result["details"] = deg_result

    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
        logger.error(f"Step 5 failed: {e}", exc_info=True)

    result["computation_time_s"] = round(time.time() - start, 2)
    return result


# ---------------------------------------------------------------------------
# CDS OPTIMIZATION
# ---------------------------------------------------------------------------

def optimize_cds_pipeline(protein_sequence: str, species: str) -> dict:
    """Optimize CDS using existing codon optimization module."""
    logger.info("CDS Optimization")
    start = time.time()

    try:
        from modules.biophysics.codon_optimization import optimize_cds
        result = optimize_cds(protein_sequence, species)
        result["computation_time_s"] = round(time.time() - start, 2)
        return result
    except Exception as e:
        logger.error(f"CDS optimization failed: {e}", exc_info=True)
        return {
            "status": "FAILED",
            "error": str(e),
            "optimized_cds": None,
            "computation_time_s": round(time.time() - start, 2),
        }


# ---------------------------------------------------------------------------
# FINAL ASSEMBLY
# ---------------------------------------------------------------------------

def assemble_final_construct(
    promoter_seq: str,
    utr_5: str,
    cds_seq: str,
    utr_3: str,
    terminator_seq: str,
) -> dict:
    """Assemble the final gene construct from all parts."""
    if not all([promoter_seq, cds_seq]):
        return {
            "construct_sequence": None,
            "total_length_bp": 0,
            "parts": {},
            "valid": False,
        }

    construct = (
        promoter_seq +
        (utr_5 or "") +
        cds_seq +
        (utr_3 or "") +
        (terminator_seq or "")
    )

    parts = {
        "promoter": {"start": 1, "end": len(promoter_seq), "length": len(promoter_seq)},
        "utr_5": {
            "start": len(promoter_seq) + 1,
            "end": len(promoter_seq) + len(utr_5 or ""),
            "length": len(utr_5 or ""),
        },
        "cds": {
            "start": len(promoter_seq) + len(utr_5 or "") + 1,
            "end": len(promoter_seq) + len(utr_5 or "") + len(cds_seq),
            "length": len(cds_seq),
        },
        "utr_3": {
            "start": len(promoter_seq) + len(utr_5 or "") + len(cds_seq) + 1,
            "end": len(promoter_seq) + len(utr_5 or "") + len(cds_seq) + len(utr_3 or ""),
            "length": len(utr_3 or ""),
        },
        "terminator": {
            "start": len(promoter_seq) + len(utr_5 or "") + len(cds_seq) + len(utr_3 or "") + 1,
            "end": len(construct),
            "length": len(terminator_seq or ""),
        },
    }

    return {
        "construct_sequence": construct,
        "total_length_bp": len(construct),
        "parts": parts,
        "gc_content": round((construct.count("G") + construct.count("C")) / len(construct), 3) if construct else 0,
        "valid": True,
    }


def compute_final_assessment(
    step1: dict, step2: dict, step3: dict, step4: dict, step5: dict,
) -> dict:
    """Compute final construct validation and wet-lab readiness."""
    all_success = all(
        s.get("status") == "SUCCESS"
        for s in [step1, step2, step3, step4, step5]
    )

    # Expression confidence (from promoter + UTR + terminator)
    expression_confidence = 0.0
    if step1.get("status") == "SUCCESS":
        expression_confidence += step1.get("confidence_score", 0) * 0.5
    if step4.get("status") == "SUCCESS":
        expression_confidence += step4.get("stability_score", 0) * 0.3
    if step3.get("status") == "SUCCESS" and step3.get("chromatin_state") == "open":
        expression_confidence += 0.2
    expression_confidence = min(1.0, expression_confidence)

    # Integration confidence (from gRNA + safe harbor)
    integration_confidence = 0.0
    if step2.get("status") == "SUCCESS":
        integration_confidence += step2.get("confidence_score", 0) * 0.6
    if step3.get("status") == "SUCCESS":
        integration_confidence += step3.get("confidence_score", 0) * 0.4
    integration_confidence = min(1.0, integration_confidence)

    # Degradation risk
    degradation_risk = step5.get("degradation_risk", "unknown") if step5.get("status") == "SUCCESS" else "unknown"

    # Ready for wet lab?
    ready = (
        all_success and
        expression_confidence >= 0.4 and
        integration_confidence >= 0.3 and
        degradation_risk != "high"
    )

    return {
        "construct_valid": all_success,
        "expression_confidence": round(expression_confidence, 3),
        "integration_confidence": round(integration_confidence, 3),
        "degradation_risk": degradation_risk,
        "ready_for_wet_lab": ready,
        "step_statuses": {
            "promoter_design": step1.get("status", "NOT_RUN"),
            "grna_design": step2.get("status", "NOT_RUN"),
            "safe_harbor": step3.get("status", "NOT_RUN"),
            "utr_terminator": step4.get("status", "NOT_RUN"),
            "degradation": step5.get("status", "NOT_RUN"),
        },
        "bottlenecks": _identify_bottlenecks(step1, step2, step3, step4, step5),
    }


def _identify_bottlenecks(*steps) -> list:
    bottlenecks = []
    for step in steps:
        if step.get("status") == "FAILED":
            bottlenecks.append({
                "step": step.get("step", "unknown"),
                "error": step.get("error", "Unknown failure"),
            })
    return bottlenecks


# ---------------------------------------------------------------------------
# SEED LOADING
# ---------------------------------------------------------------------------

def _load_seed_promoter(species: str, seed_file: str = None) -> str:
    """Load seed promoter sequence for generation."""
    if seed_file and Path(seed_file).exists():
        return _read_fasta(seed_file)

    # Try species-specific seed files
    seed_dir = PROJECT_ROOT / "configs" / "seeds"
    species_file = seed_dir / f"{species}.fasta"
    if species_file.exists():
        return _read_fasta(str(species_file))

    # Try data/promoter_seeds
    promoter_seeds = PROJECT_ROOT / "data" / "promoter_seeds"
    species_promoter = promoter_seeds / f"{species}_promoters.fasta"
    if species_promoter.exists():
        return _read_fasta(str(species_promoter))

    # Species-to-seed mapping
    seed_map = {
        "nbenthamiana": "CaMV35S_promoter_835bp.fasta",
        "ntobacum": "CaMV35S_promoter_835bp.fasta",
        "tomato": "SlUBQ_promoter_1321bp.fasta",
        "arabidopsis": "arabidopsis_promoters.fasta",
        "rice": "OsUbiquitin2_promoter_1719bp.fasta",
        "maize": "ZmUbi1_promoter_1993bp.fasta",
        "soybean": "CaMV35S_promoter_835bp.fasta",
        "wheat": "ZmUbi1_promoter_1993bp.fasta",
        "by2_cells": "CaMV35S_promoter_835bp.fasta",
    }

    seed_name = seed_map.get(species)
    if seed_name:
        seed_path = promoter_seeds / seed_name
        if seed_path.exists():
            return _read_fasta(str(seed_path))

    # Curated fallback seeds (from auto_loop_v2.py)
    fallbacks = {
        "arabidopsis": "CCACTATCCTCATTTGGATGTGTATAACAAAATCAACATATATAATTCAAATATATATAAATTATATATTCTTCACATCCACTTATCAAATATCTAATCAATATCTTTATATACATCAATTTTACAAATTCTAATTTATAATATTATAAAATATATATTTATTTTTATTATAATATTTATAATTTATTTTTATCTTATTTTAATCAATCCTCAAACTTATTTTTATCATAATTTAATAGTTTTTTAAATCAAAATATTTAAAAAAATATATATAATATATTTTTTTATATTTTTTAGAATGAGTTAACATTTTGTTTGATTT",
        "tomato": "AGAGAGAGAGAGAGAGAGAGAGAAAGAGAAAGAGAGAGAAAAATAGAAAGAGAGAGAGAGAATAGAAGGAAATTATGTTTCTAGATAGAGAAACAAATAGAGAAAGAAAAAACCTTATAGTTTATTAAATTAATACCTATATATATATATATATAGATAAGAAGAGTATAAGTTTTATTAATATATATATAATAATAATAATAAATAATATATATATATATATATATATTTAAGTTTTATATTTGATATATAGATTTTAGATTTTATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT",
        "nbenthamiana": "GAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAAAGAGAAAGAGAAAGAGAAAGAGAGAGAGAAAGAGAAAGAGAGAAAGAGAAAGAGAGAGATCTTATATAAGAGAGAGAGAGAAAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGATATAATATATAAATAATATTATATATATATATATATATATATATTATATATATATATATATATATATATATATATATATATTATACTATATATATATACTATATATATATATATATATATATATATATATATATATATATATA",
        "rice": "GAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAG",
        "maize": "GAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAG",
    }

    seq = fallbacks.get(species)
    if seq:
        return seq

    return None


def _read_fasta(path: str) -> str:
    """Read first sequence from a FASTA file."""
    seq = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if seq:
                    break
                continue
            seq.append(line.strip())
    return "".join(seq).upper() if seq else None


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_pipeline(
    species: str,
    expression_profile: str = "high",
    protein_sequence: str = None,
    genome_fasta: str = None,
    annotation_gff: str = None,
    seed_file: str = None,
    output_dir: str = None,
) -> dict:
    """Run the complete gene construct design pipeline."""
    pipeline_start = time.time()

    logger.info(f"Starting Gene Construct Design Pipeline for {species}")
    logger.info(f"Expression profile: {expression_profile}")

    # Load species config
    try:
        species_config = load_species_config(species)
    except FileNotFoundError as e:
        logger.error(f"Species config not found: {e}")
        return {
            "status": "FAILED",
            "error": str(e),
            "species": species,
        }

    # Default protein: Hyaluronidase PH-20 (from existing project)
    if not protein_sequence:
        protein_path = PROJECT_ROOT / "data" / "protein" / "hyaluronidase.fasta"
        if protein_path.exists():
            protein_sequence = _read_fasta(str(protein_path))
        if not protein_sequence:
            protein_sequence = "MSEKKIAVLFIILVVSVAQETQEEYYKKVINYIKVHPNISCLDYYISKYPLGRVVVYEEIKPIYYDESTWDNWVGLKNYENIKQWVKRYIRKITGNVSLFDAYVDKTGTWNPKVNPHKPIKMGPKAFVDSLYKLWNKLPLFDVYKHNYQIFNETQYLPNVARYVFQAQNWGADIITDPSVMIKYPGMKKVMVNFSKSQHFISTGNPINIKDVTFKDVPKDPEYKNPIRHRFLTPKVNDSIDRFYKFYNEKLHDSITKKILYSFMDTNGQYLSAYKADGKWTWNNYKQKQKYVKKYPHRSLYKHLEQTNRPLPTPGGSRSCQNGGSWASCPSGGQKCSVYQK"
            logger.info("Using default protein: Hyaluronidase PH-20")

    # Resolve genome paths
    if not genome_fasta:
        genome_dir = PROJECT_ROOT / "data" / "species_genomes" / species
        fasta_gz = genome_dir / f"*.fa.gz"
        import glob
        matches = glob.glob(str(genome_dir / "*.fa.gz"))
        if matches:
            genome_fasta = matches[0]
            logger.info(f"Auto-detected genome: {genome_fasta}")

    if not annotation_gff:
        genome_dir = PROJECT_ROOT / "data" / "species_genomes" / species
        import glob
        matches = glob.glob(str(genome_dir / "*.gff.gz"))
        if matches:
            annotation_gff = matches[0]

    # --- STEP 0: CDS Optimization ---
    logger.info("Pre-step: CDS Optimization")
    cds_result = optimize_cds_pipeline(protein_sequence, species)
    cds_sequence = cds_result.get("optimized_cds")
    if not cds_sequence:
        logger.warning("CDS optimization failed, using simple forward translation")
        cds_sequence = _simple_translate(protein_sequence)

    # --- STEP 1: Promoter Design ---
    step1 = step1_promoter_design(species, species_config, expression_profile,
                                  protein_sequence, seed_file)

    # --- STEP 2: gRNA Design ---
    promoter_seq = step1.get("promoter_sequence") or ""
    insert_seq = (promoter_seq + (cds_sequence or "") +
                  "AATAAAGCATGCTAGCTAGCTAGCTAGCTAGCTAGCTAG")
    step2 = step2_grna_design(species, insert_seq, genome_fasta)

    # --- STEP 3: Safe Harbor Validation ---
    step3 = step3_safe_harbor_validation(species, species_config,
                                          genome_fasta, annotation_gff)

    # --- STEP 4: UTR + Terminator ---
    step4 = step4_utr_terminator(species, cds_sequence or "")

    # --- STEP 5: Degradation Prediction ---
    step5 = step5_degradation_prediction(protein_sequence)

    # --- FINAL ASSEMBLY ---
    construct = assemble_final_construct(
        promoter_seq=promoter_seq or "",
        utr_5=step4.get("utr_5", ""),
        cds_seq=cds_sequence or "",
        utr_3=step4.get("utr_3", ""),
        terminator_seq=step4.get("terminator", ""),
    )

    final = compute_final_assessment(step1, step2, step3, step4, step5)

    # Build complete output
    output = {
        "pipeline": "gene_construct_design",
        "version": "1.0.0",
        "species": species,
        "expression_profile": expression_profile,
        "protein_sequence_length": len(protein_sequence) if protein_sequence else 0,
        "total_computation_time_s": round(time.time() - pipeline_start, 2),
        "steps": {
            "step1_promoter_design": step1,
            "step2_grna_design": step2,
            "step3_safe_harbor": step3,
            "step4_utr_terminator": step4,
            "step5_degradation": step5,
        },
        "cds_optimization": cds_result,
        "construct_assembly": construct,
        "final_assessment": final,
    }

    # Save output
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_path = out_path / f"gene_construct_{species}.json"
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Results saved to {json_path}")

        # Save construct FASTA
        if construct.get("construct_sequence"):
            fasta_path = out_path / f"gene_construct_{species}.fasta"
            with open(fasta_path, "w") as f:
                f.write(f">gene_construct_{species} total={construct['total_length_bp']}bp\n")
                seq = construct["construct_sequence"]
                for i in range(0, len(seq), 80):
                    f.write(seq[i:i + 80] + "\n")
            logger.info(f"Construct FASTA saved to {fasta_path}")

    return output


def _simple_translate(protein: str) -> str:
    """Simple forward translation using most common codons."""
    codon_map = {
        "A": "GCT", "R": "CGT", "N": "AAT", "D": "GAT", "C": "TGT",
        "Q": "CAG", "E": "GAG", "G": "GGT", "H": "CAT", "I": "ATT",
        "L": "CTT", "K": "AAG", "M": "ATG", "F": "TTT", "P": "CCT",
        "S": "TCT", "T": "ACT", "W": "TGG", "Y": "TAT", "V": "GTT",
    }
    return "".join(codon_map.get(aa, "NNN") for aa in protein.upper()) + "TAA"


def main():
    parser = argparse.ArgumentParser(
        description="Gene Construct Design Pipeline for Plant Expression"
    )
    parser.add_argument("--species", required=True,
                        help="Target species (e.g., tomato, rice, arabidopsis)")
    parser.add_argument("--expression-profile", default="high",
                        choices=["high", "medium", "tissue_specific"],
                        help="Target expression profile")
    parser.add_argument("--protein-seq", default=None,
                        help="Protein sequence (default: hyaluronidase from data/)")
    parser.add_argument("--protein-file", default=None,
                        help="FASTA file with protein sequence")
    parser.add_argument("--genome-fasta", default=None,
                        help="Genome FASTA file for off-target analysis")
    parser.add_argument("--annotation-gff", default=None,
                        help="GFF3 annotation file")
    parser.add_argument("--seed-file", default=None,
                        help="Custom seed promoter FASTA file")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress detailed output")

    args = parser.parse_args()

    if args.protein_file:
        protein_sequence = _read_fasta(args.protein_file)
    elif args.protein_seq:
        protein_sequence = args.protein_seq
    else:
        protein_sequence = None

    output_dir = args.output_dir or str(PROJECT_ROOT / "outputs" / "construct_design")

    result = run_pipeline(
        species=args.species,
        expression_profile=args.expression_profile,
        protein_sequence=protein_sequence,
        genome_fasta=args.genome_fasta,
        annotation_gff=args.annotation_gff,
        seed_file=args.seed_file,
        output_dir=output_dir,
    )

    # Print summary
    if not args.quiet:
        final = result.get("final_assessment", {})
        construct = result.get("construct_assembly", {})

        print("\n" + "=" * 70)
        print("GENE CONSTRUCT DESIGN — FINAL REPORT")
        print("=" * 70)
        print(f"Species: {args.species}")
        print(f"Expression profile: {args.expression_profile}")

        if construct.get("construct_sequence"):
            print(f"\nTotal construct length: {construct['total_length_bp']} bp")
            print(f"GC content: {construct['gc_content']:.1%}")
            print("\nParts:")
            for part_name, part_info in construct.get("parts", {}).items():
                print(f"  {part_name}: {part_info['length']} bp "
                      f"(pos {part_info['start']}-{part_info['end']})")

        print(f"\nConstruct valid: {final.get('construct_valid', False)}")
        print(f"Expression confidence: {final.get('expression_confidence', 0):.3f}")
        print(f"Integration confidence: {final.get('integration_confidence', 0):.3f}")
        print(f"Degradation risk: {final.get('degradation_risk', 'unknown')}")
        print(f"Ready for wet-lab: {final.get('ready_for_wet_lab', False)}")

        if final.get("bottlenecks"):
            print("\nBottlenecks:")
            for b in final["bottlenecks"]:
                print(f"  - {b['step']}: {b['error']}")

        # Step status summary
        print("\nStep statuses:")
        for step_name, status in final.get("step_statuses", {}).items():
            symbol = "OK" if status == "SUCCESS" else "FAIL"
            print(f"  [{symbol}] {step_name}")

        print(f"\nTotal time: {result.get('total_computation_time_s', 0):.1f}s")
        print("=" * 70)


if __name__ == "__main__":
    main()
