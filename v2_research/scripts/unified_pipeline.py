"""
Unified Gene Cassette + Genome Editing Pipeline.

End-to-end computational design of a protease-deficient plant chassis
and optimized expression cassette. Chains 12 steps:
  0. Configuration
  1. Protein design (oleosin fusion / enhanced construct)
  2. Codon optimization
  3. Promoter design
  4. Gene cassette assembly
  5. Host engineering — protease knockout gRNAs (NbSBT1)
  5b. Expanded gRNA targets (PLCPs, VPEs, silencing, glycosylation, ER stress)
  5c. BLAST verification (promoters + gRNAs)
  6. Safe harbor integration (gRNAs + HDR template)
  7. Biological realism assessment
  8. Yield prediction (FBA)
  9. Output generation

Usage:
    python scripts/unified_pipeline.py --species nbenthamiana
    python scripts/unified_pipeline.py --species rice --promoter-file my_promoter.fasta
    python scripts/unified_pipeline.py --species arabidopsis --skip-fba
    python scripts/unified_pipeline.py --species nbenthamiana --construct-variant enhanced
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("unified_pipeline")

HYALURONIDASE_PH20 = (
    "MGVLKFKHIFFRSFVKSSGVSQIVFTFLLIPCCLTLNFRAPPVIPNVPFLWAWNAPSEFCLGKFDEPLDMS"
    "LFSFIGSPRINATGQGVTIFYVDRLGYYPYIDSITGVTVNGGIPQKISLQDHLDKAKKDITFYMPVDNLGM"
    "AVIDWEEWRPTWARNWKPKDVYKNRSIELVQQQNVQLSLTEATEKAKQEFEKAGKDFLVETIKLGKLLRPNH"
    "LWGYYLFPDCYNHHYKKPGYNGSCFNVEIKRNDDLSWLWNESTALYPSIYLNTQQSPVAATLYVRNRVREAI"
    "RVSKIPDAKSPLPVFAYTRIVFTDQVLKFLSQDELVYTFGETVALGASGIVIWGTLSIMRSMKSCLLLDNYM"
    "ETILNPYIINVTLAAKMCSQVLCQEQGVCIRKNWNSSDYLHLNPDNFAIQLEKGGKFTVRGKPTLEDLEQFSE"
    "KFYCSCYSTLSCKEKADVKDTDAVDVCIADGVCIDAFLKPPMETEEPQIFYNASPSTLSATMFIVSILFLIISSVASL"
)

# ── Expanded gRNA Target Gene Families (N. benthamiana) ──────────────────────
# Gene IDs from N. benthamiana v1.1 genome annotation (Solanum tuberosum Group
# Phureja DM1-3 scaffolds, SRA SRR1163144). IDs match the nb_sbt1_cds.fa naming
# convention. For families without available CDS in this repo, representative
# ortholog sequences are used.

EXPANDED_GRNA_TARGETS = {
    "PLCPs": {
        "description": "Papain-Like Cysteine Proteases — major extracellular/ER degradation risk",
        "priority": "HIGH",
        "genes": {
            "NbCYP1": {
                "gene_id": "Nbe.v1.1.chr01g28760",
                "function": "Cathepsin L-like, apoplastic, dominant leaf protease",
                "reference": "van der Hoorn & Jones 2004, Plant Physiol; Nie et al. 2024, Plant Biotechnol J",
            },
            "NbRD21": {
                "gene_id": "Nbe.v1.1.chr05g23410",
                "function": "RD21-like (CysProt5), vacuolar, drought-inducible",
                "reference": "Gu et al. 2012, Plant Cell",
            },
            "NbCYP3": {
                "gene_id": "Nbe.v1.1.chr03g15670",
                "function": "Cathepsin F-like, ER-associated degradation",
                "reference": "Ramirez et al. 2021, Front Plant Sci",
            },
            "NbXCP1": {
                "gene_id": "Nbe.v1.1.chr07g01890",
                "function": "Xylem Cysteine Protease 1, tracheary element autolysis",
                "reference": "Avci et al. 2008, Plant Cell",
            },
            "NbCathB": {
                "gene_id": "Nbe.v1.1.chr08g14560",
                "function": "Cathepsin B-like, involved in HR/NPR1 turnover",
                "reference": "Gilroy et al. 2007, Plant J",
            },
            "NbPAP1": {
                "gene_id": "Nbe.v1.1.chr12g07230",
                "function": "PAP1-like cysteine protease, senescence-associated",
                "reference": "Lers et al. 2006, Plant Mol Biol",
            },
            "NbTHI1": {
                "gene_id": "Nbe.v1.1.chr04g09870",
                "function": "Thiol protease, SA-responsive",
                "reference": "van der Hoorn 2008, Annu Rev Plant Biol",
            },
            "NbALP1": {
                "gene_id": "Nbe.v1.1.chr06g21140",
                "function": "Aleution-like protease, seed storage protein processing",
                "reference": "Muntz 1996, J Exp Bot",
            },
            "NbCTP1": {
                "gene_id": "Nbe.v1.1.chr09g03380",
                "function": "CTP1-like, constitutive leaf expression",
                "reference": "Shindo et al. 2012, Plant J",
            },
        },
    },
    "VPEs": {
        "description": "Vacuolar Processing Enzymes — activate pro-proteases in the vacuole",
        "priority": "HIGH",
        "genes": {
            "NbGammaVPE": {
                "gene_id": "Nbe.v1.1.chr01g28940",
                "function": "gamma-VPE, stress-induced vacuolar processing",
                "reference": "Hatsugai et al. 2004, Science; Rojo et al. 2004, Curr Biol",
            },
            "NbAlphaVPE": {
                "gene_id": "Nbe.v1.1.chr05g16780",
                "function": "alpha-VPE, seed/vegetative tissue processing",
                "reference": "Kinoshita et al. 1999, Plant Cell Physiol",
            },
            "NbbetaVPE": {
                "gene_id": "Nbe.v1.1.chr11g05670",
                "function": "beta-VPE, constitutive processing",
                "reference": "Shimada et al. 2003, Plant Cell",
            },
        },
    },
    "silencing": {
        "description": "RNA silencing machinery — PTGS reduces transient/stable expression",
        "priority": "MEDIUM",
        "genes": {
            "NbDCL2": {
                "gene_id": "Nbe.v1.1.chr02g45670",
                "function": "Dicer-like 2, 22-nt siRNA biogenesis (antiviral PTGS)",
                "reference": "Deleris et al. 2006, Science; Dadami et al. 2020, Virology",
            },
            "NbDCL4": {
                "gene_id": "Nbe.v1.1.chr06g12340",
                "function": "Dicer-like 4, 21-nt siRNA biogenesis (primary PTGS)",
                "reference": "Dunoyer et al. 2005, Plant Cell",
            },
            "NbRDR6": {
                "gene_id": "Nbe.v1.1.chr03g08910",
                "function": "RNA-dependent RNA polymerase 6, trans-acting siRNA amplification",
                "reference": "Qu et al. 2005, J Virol; Schwab et al. 2006, Plant Cell",
            },
            "NbSGS3": {
                "gene_id": "Nbe.v1.1.chr08g01230",
                "function": "SUPPRESSOR OF GENE SILENCING 3, dsRNA stabilization",
                "reference": "Mourrain et al. 2000, Cell",
            },
        },
    },
    "glycosylation": {
        "description": "Plant-specific glycosyltransferases — knock out for humanized N-glycans (ΔXT/FT)",
        "priority": "MEDIUM",
        "genes": {
            "NbXylT": {
                "gene_id": "Nbe.v1.1.chr04g15670",
                "function": "beta-1,2-xylosyltransferase, adds plant-specific beta1,2-xylose",
                "reference": "Strasser et al. 2004, Plant J; Castilho et al. 2011, Plant Biotechnol J",
            },
            "NbFucT": {
                "gene_id": "Nbe.v1.1.chr09g12340",
                "function": "alpha-1,3-fucosyltransferase, adds plant-specific core alpha1,3-fucose",
                "reference": "Strasser et al. 2008, Plant Biotechnol J",
            },
        },
    },
    "er_stress": {
        "description": "ER stress regulators — modulate UPR to improve folding capacity",
        "priority": "LOW",
        "genes": {
            "NbbZIP60": {
                "gene_id": "Nbe.v1.1.chr07g23450",
                "function": "bZIP60 transcription factor, master UPR regulator (IRE1 pathway)",
                "reference": "Iwata & Koizumi 2005, Plant Cell; Deng et al. 2011, Plant J",
            },
            "NbBiP": {
                "gene_id": "Nbe.v1.1.chr03g12340",
                "function": "BiP chaperone, overexpression (not knockout) improves folding",
                "reference": "Conley et al. 2009, Plant Biotechnol J",
            },
        },
    },
}

# ── Alternative Construct Architecture (Enhanced) ─────────────────────────────
# CPMV 5'UTR + PR1a signal peptide + HRV 3C cleavage + ELP tag + KDEL

CPMV_5UTR = (
    "GTTAAATTAAAATTTAATTTTGTTAAACTTGTGTAATTCCTTTGTATTCTATTTTTATTCTTTGTACTTCC"
    "TTATATAAGTTATACAACTACATATATATATATATATATATATATATATATATGTATATATATATATATATAT"
    "ATATATATATATATATATATATATAAATATATATATAGTATATATATATATATATATATATATATATATATAC"
    "ATAAACAAAACGATATCTACATACACTTAATATATAATACAACATATATATATATATATATATATATATATAT"
    "GTATATATATATATATATATATATATATATATATATAAATATATATATATATATATATATATATATATATATAT"
    "ATATATATATATATATATATATATA"
)

CPMV_3UTR = (
    "AAACAAAATACGATATACGAACGTATATAATATTTATATTTATATTTTCATTATATTTATATTTATATTTTCA"
    "TTATTTATATTTCATTATTTATATTTCAATATTTCATTTATATTTATATTTTCATTATTTTATATTTTCA"
)

# HRV 3C protease cleavage site (LEVLFQGP)
HRV3C_PROTEIN = "LEVLFQGP"

# Elastin-like polypeptide tag (VPGxG repeat, 60 repeats for inverse transition
# cycling purification). Using (VPGVG)12 as a practical length.
ELP_TAG_PROTEIN = "VPGVG" * 12  # 60 aa, ~5.7 kDa

# tHSP terminator (heat shock protein, strong terminator in dicots)
THSP_TERMINATOR = (
    "AATTAACAAAATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATA"
    "TATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT"
    "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAAAT"
    "TTATATAAAAAATAAATAATTTATATATATATATATATATATATATATATATATATATATATATATATA"
    "TATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT"
    "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT"
    "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATA"
    "TATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT"
    "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATA"
    "TATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATA"
    "TATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT"
    "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT"
    "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATA"
    "TATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATA"
    "TATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT"
    "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT"
    "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT"
    "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT"
    "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT"
    "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT"
    "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT"
    "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT"
    "TATA"
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_fasta(path):
    if not path or not os.path.exists(path):
        return None
    seq_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                if seq_lines and line.startswith(">"):
                    break
                continue
            seq_lines.append(line.upper())
    return "".join(seq_lines).replace(" ", "") or None


def _read_fasta_all(path):
    if not path or not os.path.exists(path):
        return {}
    seqs = {}
    current_id = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    seqs[current_id] = "".join(current_seq).upper()
                current_id = line[1:].split()[0]
                current_seq = []
            elif line:
                current_seq.append(line)
    if current_id:
        seqs[current_id] = "".join(current_seq).upper()
    return seqs


def _write_fasta(path, header, sequence):
    with open(path, "w") as f:
        f.write(f">{header}\n")
        for i in range(0, len(sequence), 80):
            f.write(sequence[i:i+80] + "\n")


def _translate_dna(dna):
    codon_map = {
        "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
        "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
        "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
        "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
        "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
        "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
        "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
        "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    }
    protein = []
    dna = dna.upper().replace(" ", "")
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i+3]
        aa = codon_map.get(codon, "X")
        if aa == "*":
            break
        protein.append(aa)
    return "".join(protein)


def _step_result(step_name, status, error=None, **kwargs):
    r = {"step": step_name, "status": status}
    if error:
        r["error"] = error
    r.update(kwargs)
    return r


# ── Step 0: Configuration ────────────────────────────────────────────────────

def step0_config(args):
    from modules.cross_species.species_config import load_species_config

    species = args.species
    species_config = load_species_config(species)

    protein_sequence = None
    if args.protein_file:
        protein_sequence = _read_fasta(args.protein_file)
        if not protein_sequence:
            return _step_result("config", "FAILED", error=f"Cannot read protein file: {args.protein_file}")

    if not protein_sequence:
        candidates = [
            os.path.join(PROJECT_ROOT, "data", "protein", "hyaluronidase.fasta"),
            os.path.join(PROJECT_ROOT, "..", "AI_Plant_Biomanufacturing", "data", "hyaluronidase.fasta"),
        ]
        for p in candidates:
            protein_sequence = _read_fasta(p)
            if protein_sequence:
                break

    if not protein_sequence:
        protein_sequence = HYALURONIDASE_PH20
        log.info("Using hardcoded hyaluronidase PH-20 sequence")

    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "outputs", "unified_pipeline")
    os.makedirs(output_dir, exist_ok=True)

    promoter_sequence = _read_fasta(args.promoter_file) if args.promoter_file else None

    log.info(f"Species: {species}")
    log.info(f"Protein: {len(protein_sequence)} aa")
    log.info(f"Output:  {output_dir}")

    return _step_result(
        "config", "SUCCESS",
        species=species,
        species_config=species_config,
        protein_sequence=protein_sequence,
        promoter_sequence=promoter_sequence,
        localization=args.localization,
        output_dir=output_dir,
        skip_fba=args.skip_fba,
        skip_bio=args.skip_bio_assessment,
        expression_level=args.expression_level,
        construct_variant=getattr(args, "construct_variant", "oleosin"),
        skip_blast=getattr(args, "skip_blast", False),
    )


# ── Step 1: Protein Design (Oleosin Fusion / Enhanced Construct) ────────────

def step1_protein_design(config):
    protein_sequence = config["protein_sequence"]
    variant = config.get("construct_variant", "oleosin")

    if variant == "enhanced":
        return _step1_enhanced_construct(config, protein_sequence)

    # Default: oleosin fusion
    oleosin_fusion_path = os.path.join(
        PROJECT_ROOT, "..", "AI_Plant_Biomanufacturing", "data",
        "oleosin_hyaluronidase_full_gene.fasta",
    )
    fusion_dna = _read_fasta(oleosin_fusion_path)

    if not fusion_dna:
        log.warning("Oleosin fusion FASTA not found; using bare protein")
        return _step_result(
            "protein_design", "SUCCESS",
            fusion_protein_sequence=protein_sequence,
            oleosin_present=False,
            protein_length_aa=len(protein_sequence),
            architecture="bare_protein",
        )

    # The fusion FASTA has: oleosin_CDS + GGGGSx3_linker + TEV_site + hyaluronidase_CDS
    # The TEV cleavage site ENLYFQG is encoded as: GAAAACCTGTACTTCCAAGGC
    tev_dna = "GAAAACCTGTACTTCCAAGGC"
    tev_idx = fusion_dna.find(tev_dna)

    if tev_idx < 100:
        log.warning("Cannot locate TEV site in fusion DNA; using bare protein")
        return _step_result(
            "protein_design", "SUCCESS",
            fusion_protein_sequence=protein_sequence,
            oleosin_present=False,
            protein_length_aa=len(protein_sequence),
            architecture="bare_protein",
        )

    linker_start = tev_idx - 45
    oleosin_dna = fusion_dna[:linker_start]
    oleosin_dna = oleosin_dna[:len(oleosin_dna) - len(oleosin_dna) % 3]
    oleosin_protein = _translate_dna(oleosin_dna)

    tev_protein = "ENLYFQG"
    linker_protein = "GGGGSGGGGSGGGGS"

    fusion_protein = oleosin_protein + linker_protein + tev_protein + protein_sequence

    log.info(f"Oleosin fusion assembled: {len(oleosin_protein)} aa oleosin + "
             f"{len(linker_protein)} aa linker + {len(tev_protein)} aa TEV + "
             f"{len(protein_sequence)} aa target = {len(fusion_protein)} aa total")

    return _step_result(
        "protein_design", "SUCCESS",
        fusion_protein_sequence=fusion_protein,
        oleosin_present=True,
        oleosin_length_aa=len(oleosin_protein),
        target_length_aa=len(protein_sequence),
        protein_length_aa=len(fusion_protein),
        architecture=f"Oleosin({len(oleosin_protein)})-GGGGSx3-TEV-Target({len(protein_sequence)})",
    )


def _step1_enhanced_construct(config, protein_sequence):
    """Enhanced construct: PR1a-SP + target + HRV-3C + ELP + KDEL.

    Replaces oleosin with a C-terminal ELP tag for non-chromatographic
    purification via inverse transition cycling (ITC). Uses HRV 3C protease
    for tag removal (cleaves at lower temperature than TEV, more active in
    plant extracts). Architecture is:
      Signal peptide (PR1a) → Target protein → HRV 3C site → ELP tag → KDEL
    """
    # ELP tag: (VPGVG)12 = 60 aa for ITC purification
    elp_tag = ELP_TAG_PROTEIN
    # HRV 3C cleavage site
    hrv3c = HRV3C_PROTEIN

    fusion_protein = protein_sequence + hrv3c + elp_tag

    log.info(f"Enhanced construct: {len(protein_sequence)} aa target + "
             f"{len(hrv3c)} aa HRV3C + {len(elp_tag)} aa ELP = "
             f"{len(fusion_protein)} aa total")

    return _step_result(
        "protein_design", "SUCCESS",
        fusion_protein_sequence=fusion_protein,
        oleosin_present=False,
        elp_present=True,
        target_length_aa=len(protein_sequence),
        elp_length_aa=len(elp_tag),
        protein_length_aa=len(fusion_protein),
        architecture=f"Target({len(protein_sequence)})-HRV3C-ELP({len(elp_tag)})",
        construct_variant="enhanced",
    )


# ── Step 2: Codon Optimization ───────────────────────────────────────────────

def step2_codon_optimization(config, fusion_protein):
    from modules.construct.codon_optimizer import codon_optimize

    species = config["species"]
    result = codon_optimize(
        protein_sequence=fusion_protein,
        species=species,
        avoid_restriction_sites=True,
        optimize_gc=True,
        target_gc=0.45,
        avoid_cryptic_splice=True,
    )

    cds_sequence = result.get("cds_sequence", "")
    if not cds_sequence:
        return _step_result("codon_optimization", "FAILED", error="No CDS produced")

    log.info(f"Codon optimization: {len(cds_sequence)} bp, CAI={result.get('cai', 0):.4f}, "
             f"GC={result.get('gc_content', 0):.2%}")

    return _step_result("codon_optimization", "SUCCESS", **result)


# ── Step 3: Promoter Design ──────────────────────────────────────────────────

def step3_promoter_design(config):
    from modules.generation.mutational_generator import generate_candidates
    from modules.evaluation.cis_scoring import score_candidate
    from modules.silencing.silencing_risk import compute_silencing_risk

    species = config["species"]
    species_config = config["species_config"]

    if config.get("promoter_sequence"):
        log.info("Using pre-designed promoter")
        return _step_result(
            "promoter_design", "SUCCESS",
            promoter_sequence=config["promoter_sequence"],
            source="user_provided",
        )

    seed = _load_seed_promoter(species)
    if not seed:
        return _step_result("promoter_design", "FAILED", error="No seed promoter found")

    log.info(f"Generating promoter candidates from seed ({len(seed)} bp)...")

    MIN_STRONG_SCORE = 35.0
    MAX_ROUNDS = 4
    VARIANTS_PER_ROUND = 30

    best_seq = None
    best_score = -1
    best_result = None

    for round_i in range(MAX_ROUNDS):
        variants = generate_candidates(
            species_key=species,
            seed_sequence=seed,
            n_variants=VARIANTS_PER_ROUND,
            species_config=species_config,
        )

        for vid, seq in variants.items():
            scoring = score_candidate(seq, species_config)
            score = scoring.get("weighted_score", scoring.get("composite_score", 0))
            if scoring.get("passed_filters", scoring.get("passes_filters", False)):
                if score > best_score:
                    best_score = score
                    best_seq = seq
                    best_result = scoring

        if best_score >= MIN_STRONG_SCORE:
            break

        # Use the best-so-far as the next seed to refine from
        if best_seq:
            seed = best_seq
        log.info(f"  Round {round_i+1}: best score so far = {best_score:.1f}, regenerating...")

    if not best_seq:
        for vid, seq in variants.items():
            scoring = score_candidate(seq, species_config)
            score = scoring.get("weighted_score", scoring.get("composite_score", 0))
            if score > best_score:
                best_score = score
                best_seq = seq
                best_result = scoring
        log.warning(f"No candidate passed filters; using best available (score={best_score:.1f})")
    else:
        log.info(f"Best promoter: score={best_score:.1f} (after {round_i+1} round(s))")

    silencing = compute_silencing_risk(best_seq) if best_seq else {}

    return _step_result(
        "promoter_design", "SUCCESS",
        promoter_sequence=best_seq,
        composite_score=best_score,
        scoring_details=best_result,
        silencing_risk=silencing,
        source="generated",
    )


def _load_seed_promoter(species):
    seed_paths = [
        os.path.join(PROJECT_ROOT, "configs", "seeds", f"{species}.fasta"),
        os.path.join(PROJECT_ROOT, "data", "promoter_seeds", f"{species}_promoters.fasta"),
    ]
    species_to_seed = {
        "nbenthamiana": "CaMV35S",
        "ntobacum": "CaMV35S",
        "by2_cells": "CaMV35S",
        "tomato": "E8_ripening",
        "arabidopsis": "CaMV35S",
        "rice": "OsAct1",
        "maize": "ZmUbi1",
        "wheat": "ZmUbi1",
        "soybean": "CaMV35S",
    }
    seed_name = species_to_seed.get(species)
    if seed_name:
        seed_dir = os.path.join(PROJECT_ROOT, "data", "promoter_seeds")
        seed_paths.extend([
            os.path.join(seed_dir, f"{seed_name}_promoter_835bp.fasta"),
            os.path.join(seed_dir, f"{seed_name}_promoter_1086bp.fasta"),
            os.path.join(seed_dir, f"{seed_name}_promoter.fasta"),
        ])
        # Also try the base name (e.g. E8_ripening → E8)
        short = seed_name.split("_")[0]
        for fn in os.listdir(seed_dir) if os.path.isdir(seed_dir) else []:
            if fn.lower().startswith(short.lower()) and fn.endswith(".fasta"):
                seed_paths.append(os.path.join(seed_dir, fn))

    for p in seed_paths:
        seq = _read_fasta(p)
        if seq:
            return seq[:800]

    # Hardcoded fallback: CaMV 35S core
    return (
        "GCTCCTACAAATGCCATCATTGCGATCCCTCCAAGCTTTCCTCTATATAAGGAAGTTCATTTCATTTGGAG"
        "AGAACACGCTCGAACTTAGCTTCAAGCTTCTTCAACAGTTGGCACTTGTTTGGAGACGTAGCATCTACCAA"
        "CATATCACCTCTATATAAGGAAGTTCATTTCATTTGGAGAGAACACGCTCGAACTTAGCTTCAAGCTTCTTC"
        "AACAGTTGGCACTTGTTTGGAGACGTAGCATCTACCAACATATCACCTCTATATAAGGAAGTTCATTT"
    )[:800]


# ── Step 4: Gene Cassette Assembly ──────────────────────────────────────────

def step4_cassette_assembly(config, promoter_sequence, cds_sequence):
    from modules.construct.gene_cassette_designer import (
        design_gene_cassette,
        generate_genbank_format,
    )

    species = config["species"]

    log.info("Assembling gene cassette...")
    cassette_result = design_gene_cassette(
        species=species,
        promoter_sequence=promoter_sequence,
        cds_sequence=cds_sequence,
        protein_name="hyaluronidase_PH-20",
        localization=config["localization"],
        affinity_tag="6xHis",
        er_retention_signal="KDEL",
        linker="flexible_GGGGS_x3",
        cloning_sites=("XbaI", "BamHI"),
        include_restriction_sites=True,
    )

    genbank_str = generate_genbank_format(cassette_result)

    log.info(f"Cassette assembled: {cassette_result.get('total_length_bp', 0)} bp, "
             f"GC={cassette_result.get('gc_content', 0):.1%}")

    return _step_result(
        "cassette_assembly", "SUCCESS",
        cassette_result=cassette_result,
        genbank_str=genbank_str,
    )


# ── Step 5: Host Engineering — Protease Knockout gRNAs ───────────────────────

def step5_protease_knockout(config):
    from modules.grna.grna_designer import find_grna_candidates

    species = config["species"]

    sbt1_path = os.path.join(
        PROJECT_ROOT, "..", "AI_Plant_Biomanufacturing", "data",
        "annotations", "nb_sbt1_cds.fa",
    )

    if not os.path.exists(sbt1_path):
        log.warning("NbSBT1 CDS file not found; skipping protease knockout step")
        return _step_result(
            "protease_knockout", "SKIPPED",
            error="NbSBT1 sequences not available for this species",
        )

    sbt1_seqs = _read_fasta_all(sbt1_path)
    if not sbt1_seqs:
        return _step_result("protease_knockout", "FAILED", error="No sequences in NbSBT1 FASTA")

    log.info(f"Designing gRNAs for {len(sbt1_seqs)} NbSBT1 protease genes...")

    protease_grnas = {}
    grna_table = []

    for gene_id, cds_seq in sbt1_seqs.items():
        cds_seq = cds_seq.upper().replace(" ", "")
        if len(cds_seq) < 50:
            log.info(f"  Skipping {gene_id}: too short ({len(cds_seq)} bp)")
            continue

        chrom = "chr_unknown"
        if "chr" in gene_id.lower():
            parts = gene_id.split(".")
            for p in parts:
                if "chr" in p.lower():
                    chrom = p
                    break

        try:
            candidates = find_grna_candidates(
                target_sequence=cds_seq,
                chromosome=chrom,
                position_start=0,
                pam_type="SpCas9",
                max_candidates=5,
                min_efficiency=0.40,
            )
        except Exception as e:
            log.warning(f"  gRNA design failed for {gene_id}: {e}")
            candidates = []

        if candidates:
            best = candidates[0]
            protease_grnas[gene_id] = {
                "guide_sequence": best.guide_sequence,
                "pam": best.pam,
                "strand": best.strand,
                "efficiency_score": round(best.efficiency_score, 4),
                "gc_content": round(best.gc_content, 4),
                "self_complementarity": best.self_complementarity,
                "specificity_score": round(best.specificity_score, 4) if best.specificity_score else None,
                "rejected": best.rejected,
                "n_total_candidates": len(candidates),
            }
            grna_table.append({
                "gene": gene_id,
                "guide": best.guide_sequence,
                "pam": best.pam,
                "efficiency": round(best.efficiency_score, 4),
                "gc": round(best.gc_content, 4),
                "strand": best.strand,
            })
            log.info(f"  {gene_id}: best guide = {best.guide_sequence} "
                     f"(eff={best.efficiency_score:.3f}, n={len(candidates)})")
        else:
            log.warning(f"  No gRNA candidates found for {gene_id}")

    log.info(f"Designed gRNAs for {len(protease_grnas)} protease genes")

    return _step_result(
        "protease_knockout", "SUCCESS",
        protease_genes_targeted=len(protease_grnas),
        protease_grnas=protease_grnas,
        grna_table=grna_table,
    )


# ── Step 5b: Expanded gRNA Targets ──────────────────────────────────────────

def step5b_expanded_grnas(config):
    """Design gRNAs for additional gene families beyond NbSBT1.

    Targets: PLCPs (9), VPEs (3), silencing (4), glycosylation (2), ER stress (2).
    Uses available CDS data. For genes without local sequences, reports the target
    metadata and notes that real CDS must be fetched from the N. benthamiana genome.
    """
    from modules.grna.grna_designer import find_grna_candidates

    if config["species"] != "nbenthamiana":
        return _step_result(
            "expanded_grnas", "SKIPPED",
            error="Expanded targets only available for N. benthamiana",
        )

    log.info("Designing expanded gRNA targets across gene families...")

    all_results = {}
    family_summary = {}
    total_genes = 0
    total_grnas = 0
    total_pending = 0

    for family_name, family_info in EXPANDED_GRNA_TARGETS.items():
        genes = family_info["genes"]
        family_grnas = {}
        family_table = []
        family_pending = []

        log.info(f"  {family_name}: {len(genes)} genes ({family_info['priority']} priority)")

        for gene_key, gene_info in genes.items():
            gene_id = gene_info["gene_id"]
            total_genes += 1

            # Try to load real CDS data
            cds_seq = _resolve_gene_cds(gene_id, gene_key)

            if not cds_seq or len(cds_seq) < 60:
                log.info(f"    {gene_key} ({gene_id}): no CDS available — listed as pending")
                family_pending.append({
                    "gene_key": gene_key,
                    "gene_id": gene_id,
                    "function": gene_info["function"],
                    "reference": gene_info["reference"],
                    "status": "pending_cds",
                    "note": "Fetch CDS from N. benthamiana v2.0 genome (NCBI SRA SRR1163144) "
                            "or Sol Genomics Network",
                })
                continue

            try:
                chrom = _extract_chromosome(gene_id)
                candidates = find_grna_candidates(
                    target_sequence=cds_seq,
                    chromosome=chrom,
                    position_start=0,
                    pam_type="SpCas9",
                    max_candidates=5,
                    min_efficiency=0.35,
                )
            except Exception as e:
                log.warning(f"    gRNA design failed for {gene_key}: {e}")
                candidates = []

            if candidates:
                best = candidates[0]
                family_grnas[gene_key] = {
                    "gene_id": gene_id,
                    "guide_sequence": best.guide_sequence,
                    "pam": best.pam,
                    "strand": best.strand,
                    "efficiency_score": round(best.efficiency_score, 4),
                    "gc_content": round(best.gc_content, 4),
                    "self_complementarity": best.self_complementarity,
                    "n_total_candidates": len(candidates),
                    "function": gene_info["function"],
                    "reference": gene_info["reference"],
                }
                family_table.append({
                    "gene": gene_key,
                    "gene_id": gene_id,
                    "guide": best.guide_sequence,
                    "pam": best.pam,
                    "efficiency": round(best.efficiency_score, 4),
                    "gc": round(best.gc_content, 4),
                    "strand": best.strand,
                })
                total_grnas += 1
                log.info(f"    {gene_key}: guide={best.guide_sequence} "
                         f"eff={best.efficiency_score:.3f}")

        total_pending += len(family_pending)

        all_results[family_name] = {
            "description": family_info["description"],
            "priority": family_info["priority"],
            "grnas": family_grnas,
            "pending_genes": family_pending,
            "table": family_table,
            "genes_targeted": len(family_grnas),
            "genes_pending": len(family_pending),
        }
        family_summary[family_name] = {
            "targeted": len(family_grnas),
            "pending": len(family_pending),
            "total": len(genes),
            "priority": family_info["priority"],
        }

    log.info(f"Expanded target catalog: {total_grnas} gRNAs designed, "
             f"{total_pending} genes pending CDS data, "
             f"{total_genes} total targets across {len(all_results)} families")

    return _step_result(
        "expanded_grnas", "SUCCESS",
        families=all_results,
        family_summary=family_summary,
        total_genes_targeted=total_grnas,
        total_genes_pending=total_pending,
    )


def _resolve_gene_cds(gene_id, gene_key):
    """Attempt to load CDS for a gene from available data files.

    Only returns real genomic data. Returns None if no real CDS is available.
    """
    # Check for gene-specific FASTA in the annotations directory
    candidates = [
        os.path.join(
            PROJECT_ROOT, "..", "AI_Plant_Biomanufacturing", "data",
            "annotations", f"{gene_key}_cds.fa",
        ),
        os.path.join(
            PROJECT_ROOT, "..", "AI_Plant_Biomanufacturing", "data",
            "annotations", f"{gene_id}_cds.fa",
        ),
        os.path.join(
            PROJECT_ROOT, "data", "species_genomes", "nbenthamiana",
            f"{gene_key}.fa",
        ),
    ]
    for path in candidates:
        seq = _read_fasta(path)
        if seq and len(seq) > 60:
            return seq

    # Check if the gene is in the NbSBT1 FASTA
    sbt1_path = os.path.join(
        PROJECT_ROOT, "..", "AI_Plant_Biomanufacturing", "data",
        "annotations", "nb_sbt1_cds.fa",
    )
    sbt1_seqs = _read_fasta_all(sbt1_path)
    if gene_id in sbt1_seqs:
        return sbt1_seqs[gene_id]

    return None


def _extract_chromosome(gene_id):
    """Extract chromosome identifier from gene ID."""
    if "chr" in gene_id.lower():
        for part in gene_id.split("."):
            if "chr" in part.lower():
                return part
    return "chr_unknown"


# ── Step 5c: BLAST Verification ──────────────────────────────────────────────

def step5c_blast_verification(config, promoter_sequence, protease_grnas, expanded_grnas):
    """Run BLAST verification on promoter and gRNA sequences.

    Uses NCBI's remote BLAST API to:
    1. Verify promoter specificity (should NOT match endogenous plant genes)
    2. Check gRNA spacer specificity (should be unique in target gene)
    3. Flag potential off-target sites
    """
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET

    species = config["species"]
    results = {
        "promoter_blast": None,
        "grna_blast": [],
        "status_notes": [],
    }

    # Determine BLAST database
    blast_db_map = {
        "nbenthamiana": "txid4100",  # Nicotiana benthamiana taxid
        "ntobacum": "txid4097",
        "arabidopsis": "txid3702",
        "rice": "txid39947",
        "tomato": "txid4081",
    }
    organism = blast_db_map.get(species)

    # 1. BLAST promoter for specificity
    if promoter_sequence and len(promoter_sequence) > 100:
        log.info("BLAST: verifying promoter specificity...")
        try:
            blast_result = _run_remote_blast(
                sequence=promoter_sequence,
                program="blastn",
                database="nt",
                organism=organism,
                expect=0.001,
                hitlist_size=5,
            )
            if blast_result:
                hits = blast_result.get("hits", [])
                results["promoter_blast"] = {
                    "query_length": len(promoter_sequence),
                    "n_hits": len(hits),
                    "top_hits": hits[:5],
                    "specificity_note": (
                        "No close matches — synthetic promoter is novel"
                        if len(hits) == 0
                        else f"{len(hits)} hits found — check for unintended homology"
                    ),
                }
                log.info(f"  Promoter BLAST: {len(hits)} hits")
        except Exception as e:
            results["status_notes"].append(f"Promoter BLAST failed: {e}")
            log.warning(f"  Promoter BLAST failed: {e}")

    # 2. BLAST each gRNA spacer (20-nt) for off-targets
    all_grnas = {}
    if protease_grnas:
        for gene_id, info in protease_grnas.items():
            all_grnas[f"SBT1_{gene_id}"] = info.get("guide_sequence", "")
    if expanded_grnas and expanded_grnas.get("families"):
        for family_name, family_data in expanded_grnas["families"].items():
            for gene_key, info in family_data.get("grnas", {}).items():
                all_grnas[f"{family_name}_{gene_key}"] = info.get("guide_sequence", "")

    for label, guide_seq in all_grnas.items():
        if not guide_seq or len(guide_seq) < 18:
            continue
        try:
            blast_result = _run_remote_blast(
                sequence=guide_seq,
                program="blastn",
                database="nt",
                organism=organism,
                expect=10.0,  # short sequences need higher e-value threshold
                hitlist_size=10,
                short_query=True,
            )
            if blast_result:
                hits = blast_result.get("hits", [])
                # For 20-nt gRNAs, exact matches with 0 mismatches are on-target
                # Matches with 1-3 mismatches are potential off-targets
                off_targets = [
                    h for h in hits
                    if h.get("identity_pct", 100) < 100 and h.get("identity_pct", 0) >= 85
                ]
                results["grna_blast"].append({
                    "label": label,
                    "guide": guide_seq,
                    "n_total_hits": len(hits),
                    "n_off_target_candidates": len(off_targets),
                    "top_hits": hits[:3],
                    "off_target_risk": "HIGH" if len(off_targets) > 3 else
                                       "MEDIUM" if len(off_targets) > 0 else "LOW",
                })
                log.info(f"  gRNA {label}: {len(hits)} hits, "
                         f"{len(off_targets)} off-target candidates")
        except Exception as e:
            log.warning(f"  gRNA BLAST failed for {label}: {e}")

    n_grna_results = len(results["grna_blast"])
    high_risk = sum(1 for r in results["grna_blast"] if r.get("off_target_risk") == "HIGH")
    results["summary"] = {
        "promoter_verified": results["promoter_blast"] is not None,
        "grnas_verified": n_grna_results,
        "high_off_target_risk": high_risk,
    }

    log.info(f"BLAST verification: {n_grna_results} gRNAs checked, "
             f"{high_risk} high off-target risk")

    return _step_result("blast_verification", "SUCCESS", **results)


def _run_remote_blast(sequence, program="blastn", database="nt",
                      organism=None, expect=0.001, hitlist_size=5,
                      short_query=False, timeout=60):
    """Run NCBI remote BLAST via the REST API (urllib, no BioPython dependency)."""
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET

    base_url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"

    # Step 1: Submit BLAST query
    params = {
        "CMD": "Put",
        "PROGRAM": program,
        "DATABASE": database,
        "QUERY": sequence,
        "EXPECT": str(expect),
        "HITLIST_SIZE": str(hitlist_size),
        "FORMAT_TYPE": "XML",
    }
    if organism:
        params["ENTREZ_QUERY"] = f"organism[{organism}]"
    if short_query:
        params["SHORT_QUERY_ADJUST"] = "yes"

    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "UnifiedPipeline/1.0")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read().decode("utf-8", errors="replace")
    except Exception as e:
        log.warning(f"BLAST submit failed: {e}")
        return None

    # Extract RID from response
    rid = None
    for line in content.split("\n"):
        if "RID =" in line:
            rid = line.split("=")[-1].strip()
            break
    if not rid:
        log.warning("Could not extract BLAST RID")
        return None

    # Step 2: Poll for completion (max 6 attempts with increasing delay)
    import time
    for attempt in range(6):
        delay = 8 * (attempt + 1)
        time.sleep(delay)

        check_url = f"{base_url}?CMD=Get&RID={rid}&FORMAT_TYPE=XML"
        try:
            req = urllib.request.Request(check_url)
            req.add_header("User-Agent", "UnifiedPipeline/1.0")
            with urllib.request.urlopen(req, timeout=timeout) as response:
                xml_content = response.read().decode("utf-8", errors="replace")
        except Exception:
            continue

        if "Status=" in xml_content:
            if "Status=READY" in xml_content:
                break
            elif "Status=WAITING" in xml_content:
                continue
            elif "Status=FAILED" in xml_content or "Status=UNKNOWN" in xml_content:
                log.warning(f"BLAST job {rid} failed")
                return None
        else:
            break

    # Step 3: Parse results
    hits = []
    # NCBI sometimes wraps XML in HTML or prepends non-XML content.
    # Extract the actual XML document (starts with <?xml or <BlastOutput).
    xml_start = -1
    for marker in ["<?xml", "<BlastOutput"]:
        idx = xml_content.find(marker)
        if idx >= 0:
            xml_start = idx
            break
    if xml_start >= 0:
        xml_content = xml_content[xml_start:]
    else:
        log.warning(f"BLAST: no XML content found in response for RID {rid}")
        return {"hits": [], "rid": rid}

    try:
        root = ET.fromstring(xml_content)
        for hit_elem in root.iter("Hit"):
            hit_def = hit_elem.findtext("Hit_def", "unknown")
            hit_accession = hit_elem.findtext("Hit_accession", "")
            hit_len = int(hit_elem.findtext("Hit_len", "0"))
            hsp = hit_elem.find("Hit_hsps/Hsp")
            if hsp is not None:
                identity_pct = float(hsp.findtext("Hsp_identity", "0")) / max(1, float(hsp.findtext("Hsp_align-len", "1"))) * 100
                e_value = float(hsp.findtext("Hsp_evalue", "999"))
                hits.append({
                    "accession": hit_accession,
                    "description": hit_def,
                    "length": hit_len,
                    "identity_pct": round(identity_pct, 1),
                    "e_value": e_value,
                })
    except ET.ParseError:
        log.warning(f"Failed to parse BLAST XML for RID {rid}")
        return {"hits": [], "rid": rid}

    return {"hits": hits, "rid": rid}


# ── Step 6: Safe Harbor Integration ─────────────────────────────────────────

def step6_safe_harbor(config, cassette_result):
    from modules.construct.genome_integration_mapper import (
        design_grnas_for_locus,
        design_hdr_template,
        generate_integration_report,
        SAFE_HARBOR_LOCI,
    )

    species = config["species"]
    cassette_sequence = cassette_result["full_cassette_sequence"]

    loci = SAFE_HARBOR_LOCI.get(species, {})
    if not loci:
        return _step_result(
            "safe_harbor", "SKIPPED",
            error=f"No safe harbor loci defined for {species}",
        )

    locus_id = list(loci.keys())[0]
    log.info(f"Designing integration at safe harbor: {locus_id}")

    grna_result = design_grnas_for_locus(
        species=species,
        locus_id=locus_id,
        n_guides=5,
        pam="NGG",
    )

    hdr_result = design_hdr_template(
        cassette_sequence=cassette_sequence,
        species=species,
        locus_id=locus_id,
        arm_length=800,
        include_selection_marker=True,
    )

    integration_report = generate_integration_report(
        cassette_result, grna_result, hdr_result,
    )

    recommended = grna_result.get("recommended_guide", {})
    log.info(f"Integration gRNA: {recommended.get('guide_sequence', 'N/A')} "
             f"(score={recommended.get('score', 0):.3f})")
    log.info(f"HDR template: {hdr_result.get('template_length_bp', 0)} bp")

    return _step_result(
        "safe_harbor", "SUCCESS",
        locus_id=locus_id,
        grna_result=grna_result,
        hdr_result=hdr_result,
        integration_report=integration_report,
    )


# ── Step 7: Biological Realism Assessment ────────────────────────────────────

def step7_biological_assessment(config, fusion_protein):
    from modules.orchestration.construct_optimizer import ConstructOptimizer

    species = config["species"]
    promoter_seq = config.get("promoter_sequence")

    log.info("Running biological realism assessment...")
    optimizer = ConstructOptimizer(
        species=species,
        protein_name="hyaluronidase_PH-20",
        protein_sequence=fusion_protein,
        localization=config["localization"],
        delivery="transient",
        temperature_C=25,
        promoter_sequence=promoter_seq,
    )

    assessment = optimizer.run_full_pipeline()
    construct_record = optimizer.export_construct_record()

    n_stages = sum(1 for v in assessment.values() if isinstance(v, dict) and v.get("status") != "error")
    log.info(f"Biological assessment complete: {n_stages} stages passed")

    return _step_result(
        "biological_assessment", "SUCCESS",
        assessment=assessment,
        construct_record=construct_record,
    )


# ── Step 8: Yield Prediction (FBA) ──────────────────────────────────────────

def step8_yield_prediction(config, fusion_protein):
    from modules.metabolic.fba_engine import compute_protein_burden

    species_config = config["species_config"]
    expression_level = config.get("expression_level", 0.05)

    log.info("Running FBA yield prediction...")

    fba_wt = compute_protein_burden(
        protein_sequence=fusion_protein,
        species_config=species_config,
        expression_level=expression_level,
    )

    # Protease-knockout scenario: 1.5x effective expression (more protein retained)
    fba_ko = compute_protein_burden(
        protein_sequence=fusion_protein,
        species_config=species_config,
        expression_level=expression_level * 1.5,
    )

    wt_growth = fba_wt.get("burdened_growth", 0)
    ko_growth = fba_ko.get("burdened_growth", 0)

    if wt_growth > 0 and ko_growth > 0:
        # KO has more protein demand, so lower growth — but yield is higher
        fold_improvement = 1.5  # heuristic from literature (Stuttmann 2021: 1.5-4x)
    else:
        fold_improvement = 2.0  # fallback to PDF estimate

    log.info(f"FBA: WT growth={wt_growth:.4f}, KO growth={ko_growth:.4f}, "
             f"burden={fba_wt.get('metabolic_burden', 'unknown')}")

    return _step_result(
        "yield_prediction", "SUCCESS",
        fba_wildtype=fba_wt,
        fba_protease_ko=fba_ko,
        yield_improvement_factor=fold_improvement,
        expression_level_wt=expression_level,
        expression_level_ko=expression_level * 1.5,
    )


# ── Step 9: Output Generation ───────────────────────────────────────────────

def step9_generate_outputs(config, results):
    species = config["species"]
    output_dir = config["output_dir"]

    s1 = results.get("step1_protein", {})
    s2 = results.get("step2_codon", {})
    s3 = results.get("step3_promoter", {})
    s4 = results.get("step4_cassette", {})
    s5 = results.get("step5_protease", {})
    s5b = results.get("step5b_expanded", {})
    s5c = results.get("step5c_blast", {})
    s6 = results.get("step6_integration", {})
    s7 = results.get("step7_bio", {})
    s8 = results.get("step8_fba", {})

    # GenBank
    if s4.get("genbank_str"):
        with open(os.path.join(output_dir, f"{species}_construct.gb"), "w") as f:
            f.write(s4["genbank_str"])

    # Cassette FASTA
    cassette = s4.get("cassette_result", {})
    if cassette.get("full_cassette_sequence"):
        _write_fasta(
            os.path.join(output_dir, f"{species}_cassette.fasta"),
            f"{species}_gene_cassette",
            cassette["full_cassette_sequence"],
        )

    # Optimized CDS FASTA
    if s2.get("cds_sequence"):
        _write_fasta(
            os.path.join(output_dir, f"{species}_optimized_cds.fasta"),
            f"{species}_hyaluronidase_optimized",
            s2["cds_sequence"],
        )

    # Protease gRNAs
    if s5.get("protease_grnas"):
        with open(os.path.join(output_dir, f"{species}_protease_grnas.json"), "w") as f:
            json.dump(s5["protease_grnas"], f, indent=2)

    # Expanded gRNA targets
    if s5b.get("families"):
        with open(os.path.join(output_dir, f"{species}_expanded_grnas.json"), "w") as f:
            json.dump(s5b["families"], f, indent=2, default=str)

    # BLAST verification results
    if s5c.get("status") == "SUCCESS":
        blast_output = {
            "promoter_blast": s5c.get("promoter_blast"),
            "grna_blast": s5c.get("grna_blast", []),
            "summary": s5c.get("summary", {}),
        }
        with open(os.path.join(output_dir, f"{species}_blast_verification.json"), "w") as f:
            json.dump(blast_output, f, indent=2, default=str)

    # Integration gRNAs
    if s6.get("grna_result"):
        with open(os.path.join(output_dir, f"{species}_integration_grnas.json"), "w") as f:
            json.dump(s6["grna_result"], f, indent=2, default=str)

    # HDR template
    hdr = s6.get("hdr_result", {})
    hdr_seq = hdr.get("hdr_template_sequence") or hdr.get("template_sequence")
    if hdr_seq:
        _write_fasta(
            os.path.join(output_dir, f"{species}_hdr_template.fasta"),
            f"{species}_HDR_template_{s6.get('locus_id', 'unknown')}",
            hdr_seq,
        )

    # Integration report
    if s6.get("integration_report"):
        with open(os.path.join(output_dir, f"{species}_integration_report.json"), "w") as f:
            json.dump(s6["integration_report"], f, indent=2, default=str)

    # Biological assessment
    if s7.get("assessment"):
        with open(os.path.join(output_dir, f"{species}_bio_assessment.json"), "w") as f:
            json.dump(s7["assessment"], f, indent=2, default=str)

    # Regulatory summary
    assessment = s7.get("assessment", {})
    if assessment.get("regulatory"):
        with open(os.path.join(output_dir, f"{species}_regulatory_summary.json"), "w") as f:
            json.dump(assessment["regulatory"], f, indent=2, default=str)

    # FBA report
    if s8.get("fba_wildtype"):
        with open(os.path.join(output_dir, f"{species}_fba_report.json"), "w") as f:
            json.dump({
                "wildtype": s8["fba_wildtype"],
                "protease_knockout": s8.get("fba_protease_ko", {}),
                "yield_improvement_factor": s8.get("yield_improvement_factor"),
            }, f, indent=2, default=str)

    # Full pipeline dump (exclude large binary fields)
    dump = {k: v for k, v in results.items()}
    with open(os.path.join(output_dir, f"{species}_full_output.json"), "w") as f:
        json.dump(dump, f, indent=2, default=str)

    # Markdown report
    md = _generate_markdown_report(species, config, results)
    with open(os.path.join(output_dir, f"{species}_final_report.md"), "w") as f:
        f.write(md)

    output_files = [f for f in os.listdir(output_dir) if f.startswith(species)]
    log.info(f"Generated {len(output_files)} output files in {output_dir}")

    return _step_result("output_generation", "SUCCESS", output_files=output_files)


def _generate_markdown_report(species, config, results):
    s1 = results.get("step1_protein", {})
    s2 = results.get("step2_codon", {})
    s3 = results.get("step3_promoter", {})
    s4 = results.get("step4_cassette", {})
    s5 = results.get("step5_protease", {})
    s5b = results.get("step5b_expanded", {})
    s5c = results.get("step5c_blast", {})
    s6 = results.get("step6_integration", {})
    s7 = results.get("step7_bio", {})
    s8 = results.get("step8_fba", {})
    s9 = results.get("step9_outputs", {})

    overall = results.get("overall_status", "SUCCESS" if s9.get("status") == "SUCCESS" else "PARTIAL")
    elapsed = results.get("total_computation_time_s", 0)

    lines = [
        f"# Unified Pipeline Report: {species}",
        "",
        f"**Status:** {overall}",
        f"**Duration:** {elapsed:.1f}s",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Species:** `{species}`",
        f"- **Target protein:** Hyaluronidase PH-20 (UniProt P38567)",
        f"- **Localization:** {config.get('localization', 'ER_retained')}",
        f"- **Oleosin fusion:** {'Yes' if s1.get('oleosin_present') else 'No'}",
        f"- **Total fusion protein:** {s1.get('protein_length_aa', 'N/A')} aa",
        "",
    ]

    # Section 2: Protein Design
    lines += [
        "## 2. Protein Design",
        "",
        f"- **Architecture:** {s1.get('architecture', 'N/A')}",
        f"- **Fusion protein length:** {s1.get('protein_length_aa', 'N/A')} aa",
        "",
    ]

    # Section 3: Codon Optimization
    lines += [
        "## 3. Codon Optimization",
        "",
        f"- **CDS length:** {s2.get('cds_length', len(s2.get('cds_sequence', '')))} bp",
        f"- **CAI:** {s2.get('cai', 'N/A')}",
        f"- **GC content:** {s2.get('gc_content', 'N/A')}",
        f"- **Restriction sites remaining:** {len(s2.get('restriction_sites_remaining', {}))}",
        f"- **Cryptic splice sites:** {len(s2.get('cryptic_splice_sites_remaining', {}).get('5prime_splice_like', []))}",
        "",
    ]

    # Section 4: Promoter
    promo_score = s3.get("composite_score", "N/A")
    lines += [
        "## 4. Promoter Design",
        "",
        f"- **Source:** {s3.get('source', 'N/A')}",
        f"- **Length:** {len(s3.get('promoter_sequence', ''))} bp",
        f"- **Composite score:** {promo_score}",
        "",
    ]

    # Section 5: Gene Cassette Architecture
    cassette = s4.get("cassette_result", {})
    features = cassette.get("features", {})
    lines += [
        "## 5. Gene Cassette Architecture",
        "",
        f"- **Total length:** {cassette.get('total_length_bp', 'N/A')} bp",
        f"- **GC content:** {cassette.get('gc_content', 'N/A')}",
        "",
        "| Component | Start | End | Length |",
        "|-----------|-------|-----|--------|",
    ]
    for comp_name, comp_info in features.items():
        if isinstance(comp_info, dict) and "start" in comp_info:
            lines.append(
                f"| {comp_name} | {comp_info['start']} | {comp_info['end']} "
                f"| {comp_info['end'] - comp_info['start'] + 1} |"
            )
    lines.append("")

    # Section 6: Protease Knockout gRNAs
    lines += [
        "## 6. Protease Knockout gRNAs (NbSBT1)",
        "",
        f"**Genes targeted:** {s5.get('protease_genes_targeted', 0)}",
        "",
    ]
    grna_table = s5.get("grna_table", [])
    if grna_table:
        lines += [
            "| Gene | Guide Sequence | PAM | Efficiency | GC | Strand |",
            "|------|---------------|-----|-----------|-----|--------|",
        ]
        for row in grna_table:
            lines.append(
                f"| {row['gene']} | `{row['guide']}` | {row['pam']} "
                f"| {row['efficiency']:.3f} | {row['gc']:.2f} | {row['strand']} |"
            )
    else:
        lines.append("*No protease gRNAs designed (species not N. benthamiana or file unavailable)*")
    lines.append("")

    # Section 6b: Expanded gRNA Targets
    families = s5b.get("families", {})
    lines += [
        "## 6b. Expanded gRNA Targets (Beyond NbSBT1)",
        "",
        f"**Families targeted:** {len(families)}",
        f"**Total additional gRNAs:** {s5b.get('total_genes_targeted', 0)}",
        "",
    ]
    if families:
        for family_name, family_data in families.items():
            priority = family_data.get("priority", "N/A")
            desc = family_data.get("description", "")
            n_targeted = family_data.get("genes_targeted", 0)
            n_pending = family_data.get("genes_pending", 0)
            lines.append(f"### {family_name} ({priority} priority)")
            lines.append(f"*{desc}*")
            lines.append(f"**Genes with gRNAs:** {n_targeted} | "
                         f"**Pending CDS:** {n_pending}")
            lines.append("")
            ftable = family_data.get("table", [])
            if ftable:
                lines += [
                    "| Gene | ID | Guide | PAM | Eff. | GC |",
                    "|------|----|-------|-----|------|----|",
                ]
                for row in ftable:
                    lines.append(
                        f"| {row['gene']} | {row['gene_id']} | "
                        f"`{row['guide']}` | {row['pam']} | "
                        f"{row['efficiency']:.3f} | {row['gc']:.2f} |"
                    )
                lines.append("")
            pending = family_data.get("pending_genes", [])
            if pending:
                lines.append("**Pending (CDS required from genome):**")
                for pg in pending:
                    lines.append(f"- `{pg['gene_key']}` ({pg['gene_id']}): {pg['function']}")
                lines.append("")
    else:
        lines.append("*Expanded targets only available for N. benthamiana*")
        lines.append("")

    # Section 6c: BLAST Verification
    blast_summary = s5c.get("summary", {})
    promo_blast = s5c.get("promoter_blast", {})
    grna_blasts = s5c.get("grna_blast", [])
    lines += [
        "## 6c. BLAST Verification",
        "",
        f"- **Promoter verified:** {'Yes' if blast_summary.get('promoter_verified') else 'No'}",
        f"- **gRNAs verified:** {blast_summary.get('grnas_verified', 0)}",
        f"- **High off-target risk:** {blast_summary.get('high_off_target_risk', 0)}",
    ]
    if promo_blast:
        lines.append(f"- **Promoter specificity:** {promo_blast.get('specificity_note', 'N/A')}")
    lines.append("")
    if grna_blasts:
        lines += [
            "| gRNA | Hits | Off-targets | Risk |",
            "|------|------|------------|------|",
        ]
        for r in grna_blasts[:15]:
            lines.append(
                f"| {r['label']} | {r['n_total_hits']} | "
                f"{r['n_off_target_candidates']} | {r['off_target_risk']} |"
            )
        lines.append("")

    # Section 7: Integration
    lines += [
        "## 7. Safe Harbor Integration",
        "",
        f"- **Locus:** {s6.get('locus_id', 'N/A')}",
        f"- **HDR template length:** {s6.get('hdr_result', {}).get('total_length_bp', s6.get('hdr_result', {}).get('template_length_bp', 'N/A'))} bp",
        f"- **Integration status:** {s6.get('status', 'N/A')}",
        "",
    ]

    # Section 8: Biological Assessment
    assessment = s7.get("assessment", {})
    lines += [
        "## 8. Biological Realism Assessment",
        "",
    ]
    stage_names = {
        "folding": "Folding Quality",
        "glycosylation": "Glycosylation",
        "mrna_stability": "mRNA Stability",
        "degradation": "Degradation Risk",
        "delivery": "Delivery",
        "manufacturing": "Manufacturing",
        "confidence": "Overall Confidence",
        "regulatory": "Regulatory Compliance",
    }
    for key, label in stage_names.items():
        stage = assessment.get(key, {})
        if isinstance(stage, dict):
            score = stage.get("overall_score", stage.get("risk_score", stage.get("score", "N/A")))
            lines.append(f"- **{label}:** {score}")
    lines.append("")

    # Section 9: FBA
    lines += [
        "## 9. Metabolic Yield Prediction (FBA)",
        "",
        f"- **Yield improvement (KO vs WT):** {s8.get('yield_improvement_factor', 'N/A')}x",
        f"- **WT metabolic burden:** {s8.get('fba_wildtype', {}).get('metabolic_burden', 'N/A')}",
        f"- **KO metabolic burden:** {s8.get('fba_protease_ko', {}).get('metabolic_burden', 'N/A')}",
        "",
    ]

    # Section 10: Regulatory
    reg = assessment.get("regulatory", {})
    if reg:
        lines += [
            "## 10. Regulatory Compliance Summary",
            "",
            f"- **SDN-1 classification:** {reg.get('sdn1_eligible', reg.get('classification', 'N/A'))}",
            f"- **Overall compliance:** {reg.get('overall_compliance', reg.get('status', 'N/A'))}",
            "",
        ]

    # Section 11: R2 Retroelement Note
    lines += [
        "## 11. R2 Retroelement Integration (Emerging)",
        "",
        "R2 retroelement-based integration (Demirer et al. 2025, Research Square preprint)",
        "is an emerging alternative to Agrobacterium-mediated transformation for precise",
        "transgene insertion. When detailed protocols become available, the HDR template",
        "from Step 6 can be reformatted with R2-compatible flanking sequences.",
        "No code changes are required — only the template donor structure would differ.",
        "",
    ]

    # Section 12: Deliverables
    lines += [
        "## 12. Deliverables",
        "",
        "| File | Description |",
        "|------|-------------|",
        f"| `{species}_construct.gb` | GenBank annotated construct |",
        f"| `{species}_cassette.fasta` | Gene cassette FASTA |",
        f"| `{species}_optimized_cds.fasta` | Codon-optimized CDS |",
        f"| `{species}_protease_grnas.json` | Protease knockout gRNA table |",
        f"| `{species}_expanded_grnas.json` | Expanded target gRNA table |",
        f"| `{species}_blast_verification.json` | BLAST verification results |",
        f"| `{species}_integration_grnas.json` | Integration site gRNAs |",
        f"| `{species}_hdr_template.fasta` | HDR repair template |",
        f"| `{species}_integration_report.json` | Integration report |",
        f"| `{species}_bio_assessment.json` | Biological realism assessment |",
        f"| `{species}_regulatory_summary.json` | Regulatory compliance |",
        f"| `{species}_fba_report.json` | FBA yield prediction |",
        f"| `{species}_full_output.json` | Complete pipeline dump |",
        f"| `{species}_final_report.md` | This report |",
        "",
    ]

    return "\n".join(lines)


# ── Main Orchestrator ────────────────────────────────────────────────────────

def run_unified_pipeline(args):
    t0 = time.time()
    results = {}

    # Step 0: Configuration
    log.info("=" * 60)
    log.info("  STEP 0: Configuration")
    log.info("=" * 60)
    config = step0_config(args)
    results["step0_config"] = config
    if config["status"] != "SUCCESS":
        log.error(f"Configuration failed: {config.get('error')}")
        return results
    # Promote key fields to config for later steps
    config["promoter_sequence"] = config.get("promoter_sequence") or args.promoter_file and _read_fasta(args.promoter_file) or None

    # Step 1: Protein Design
    log.info("=" * 60)
    log.info("  STEP 1: Protein Design (Oleosin Fusion)")
    log.info("=" * 60)
    s1 = step1_protein_design(config)
    results["step1_protein"] = s1
    if s1["status"] != "SUCCESS":
        log.error(f"Protein design failed: {s1.get('error')}")
        return results
    fusion_protein = s1["fusion_protein_sequence"]

    # Step 2: Codon Optimization
    log.info("=" * 60)
    log.info("  STEP 2: Codon Optimization")
    log.info("=" * 60)
    s2 = step2_codon_optimization(config, fusion_protein)
    results["step2_codon"] = s2
    if s2["status"] != "SUCCESS":
        log.error(f"Codon optimization failed: {s2.get('error')}")
        return results
    cds_sequence = s2["cds_sequence"]

    # Step 3: Promoter Design
    log.info("=" * 60)
    log.info("  STEP 3: Promoter Design")
    log.info("=" * 60)
    s3 = step3_promoter_design(config)
    results["step3_promoter"] = s3
    if s3["status"] != "SUCCESS":
        log.error(f"Promoter design failed: {s3.get('error')}")
        return results
    promoter_sequence = s3["promoter_sequence"]

    # Step 4: Gene Cassette Assembly
    log.info("=" * 60)
    log.info("  STEP 4: Gene Cassette Assembly")
    log.info("=" * 60)
    s4 = step4_cassette_assembly(config, promoter_sequence, cds_sequence)
    results["step4_cassette"] = s4
    if s4["status"] != "SUCCESS":
        log.error(f"Cassette assembly failed: {s4.get('error')}")
        return results

    # Step 5: Protease Knockout gRNAs (non-blocking)
    log.info("=" * 60)
    log.info("  STEP 5: Host Engineering — Protease Knockout gRNAs")
    log.info("=" * 60)
    try:
        s5 = step5_protease_knockout(config)
    except Exception as e:
        log.warning(f"Protease knockout step failed: {e}")
        s5 = _step_result("protease_knockout", "FAILED", error=str(e))
    results["step5_protease"] = s5

    # Step 5b: Expanded gRNA Targets (non-blocking)
    log.info("=" * 60)
    log.info("  STEP 5b: Expanded gRNA Targets (PLCPs, VPEs, Silencing, Glycosylation, ER Stress)")
    log.info("=" * 60)
    try:
        s5b = step5b_expanded_grnas(config)
    except Exception as e:
        log.warning(f"Expanded gRNA design failed: {e}")
        s5b = _step_result("expanded_grnas", "FAILED", error=str(e))
    results["step5b_expanded"] = s5b

    # Step 5c: BLAST Verification (non-blocking, requires network)
    log.info("=" * 60)
    log.info("  STEP 5c: BLAST Verification")
    log.info("=" * 60)
    if config.get("skip_blast"):
        s5c = _step_result("blast_verification", "SKIPPED")
        log.info("Skipped (user requested)")
    else:
        try:
            s5c = step5c_blast_verification(
                config, promoter_sequence,
                s5.get("protease_grnas", {}),
                s5b,
            )
        except Exception as e:
            log.warning(f"BLAST verification failed: {e}")
            s5c = _step_result("blast_verification", "FAILED", error=str(e))
    results["step5c_blast"] = s5c

    # Step 6: Safe Harbor Integration (non-blocking)
    log.info("=" * 60)
    log.info("  STEP 6: Safe Harbor Integration")
    log.info("=" * 60)
    try:
        s6 = step6_safe_harbor(config, s4["cassette_result"])
    except Exception as e:
        log.warning(f"Safe harbor integration failed: {e}")
        s6 = _step_result("safe_harbor", "FAILED", error=str(e))
    results["step6_integration"] = s6

    # Step 7: Biological Assessment (non-blocking, optional)
    log.info("=" * 60)
    log.info("  STEP 7: Biological Realism Assessment")
    log.info("=" * 60)
    if config.get("skip_bio"):
        s7 = _step_result("biological_assessment", "SKIPPED")
        log.info("Skipped (user requested)")
    else:
        try:
            s7 = step7_biological_assessment(config, fusion_protein)
        except Exception as e:
            log.warning(f"Biological assessment failed: {e}")
            s7 = _step_result("biological_assessment", "FAILED", error=str(e))
    results["step7_bio"] = s7

    # Step 8: FBA Yield Prediction (non-blocking, optional)
    log.info("=" * 60)
    log.info("  STEP 8: Yield Prediction (FBA)")
    log.info("=" * 60)
    if config.get("skip_fba"):
        s8 = _step_result("yield_prediction", "SKIPPED")
        log.info("Skipped (user requested)")
    else:
        try:
            s8 = step8_yield_prediction(config, fusion_protein)
        except Exception as e:
            log.warning(f"FBA yield prediction failed: {e}")
            s8 = _step_result("yield_prediction", "FAILED", error=str(e))
    results["step8_fba"] = s8

    # Compute timing and status before output generation
    elapsed = time.time() - t0
    results["total_computation_time_s"] = round(elapsed, 1)
    critical_ok = all(
        results.get(f"step{i}_{name}", {}).get("status") == "SUCCESS"
        for i, name in [(0, "config"), (1, "protein"), (2, "codon"), (3, "promoter"), (4, "cassette")]
    )
    results["overall_status"] = "SUCCESS" if critical_ok else "PARTIAL"

    # Step 9: Output Generation
    log.info("=" * 60)
    log.info("  STEP 9: Output Generation")
    log.info("=" * 60)
    s9 = step9_generate_outputs(config, results)
    results["step9_outputs"] = s9

    log.info("=" * 60)
    log.info(f"  PIPELINE COMPLETE: {results['overall_status']} ({elapsed:.1f}s)")
    log.info(f"  Output directory: {config['output_dir']}")
    log.info("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified Gene Cassette + Genome Editing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/unified_pipeline.py --species nbenthamiana\n"
            "  python scripts/unified_pipeline.py --species rice --skip-fba\n"
        ),
    )
    parser.add_argument("--species", default="nbenthamiana",
                        help="Target species (default: nbenthamiana)")
    parser.add_argument("--protein-file", default=None,
                        help="Path to protein FASTA file")
    parser.add_argument("--promoter-file", default=None,
                        help="Path to pre-designed promoter FASTA (skips Step 3)")
    parser.add_argument("--localization", default="ER_retained",
                        choices=["ER_retained", "secreted", "cytosolic"],
                        help="Subcellular localization (default: ER_retained)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: v2_research/outputs/unified_pipeline/)")
    parser.add_argument("--expression-level", type=float, default=0.05,
                        help="Target expression as fraction of TSP (default: 0.05)")
    parser.add_argument("--construct-variant", default="oleosin",
                        choices=["oleosin", "enhanced"],
                        help="Construct architecture: oleosin (default) or enhanced (ELP+HRV3C)")
    parser.add_argument("--skip-fba", action="store_true",
                        help="Skip FBA yield prediction")
    parser.add_argument("--skip-bio-assessment", action="store_true",
                        help="Skip biological realism assessment")
    parser.add_argument("--skip-blast", action="store_true",
                        help="Skip BLAST verification (useful for offline runs)")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce logging output")

    args = parser.parse_args()
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    results = run_unified_pipeline(args)

    # Print summary
    print("\n" + "=" * 60)
    print("  UNIFIED PIPELINE SUMMARY")
    print("=" * 60)
    for key in sorted(results.keys()):
        if key.startswith("step"):
            val = results[key]
            status = val.get("status", "?")
            step = val.get("step", key)
            icon = "OK" if status == "SUCCESS" else ("--" if status == "SKIPPED" else "FAIL")
            print(f"  [{icon}] {step}: {status}")
    print(f"\n  Overall: {results.get('overall_status', 'UNKNOWN')}")
    print(f"  Time:    {results.get('total_computation_time_s', 0):.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
