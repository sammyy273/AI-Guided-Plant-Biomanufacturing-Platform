# =============================================================================
# config.py
# All settings for the promoter design pipeline.
# Copy .env.example to .env and fill in your credentials before running.
# =============================================================================

import os
from pathlib import Path

# ── YOUR CREDENTIALS ──────────────────────────────────────────────────────────
# Load from environment / .env file. Never hardcode keys.
_dotenv = Path(__file__).resolve().parent.parent / ".env"
if _dotenv.exists():
    for line in _dotenv.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "")

# Validate key format — catch accidentally committed real keys
if NVIDIA_API_KEY and not NVIDIA_API_KEY.startswith("nvapi-"):
    print(f"WARNING: NVIDIA_API_KEY does not start with 'nvapi-'. "
          f"Check .env file. Key prefix: '{NVIDIA_API_KEY[:8]}...'")
if NVIDIA_API_KEY and len(NVIDIA_API_KEY) < 20:
    print("WARNING: NVIDIA_API_KEY appears too short. "
          "Copy the full key from build.nvidia.com")

# ── EVO2 SETTINGS ─────────────────────────────────────────────────────────────
# "evo2-7b"  → faster, good for testing your key works
# "evo2-40b" → better quality sequences, use for real runs
EVO2_MODEL       = "evo2-40b"  # evo2-7b produces degenerate sequences
EVO2_N_VARIANTS  = 20       # Variants per seed (20 × 3 seeds = 60 total)
EVO2_LENGTH      = 800      # Generated promoter length in nucleotides
EVO2_TEMPERATURE = 0.7      # 0.5 = conservative, 0.9 = diverse
EVO2_TOP_K       = 4           # MUST be <= 6 (API hard limit)
EVO2_TOP_P       = 0.9

# N. benthamiana taxonomy prefix — tells Evo2 to generate plant-like sequences
EVO2_TAXONOMY = (
    "|k__Viridiplantae;p__Streptophyta;c__Magnoliopsida;"
    "o__Solanales;g__Nicotiana;s__Nicotiana benthamiana|"
)

# ── FILTER THRESHOLDS ─────────────────────────────────────────────────────────
# Hard filters — candidates failing these are discarded
MIN_GC_PCT   = 40.0   # Below this = synthesis problems and mRNA instability
MAX_GC_PCT   = 60.0   # Above this = secondary structure risk
REQUIRE_TATA = False  # Soft: TATA presence contributes to score, not a hard gate
REQUIRE_CAAT = False  # Soft: CAAT presence contributes to score, not a hard gate

# NOTE: G-box (CACGTG) is NOT a hard filter.
# Validation showed the 2×CaMV 35S reference (our gold standard) has 0 G-boxes.
# CaMV 35S is driven by as-1 elements (TGACG), not G-boxes.
# G-box presence improves the score but does not disqualify a candidate.

# ── OUTPUT PATHS ──────────────────────────────────────────────────────────────
# All paths resolved relative to the project root (parent of scripts/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR      = str(_PROJECT_ROOT / "outputs")
LOG_FILE        = str(_PROJECT_ROOT / "logs" / "run.log")
REF_FASTA       = str(_PROJECT_ROOT / "data" / "reference_promoters.fasta")
REF_SCORES_CSV  = str(_PROJECT_ROOT / "data" / "reference_element_scores.csv")
CANDIDATES_FASTA= str(_PROJECT_ROOT / "data" / "all_candidates.fasta")
ALL_SCORED_CSV  = str(_PROJECT_ROOT / "outputs" / "all_candidates_scored.csv")
TOP3_FASTA      = str(_PROJECT_ROOT / "outputs" / "top3_candidates.fasta")
RANKING_CSV     = str(_PROJECT_ROOT / "outputs" / "ranking_table.csv")
