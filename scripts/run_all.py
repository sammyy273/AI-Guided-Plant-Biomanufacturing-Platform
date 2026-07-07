# =============================================================================
# run_all.py
# Runs the complete pipeline in one command.
#
# Usage:
#   python run_all.py
#
# Or run each step individually (recommended for first run):
#   python fetch_references.py
#   python score_references.py
#   python generate_candidates.py
#   python filter_and_rank.py
# =============================================================================

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import setup_logger, log_info

def main():
    logger = setup_logger(config.LOG_FILE)
    logger.info("\n" + "=" * 62)
    logger.info("  PROMOTER DESIGN PIPELINE — FULL RUN")
    logger.info("=" * 62)

    from fetch_references import main as step1
    from score_references import main as step2
    from generate_candidates import main as step3
    from filter_and_rank import main as step4

    step1()
    step2()
    step3()
    step4()

    logger.info("\n" + "=" * 62)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 62)
    logger.info(f"\n  Top 3 candidates:   {config.TOP3_FASTA}")
    logger.info(f"  Full ranking:       {config.RANKING_CSV}")
    logger.info(f"  All scored:         {config.ALL_SCORED_CSV}")
    logger.info(f"  Full log:           {config.LOG_FILE}")
    logger.info("\n  Take top3_candidates.fasta to PlantCARE for final validation.")

if __name__ == "__main__":
    main()
