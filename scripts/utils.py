# =============================================================================
# utils.py
# Shared utilities: coloured logging, FASTA read/write.
# =============================================================================

import os
import logging
from colorama import Fore, Style, init

init(autoreset=True)   # Required for colour support on Windows


def setup_logger(log_path: str) -> logging.Logger:
    """Logger that writes coloured output to terminal and plain text to file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("promoter_design")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Terminal — INFO and above, coloured
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

        # File — DEBUG and above, plain text with timestamp
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
        logger.addHandler(fh)

    return logger


def log_step(logger, n: int, title: str):
    line = "=" * 62
    logger.info(f"\n{Fore.CYAN}{line}\n  STEP {n}: {title}\n{line}{Style.RESET_ALL}")

def log_ok(logger, msg):   logger.info(f"{Fore.GREEN}  OK  {Style.RESET_ALL} {msg}")
def log_warn(logger, msg): logger.info(f"{Fore.YELLOW}  WARN{Style.RESET_ALL} {msg}")
def log_err(logger, msg):  logger.info(f"{Fore.RED}  ERR {Style.RESET_ALL} {msg}")
def log_info(logger, msg): logger.info(f"       {msg}")


def save_fasta(sequences: dict, filepath: str):
    """Save {name: sequence} dict to a FASTA file (60 chars per line)."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for name, seq in sequences.items():
            f.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")
            f.write("\n")


def load_fasta(filepath: str) -> dict:
    """Load a FASTA file into a {name: sequence} dict."""
    seqs, cur_name, cur_seq = {}, None, []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur_name:
                    seqs[cur_name] = "".join(cur_seq)
                cur_name, cur_seq = line[1:], []
            elif line:
                cur_seq.append(line.upper())
    if cur_name:
        seqs[cur_name] = "".join(cur_seq)
    return seqs
