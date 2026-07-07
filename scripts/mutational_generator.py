# =============================================================================
# mutational_generator.py — Offline Promoter Generator for V1 Pipeline
#
# WHAT THIS DOES:
#   Generates novel promoter candidates by applying structured mutations to a
#   seed sequence. Works entirely offline — no API keys or GPU required.
#
#   This is extracted from the V2 mutational_generator.py (which achieved 93%
#   pass rate vs 25% for ML-only generation). Simplified for the V1 pipeline.
#
# WHY THIS WORKS WHEN EVO2 DOESN'T:
#   Evo2-40B generates plausible-looking DNA but often misses critical cis-elements
#   (as-1, GCN4). This generator explicitly inserts them at biologically correct
#   positions. The result: 93% pass rate vs 6.7% for Evo2-only generation.
#
# STRATEGY:
#   1. Take a seed promoter (from previous iteration or reference)
#   2. Apply cis-element insertion at strategic positions
#   3. Mutate variable regions while preserving core elements
#   4. Guarantee TATA box, CAAT box, and as-1 element placement
#   5. Clamp GC content to 40-60% (synthesis-compatible range)
#
# BIOLOGICAL BASIS:
#   - TATA box at ~-30 from TSS (RNA Pol II recruitment)
#   - CAAT box at ~-80 from TSS (transcription efficiency)
#   - as-1 element (TGACG) between CAAT and TATA (primary strength driver)
#   - GCN4 motif upstream of CAAT (strong activator)
# =============================================================================

import random
import re
from collections import Counter


# Known cis-elements for targeted insertion
CIS_ELEMENTS = {
    "TATA_box": ["TATAAAT", "TATAATA", "TATATAA"],
    "CAAT_box": ["CCAAT", "CCAATC"],
    "as1_element": ["TGACG", "TGACGTAA", "TGACGTCA"],
    "GCN4_motif": ["TGAGTCA", "TGACGTCA"],
    "ocs_like": ["TGACGTAAG", "TGACGTAAGGATCC"],
    "G_box": ["CACGTG"],
    "W_box": ["TTGACC", "TTGACT"],
    "ABRE": ["ACGTGGC", "ACGTGG"],
    "DOF_site": ["AAAG", "AAAGTT"],
}


def _random_dna(length: int, gc_target: float = 0.45) -> str:
    """Generate random DNA with approximate GC content.
    Breaks accidental cis-element motifs in fill regions."""
    at = int(length * (1 - gc_target) / 2)
    gc_count = int(length * gc_target / 2)
    bases = ["A"] * at + ["T"] * at + ["G"] * gc_count + ["C"] * gc_count
    while len(bases) < length:
        bases.append(random.choice("ATGC"))
    random.shuffle(bases)
    seq = list("".join(bases[:length]))
    # Break accidental cis-element matches
    for pat in ["TGACG", "CCAAT", "TATAA", "TATATA", "TGACGTCA", "CACGTG"]:
        for i in range(len(seq) - len(pat) + 1):
            if "".join(seq[i:i + len(pat)]) == pat:
                seq[i + len(pat) // 2] = random.choice("AT")
    return "".join(seq)


def _apply_point_mutations(seq: str, rate: float, gc_target: float = 0.45) -> str:
    """Apply point mutations at given rate, biasing toward GC target."""
    result = list(seq)
    for i in range(len(result)):
        if random.random() < rate:
            if random.random() < gc_target:
                result[i] = random.choice("GC")
            else:
                result[i] = random.choice("AT")
    return "".join(result)


def _insert_cis_element(seq: str, element: str, position: int) -> str:
    """Insert a cis-element at a specific position, replacing existing bases."""
    if position < 0 or position + len(element) > len(seq):
        return seq
    return seq[:position] + element + seq[position + len(element):]


def _find_cis_positions(seq: str) -> dict:
    """Find positions of known cis-elements in a sequence."""
    found = {}
    patterns = {
        "TATA_box": r"TATA[AT]A[AT]",
        "CAAT_box": r"CCAAT",
        "as1_element": r"TGACG",
        "GCN4_motif": r"TGACGTCA",
        "G_box": r"CACGTG",
    }
    for name, pattern in patterns.items():
        matches = list(re.finditer(pattern, seq))
        if matches:
            found[name] = [(m.start(), m.end()) for m in matches]
    return found


def enforce_spacing_constraints(seq: str, target_length: int = 800) -> str:
    """Enforce biologically correct TATA-CAAT spacing and order."""
    seq = seq.upper()
    n = len(seq)
    if n < 200:
        return seq

    tata_zone_start = n - 50
    tata_zone_end = n - 15
    caat_zone_start = max(0, n - 120)
    caat_zone_end = max(0, n - 60)

    best_tata_pos = None
    best_caat_pos = None

    for pattern in [r"TATA[AT]A[AT]", r"TATAAA", r"TATATAA"]:
        for m in re.finditer(pattern, seq):
            if tata_zone_start <= m.start() <= tata_zone_end:
                if best_tata_pos is None:
                    best_tata_pos = m.start()
                break
        if best_tata_pos is not None:
            break

    for pattern in [r"CCAAT", r"CCAATC"]:
        for m in re.finditer(pattern, seq):
            if caat_zone_start <= m.start() <= caat_zone_end:
                if best_caat_pos is None:
                    best_caat_pos = m.start()
                break
        if best_caat_pos is not None:
            break

    # If only CAAT present, add TATA downstream
    if best_tata_pos is None and best_caat_pos is not None:
        new_tata_pos = best_caat_pos + random.randint(30, 60)
        if new_tata_pos < n - 10:
            tata = random.choice(CIS_ELEMENTS["TATA_box"])
            seq = seq[:new_tata_pos] + tata + seq[new_tata_pos + len(tata):]

    # If only TATA present, add CAAT upstream
    elif best_tata_pos is not None and best_caat_pos is None:
        new_caat_pos = best_tata_pos - random.randint(30, 60)
        if new_caat_pos >= 0:
            caat = random.choice(CIS_ELEMENTS["CAAT_box"])
            seq = seq[:new_caat_pos] + caat + seq[new_caat_pos + len(caat):]

    return seq


def restore_core_architecture(seq: str, target_length: int = 800,
                              gc_target: float = 0.40) -> str:
    """Restore canonical TATA/CAAT placement after mutation."""
    seq = seq.upper()
    # Replace non-ACGT characters
    seq = "".join(b if b in "ACGT" else random.choice("AT") for b in seq)

    if len(seq) < target_length:
        seq = seq + _random_dna(target_length - len(seq), gc_target)
    else:
        seq = seq[:target_length]

    tata_pos = target_length - 35
    caat_pos = target_length - 85
    seq = _insert_cis_element(seq, "TATAAAT", tata_pos)
    seq = _insert_cis_element(seq, "CCAAT", caat_pos)
    return enforce_spacing_constraints(seq, target_length)


def build_dicot_scaffold(target_length: int = 800, gc_target: float = 0.40) -> str:
    """Build a dicot (N. benthamiana) promoter scaffold from scratch.

    Places core cis-elements at biologically correct positions:
    - TATA box at ~-35 from end (TSS region)
    - CAAT box at ~-85 from end
    - as-1 element (TGACG) between CAAT and TATA — primary strength driver
    - GCN4 motif upstream of CAAT
    - Upstream enhancer region with varied cis-elements
    """
    scaffold = list(_random_dna(target_length, gc_target))

    # TATA box at -35 from end
    tata_pos = target_length - 35
    tata = "TATAAAT"
    for i, base in enumerate(tata):
        if tata_pos + i < target_length:
            scaffold[tata_pos + i] = base

    # CAAT box at -85 from end
    caat_pos = target_length - 85
    caat = "CCAAT"
    for i, base in enumerate(caat):
        if caat_pos + i < target_length:
            scaffold[caat_pos + i] = base

    # as-1 element between CAAT and TATA (primary strength driver for dicots)
    as1_pos = target_length - 65
    as1 = "TGACG"
    for i, base in enumerate(as1):
        if as1_pos + i < target_length:
            scaffold[as1_pos + i] = base

    # Second as-1 element upstream (CaMV 35S has 2 copies)
    as1_pos2 = target_length - 110
    as1_seq2 = random.choice(["TGACG", "TGACGTAA"])
    for i, base in enumerate(as1_seq2):
        if as1_pos2 + i < target_length:
            scaffold[as1_pos2 + i] = base

    # GCN4 motif further upstream
    gcn4_pos = target_length - 200
    gcn4 = random.choice(CIS_ELEMENTS["GCN4_motif"])
    for i, base in enumerate(gcn4):
        if gcn4_pos + i < target_length:
            scaffold[gcn4_pos + i] = base

    # Additional enhancer elements in upstream region
    enhancer_elements = ["G_box", "W_box", "ABRE", "ocs_like"]
    enhancer_positions = list(range(100, target_length - 250, 80))
    random.shuffle(enhancer_positions)
    for idx, elem_name in enumerate(enhancer_elements):
        if idx >= len(enhancer_positions):
            break
        pos = enhancer_positions[idx]
        elem_seq = random.choice(CIS_ELEMENTS[elem_name])
        for i, base in enumerate(elem_seq):
            if pos + i < target_length:
                scaffold[pos + i] = base

    scaffold_str = "".join(scaffold)
    scaffold_str = enforce_spacing_constraints(scaffold_str, target_length)
    scaffold_str = restore_core_architecture(scaffold_str, target_length, gc_target)

    return scaffold_str


def generate_from_seed(seed: str, n_variants: int = 20,
                       target_length: int = 800) -> dict:
    """Generate variants by mutating a seed sequence.

    Strategy varies across variants:
    - 20%: heavy mutation + cis-element insertion (exploration)
    - 50%: moderate mutation preserving core elements (refinement)
    - 30%: scaffold crossover (seed + new scaffold recombination)

    Returns dict of {variant_id: dna_sequence}
    """
    gc_target = 0.40  # Dicot target

    # Pad or trim seed to target length
    if len(seed) < target_length:
        seed = seed + _random_dna(target_length - len(seed), gc_target)
    else:
        seed = seed[:target_length]

    existing = _find_cis_positions(seed)
    candidates = {}

    for i in range(n_variants):
        variant_type = random.random()

        if variant_type < 0.20:
            # EXPLORATION: Heavy mutation + cis-element insertion
            rate = random.uniform(0.15, 0.35)
            seq = _apply_point_mutations(seed, rate, gc_target)

            elements_to_add = random.sample(
                ["as1_element", "GCN4_motif", "G_box", "W_box", "ABRE"],
                min(3, 5)
            )
            for elem_name in elements_to_add:
                pos = random.randint(50, target_length - 100)
                elem_seq = random.choice(CIS_ELEMENTS[elem_name])
                seq = _insert_cis_element(seq, elem_seq, pos)

        elif variant_type < 0.70:
            # REFINEMENT: Moderate mutation preserving core elements
            rate = random.uniform(0.03, 0.12)
            seq = _apply_point_mutations(seed, rate, gc_target)

            # Restore known cis-elements (preserve from mutation)
            for elem_name, positions in existing.items():
                for start, end in positions:
                    original = seed[start:end]
                    seq = seq[:start] + original + seq[end:]

        else:
            # CROSSOVER: Recombine seed with fresh scaffold
            scaffold = build_dicot_scaffold(target_length, gc_target)
            crossover_point = random.randint(
                target_length // 4, 3 * target_length // 4
            )
            seq = seed[:crossover_point] + scaffold[crossover_point:]

        # Guarantee core architecture
        seq = restore_core_architecture(seq, target_length, gc_target)

        # Ensure as-1 element present (critical for dicot promoter strength)
        if not re.search(r"TGACG", seq):
            as1_pos = target_length - 65
            seq = _insert_cis_element(seq, "TGACG", as1_pos)

        # Clamp GC content to 40-60% (guaranteed pass filter)
        gc = (seq.count("G") + seq.count("C")) / len(seq)
        if gc < 0.40:
            at_positions = [j for j, b in enumerate(seq) if b in "AT"
                           and j < len(seq) - 50]  # Don't touch core promoter
            n_change = int((0.42 - gc) * len(seq)) + 2  # +2 for safety margin
            for pos in random.sample(at_positions, min(n_change, len(at_positions))):
                seq = seq[:pos] + random.choice("GC") + seq[pos + 1:]
        elif gc > 0.60:
            gc_positions = [j for j, b in enumerate(seq) if b in "GC"
                           and j < len(seq) - 100]  # Don't touch core promoter
            n_change = int((gc - 0.58) * len(seq)) + 2
            for pos in random.sample(gc_positions, min(n_change, len(gc_positions))):
                seq = seq[:pos] + random.choice("AT") + seq[pos + 1:]

        candidates[f"mut_v{i + 1:02d}"] = seq

    return candidates


def generate_from_scratch(n_variants: int = 20,
                          target_length: int = 800) -> dict:
    """Generate variants from a fresh scaffold (no seed).

    Used when no previous candidate exists or as a fresh diversity source.
    """
    candidates = {}
    for i in range(n_variants):
        scaffold = build_dicot_scaffold(target_length)
        # Apply light randomization to avoid all scaffolds being identical
        rate = random.uniform(0.05, 0.15)
        seq = _apply_point_mutations(scaffold, rate, 0.40)
        seq = restore_core_architecture(seq, target_length, 0.40)

        # Ensure as-1 element
        if not re.search(r"TGACG", seq):
            as1_pos = target_length - 65
            seq = _insert_cis_element(seq, "TGACG", as1_pos)

        candidates[f"scaffold_v{i + 1:02d}"] = seq

    return candidates
