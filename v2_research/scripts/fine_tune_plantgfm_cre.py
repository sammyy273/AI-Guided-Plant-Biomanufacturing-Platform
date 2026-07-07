#!/usr/bin/env python3
"""Fine-tune PlantGFM for CRE (cis-regulatory element) strength prediction.

Uses the PlantGFM base model (~350M params) from external/PlantGFM/ and
fine-tunes it on the CRE Strength Prediction dataset (regression task).

The training data (61K+ samples) is already available in the PlantGFM repo
at sample_data/CREs_Strength_Prediction/. Each sample is a space-separated
DNA sequence with a log-transformed expression strength label.

This script:
1. Loads and prepares the CRE strength data
2. Tokenizes using PlantGFM's single-nucleotide tokenizer
3. Fine-tunes PlantGFMForSequenceClassification (num_labels=1, regression)
4. Saves the fine-tuned checkpoint for use by plantgfm_predictor.py
5. Evaluates on held-out test set

USAGE:
    cd /home/boltzmann5/samitha/dna/promoter_design/v2_research
    python scripts/fine_tune_plantgfm_cre.py --epochs 15 --batch_size 32

REQUIRES:
    - PlantGFM cloned at external/PlantGFM/ (already done)
    - Base model at external/PlantGFM/models/plantgfm_base/ (already downloaded)
    - torch, transformers, datasets, scikit-learn, pandas
    - GPU strongly recommended (470MB model in fp16)
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PLANTGFM_DIR = PROJECT_DIR / "external" / "PlantGFM"
sys.path.insert(0, str(PLANTGFM_DIR))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from transformers import (
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Force single GPU — DataParallel on 2 GPUs doubles overhead; one 16GB card is enough
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class CREStrengthDataset(Dataset):
    """Dataset for CRE strength regression from PlantGFM CSV files."""

    def __init__(self, csv_path, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row["sequence"]
        label = float(row["labels"])

        encoding = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding.get("attention_mask", torch.ones_like(
                encoding["input_ids"].squeeze(0)
            )).squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float32),
        }


class PlantGFMTrainer(Trainer):
    """Custom Trainer that saves PlantGFM checkpoints via torch.save.

    PlantGFM's Hyena backbone has shared frequency tensors
    (implicit_filter.{1,3,5}.freq per layer) that HuggingFace's
    save_pretrained rejects as undeclared tied weights.
    Override _save to use torch.save directly.
    """

    def _save(self, output_dir=None, state_dict=None):
        from transformers.utils import WEIGHTS_NAME

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()

        # torch.save handles shared tensors fine; save_pretrained does not.
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        if hasattr(self.model, "config"):
            self.model.config.save_pretrained(output_dir)

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def compute_metrics(eval_pred):
    """Compute R² and MAE for regression evaluation."""
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # PlantGFMForSequenceClassification with num_labels=1 outputs shape (batch, 1)
    if predictions.ndim > 1:
        predictions = predictions.squeeze(-1)

    r2 = r2_score(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mean_squared_error(labels, predictions))

    # Pearson correlation
    corr = np.corrcoef(labels, predictions)[0, 1] if len(labels) > 1 else 0.0

    return {
        "r2": round(float(r2), 4),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "pearson": round(float(corr), 4),
    }


def prepare_promoter_test_data(tokenizer, max_length=512):
    """Create a small evaluation set from the project's scored promoters.

    Uses the cis-element weighted scores as ground truth for CRE strength,
    providing an in-domain validation beyond the generic PlantGFM test set.
    """
    scored_csv = PROJECT_DIR / "outputs" / "all_candidates_scored.csv"
    if not scored_csv.exists():
        return None

    df = pd.read_csv(str(scored_csv), index_col=0)
    candidates_fasta = PROJECT_DIR / "data" / "all_candidates.fasta"

    if not candidates_fasta.exists():
        return None

    # Parse FASTA
    sequences = {}
    current_id = None
    current_seq = []
    with open(str(candidates_fasta)) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper())
        if current_id:
            sequences[current_id] = "".join(current_seq)

    # Match sequences to scores
    records = []
    for cid, seq in sequences.items():
        if cid in df.index:
            score = df.loc[cid, "weighted_score"]
            # Normalize to log-space similar to PlantGFM CRE training labels
            # PlantGFM labels range ~[-6, 4], our scores range [0, ~100]
            # Map: log10(score+1) - 1 to center around 0
            label = np.log10(score + 1) - 1.0
            spaced = " ".join(seq)
            records.append({
                "sequence": spaced,
                "labels": label,
                "original_score": score,
            })

    if not records:
        return None

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune PlantGFM for CRE strength regression"
    )
    parser.add_argument(
        "--model_path", type=str,
        default=str(PLANTGFM_DIR / "models" / "plantgfm_base"),
        help="Path to PlantGFM base model checkpoint",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=str(PLANTGFM_DIR / "sample_data" / "CREs_Strength_Prediction"),
        help="Path to CRE strength prediction data directory",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(PROJECT_DIR / "models" / "plantgfm_cre_regression"),
        help="Output directory for fine-tuned model",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=172,
                        help="Max tokenization length (PlantGFM default is 172; 512 causes OOM)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("PlantGFM CRE Strength Regression Fine-Tuning")
    print("=" * 60)
    print(f"Base model:     {args.model_path}")
    print(f"Data dir:       {args.data_dir}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Learning rate:  {args.learning_rate}")
    print(f"Max length:     {args.max_length}")
    print()

    # Check prerequisites
    if not Path(args.model_path).exists():
        print(f"ERROR: Base model not found at {args.model_path}")
        print("Download it first or check the path.")
        sys.exit(1)

    train_csv = Path(args.data_dir) / "train.csv"
    val_csv = Path(args.data_dir) / "val.csv"
    test_csv = Path(args.data_dir) / "test.csv"

    for p in [train_csv, val_csv, test_csv]:
        if not p.exists():
            print(f"ERROR: Data file not found: {p}")
            sys.exit(1)

    for label, path in [("Train", train_csv), ("Val", val_csv), ("Test", test_csv)]:
        n = sum(1 for _ in open(path)) - 1
        print(f"  {label}: {n:,} samples")

    # Load tokenizer
    tokenizer_path = str(PLANTGFM_DIR / "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    # Load model
    print("Loading PlantGFM base model...")
    from plantgfm.modeling_plantgfm import PlantGFMForSequenceClassification

    model = PlantGFMForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=1,  # Regression: single output
        # Load in float32; the Trainer's fp16=True handles mixed-precision
        # autocasting. Loading directly in float16 causes a dtype mismatch
        # between the fp16 model output and the fp32 labels during loss
        # computation ("Found dtype Float but expected Half").
        torch_dtype=torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded: {n_params:,} params ({n_trainable:,} trainable)")
    print()

    # Create datasets
    print("Creating datasets...")
    train_dataset = CREStrengthDataset(train_csv, tokenizer, args.max_length)
    val_dataset = CREStrengthDataset(val_csv, tokenizer, args.max_length)
    test_dataset = CREStrengthDataset(test_csv, tokenizer, args.max_length)

    # Training arguments
    # Memory budget: ~10 GiB free on GPU 0 (other processes use ~5.5 GiB).
    # PlantGFM base is ~900 MB in fp32, bf16 halves that. With bf16, batch 8
    # × seq 172 fits in ~3 GiB peak. Gradient accumulation keeps effective
    # batch size at 16 (= 8 micro-batch × 2 accum steps).
    #
    # save_strategy="no" because PlantGFM's Hyena layers use shared frequency
    # tensors (implicit_filter.{1,3,5}.freq per layer) that HuggingFace's
    # save_pretrained rejects as undeclared tied weights. We save manually
    # after training with torch.save instead.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=max(1, args.batch_size // 2),
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=2,  # effective batch = (batch_size//2) × 2
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="no",  # Prevents tied-weights crash during checkpointing
        load_best_model_at_end=False,  # Incompatible with save_strategy="no"
        # Use bf16 if available (RTX 5060 Ti supports it), else fall back to
        # full fp32. Avoid fp16 — PlantGFM's Hyena layers have custom ops that
        # cause "Found dtype Float but expected Half" during backward with fp16
        # + gradient_checkpointing.
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        gradient_checkpointing=False,
        optim="adamw_torch_fused",  # slightly less memory than default adamw
        seed=args.seed,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        report_to="none",
    )

    # Trainer — uses PlantGFMTrainer to avoid tied-weights crash on save
    trainer = PlantGFMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting fine-tuning...")
    print("-" * 60)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nFine-tuning completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"  R²:       {test_results.get('r2', 'N/A')}")
    print(f"  MAE:      {test_results.get('mae', 'N/A')}")
    print(f"  RMSE:     {test_results.get('rmse', 'N/A')}")
    print(f"  Pearson:  {test_results.get('pearson', 'N/A')}")

    # Save model — use torch.save directly because PlantGFM's Hyena layers have
    # shared frequency tensors (implicit_filter.{1,3,5}.freq) that HuggingFace's
    # save_pretrained rejects as undeclared tied weights. torch.save handles them
    # fine since it serializes the raw state_dict.
    print("\nSaving fine-tuned model...")
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "pytorch_model.bin")
    torch.save(trainer.model.state_dict(), model_path)
    # Also save config and tokenizer for pipeline loading
    trainer.model.config.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Saved to: {args.output_dir}")
    print(f"  Model size: {model_size_mb:.0f} MB")

    # Evaluate on project promoter data
    promoter_test = prepare_promoter_test_data(tokenizer, args.max_length)
    if promoter_test is not None:
        print(f"\nEvaluating on project promoters ({len(promoter_test)} sequences)...")
        promo_dataset = CREStrengthDataset.__new__(CREStrengthDataset)
        promo_dataset.data = promoter_test
        promo_dataset.tokenizer = tokenizer
        promo_dataset.max_length = args.max_length

        promo_results = trainer.evaluate(eval_dataset=promo_dataset)
        print(f"  R²:       {promo_results.get('r2', 'N/A')}")
        print(f"  MAE:      {promo_results.get('mae', 'N/A')}")

    # Save training report
    report = {
        "model_path": args.model_path,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "test_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                        for k, v in test_results.items()},
        "training_time_seconds": round(elapsed, 1),
    }
    report_path = Path(args.output_dir) / "training_report.json"
    with open(str(report_path), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nTraining report saved to: {report_path}")


if __name__ == "__main__":
    main()
