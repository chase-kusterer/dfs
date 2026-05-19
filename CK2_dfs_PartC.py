#!/usr/bin/env python
# coding: utf-8
"""
MSDS458 Assignment 3 — Part C: Fine-tuning Pretrained Models for Text-to-Code
                                Generation — **Loads from Part B checkpoint**

This version skips retraining. It:
  1. Loads the tokenizer and base architecture the same way Part B does
     (AutoTokenizer, not RobertaTokenizer).
  2. Restores the fine-tuned weights from outputs/codet5_finetuned_mbpp.pt.
  3. Pulls generation hyperparameters (num_beams, max_new_tokens,
     no_repeat_ngram_size) from the Optuna DB written by Part B.
  4. Evaluates the restored model on the test set and compares against
     Part B's zero-shot baseline.
"""

# ---------------------------------------------------------------------------
# Versioning / timing
# ---------------------------------------------------------------------------
NOTEBOOK_VERSION = "2.0-from-partB"
QUARTER = "Spring 2026"

from datetime import datetime
NOTEBOOK_START_TIME = datetime.now()
print(f"Notebook Version: {NOTEBOOK_VERSION} | {QUARTER}")


# ===========================================================================
# ## Step 1: Setup and Imports
# ===========================================================================
import os, sys, random, copy
import numpy as np
import pandas as pd
import evaluate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, RobertaTokenizer, get_linear_schedule_with_warmup

from datasets import load_dataset

# Reproducibility — use same seed as Part B
SEED = 702
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Match Part B's environment flags
hf_token = os.getenv("HF_TOKEN") or None
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]  = "1"
os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"

os.makedirs("outputs", exist_ok=True)
print("✅ Libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")


# ===========================================================================
# ## Step 1B: Import Shared Utilities
# ===========================================================================
from dfs_utils_torch import (
    prepare_training_data,
    add_python_hint,
    compute_bleu_for_code,
    evaluate_model_with_bleu,
    analyze_bleu_results,
    batch_generate_codes,
)
print("✅ Utilities imported successfully")


# ===========================================================================
# ## Step 2: Load and Process MBPP Dataset
# ===========================================================================
print("\nLoading and processing MBPP dataset...")
print("=" * 60)

data = prepare_training_data("MBPP")
train_prompts = data["train"]["prompts"]
train_code    = data["train"]["code"]
val_prompts   = data["val"]["prompts"]   if "val"  in data else train_prompts[:50]
val_code      = data["val"]["code"]      if "val"  in data else train_code[:50]
test_prompts  = data["test"]["prompts"]
test_code     = data["test"]["code"]

print("=" * 60)
print(f"📊 Dataset sizes:")
print(f"  Train: {len(train_prompts)}")
print(f"  Val:   {len(val_prompts)}")
print(f"  Test:  {len(test_prompts)}")
print("✅ Dataset loaded and processed successfully!")


# ===========================================================================
# ## Step 3: Load Model Architecture + Tokenizer (matching Part B exactly)
# ===========================================================================
MODEL_NAME = "Salesforce/codet5p-220m-py"

print(f"\nLoading model architecture: {MODEL_NAME}")
#print("Using AutoTokenizer (same as Part B) — not RobertaTokenizer.")

# Part B uses AutoTokenizer with use_fast=True; mirror that here so
# tokenized sequences are identical between training and inference.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token    = hf_token,
    use_fast = True,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    token = hf_token,
).to(DEVICE)

print(f"✅ Base architecture loaded on {DEVICE}")
print(f"   Tokenizer vocab size: {tokenizer.vocab_size:,}")
print(f"   Model parameters:     {sum(p.numel() for p in model.parameters()):,}")


# ===========================================================================
# ## Step 4: Restore Fine-Tuned Weights from Part B Checkpoint
# ===========================================================================
# ## 🤖 Step 4: Load Fine-Tuned CodeT5+ Model from .pt file

MODEL_NAME = "Salesforce/codet5p-220m-py"
PT_MODEL_PATH = "outputs_PartB_backup/codet5_finetuned_mbpp.pt"  # <-- path to your .pt file

print(f"Loading architecture: {MODEL_NAME}")
print(f"Loading fine-tuned weights from: {PT_MODEL_PATH}")

# Tokenizer stays exactly the same
tokenizer = RobertaTokenizer.from_pretrained(
    "roberta-base",
    use_fast=False,
    add_prefix_space=True
)
tokenizer.add_tokens(["<pad>", "<s>", "</s>", "<unk>", "<mask>"], special_tokens=True)

# Load the architecture shell (no pretrained weights needed)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    use_safetensors=False
)

# Load your fine-tuned weights on top
state_dict = torch.load(PT_MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()  # Set to eval mode since we're running inference, not training

print(f"\n✅ Fine-tuned model loaded from {PT_MODEL_PATH}!")
print(f"Model type: {type(model).__name__}")
print(f"Tokenizer vocab size: {tokenizer.vocab_size:,}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# ===========================================================================
# ## Step 5: Recover Best Generation Hyperparameters from Part B Optuna DB
#
# Part B stores its Optuna study in optuna_partB.db.  We pull the best
# trial's params so generation here uses the exact same settings that
# produced the Part B fine-tuned result.
# ===========================================================================
OPTUNA_DB   = "sqlite:///optuna_partB.db"
STUDY_NAME  = "partB_codet5_finetune"

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=STUDY_NAME, storage=OPTUNA_DB)
    best_params = study.best_trial.params
    print(f"\n✅ Loaded best Optuna params from Part B (Trial {study.best_trial.number}):")
    for k, v in best_params.items():
        print(f"   {k}: {v}")
except Exception as e:
    # Fall back to sensible defaults if the DB is unavailable
    print(f"\n⚠️  Could not load Optuna DB ({e}).")
    print("   Using fallback generation hyperparameters.")
    best_params = {
        "num_beams":            4,
        "max_new_tokens":       128,
        "no_repeat_ngram_size": 3,
        "max_input_length":     128,
        "max_target_length":    256,
        "batch_size":           8,
        "learning_rate":        5e-5,
        "warmup_ratio":         0.1,
        "weight_decay":         0.01,
    }


# ===========================================================================
# ## Step 6: Generation Helper (matches Part B's generate_code function)
# ===========================================================================
@torch.no_grad()
def generate_code(prompt, gen_model=None, num_beams=None, max_new_tokens=None,
                  no_repeat_ngram_size=None, max_input_length=128):
    """Generate Python code from a natural-language prompt."""
    if gen_model is None:
        gen_model = model
    if num_beams            is None: num_beams            = best_params["num_beams"]
    if max_new_tokens       is None: max_new_tokens       = best_params["max_new_tokens"]
    if no_repeat_ngram_size is None: no_repeat_ngram_size = best_params["no_repeat_ngram_size"]

    inputs = tokenizer(
        add_python_hint(prompt),
        return_tensors = "pt",
        truncation     = True,
        max_length     = max_input_length,
    ).to(DEVICE)

    outputs = gen_model.generate(
        inputs["input_ids"],
        attention_mask       = inputs.get("attention_mask"),
        max_new_tokens       = max_new_tokens,
        num_beams            = num_beams,
        early_stopping       = True,
        no_repeat_ngram_size = no_repeat_ngram_size,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ===========================================================================
# ## Step 7: Evaluate the Fine-Tuned Model on the Test Set
# ===========================================================================
bleu_metric = evaluate.load("bleu")
test_prompts_python = [add_python_hint(p) for p in test_prompts]

print("\n⚡ Generating test codes with Part B fine-tuned model...")
finetuned_generated = [
    generate_code(p) for p in tqdm(test_prompts_python, desc="Generating")
]

finetuned_bleu_scores, finetuned_syntax_valid, finetuned_codes, finetuned_stats = evaluate_model_with_bleu(
    test_prompts_python,
    test_code,
    None,
    bleu_metric,
    model_label     = "Fine-Tuned CodeT5+ (Part B checkpoint)",
    generated_codes = finetuned_generated,
)

analyze_bleu_results(
    finetuned_bleu_scores,
    finetuned_syntax_valid,
    finetuned_codes,
    test_prompts_python,
    test_code,
    finetuned_stats,
    model_label = "Fine-Tuned CodeT5+ (Part B checkpoint)",
)


# ===========================================================================
# ## Step 8: Comparison Against Part B Zero-Shot Baseline
#
# Prefer reading the actual saved values from Part B's CSV output.
# Fall back to the hardcoded placeholder only if the file is absent.
# ===========================================================================
PARTB_CSV = "outputs/tuning_results_partB.csv"

# Try to read Part B's zero-shot stats from the Optuna results CSV.
# The CSV doesn't store zero-shot BLEU directly, so we also check for a
# small JSON sidecar that Part B can optionally write (see note below).
PARTB_STATS_JSON = "outputs/partB_zeroshot_stats.json"

zeroshot_mean   = None
zeroshot_syntax = None

if os.path.exists(PARTB_STATS_JSON):
    import json
    with open(PARTB_STATS_JSON) as f:
        zs = json.load(f)
    zeroshot_mean   = zs.get("mean")
    zeroshot_syntax = zs.get("syntax_pct")
    print(f"\n✅ Loaded Part B zero-shot stats from {PARTB_STATS_JSON}")

if zeroshot_mean is None:
    # Hardcoded fallback — replace with your actual Part B output if the
    # JSON sidecar doesn't exist yet.
    zeroshot_mean   = 4.24
    zeroshot_syntax = 44.0
    print(f"\n⚠️  Using hardcoded Part B zero-shot baseline "
          f"(mean BLEU={zeroshot_mean}, syntax={zeroshot_syntax}%).")
    print("   To avoid this, add the following lines near the end of "
          "dfs_PartB_tuning.py, after the zero-shot evaluation block:\n"
          "     import json\n"
          "     with open('outputs/partB_zeroshot_stats.json', 'w') as _f:\n"
          "         json.dump({'mean': zeroshot_stats['mean'],\n"
          "                    'syntax_pct': sum(zeroshot_syntax_valid)/len(zeroshot_syntax_valid)*100}, _f)")

improvement_factor = (finetuned_stats["mean"] / zeroshot_mean
                      if zeroshot_mean > 0 else float("nan"))

print(f"\n{'='*60}")
print(f"📈 Results Summary")
print(f"{'='*60}")
print(f"  Zero-Shot  (Part B) → Mean BLEU: {zeroshot_mean:.2f}  |  Syntax valid: {zeroshot_syntax:.1f}%")
print(f"  Fine-Tuned (Part B) → Mean BLEU: {finetuned_stats['mean']:.2f}  "
      f"|  Syntax valid: {sum(finetuned_syntax_valid)/len(finetuned_syntax_valid)*100:.1f}%")
print(f"  Improvement factor:  {improvement_factor:.1f}x")
print(f"{'='*60}")


# ===========================================================================
# ## Step 9: Plots
# ===========================================================================

# — Fine-tuned BLEU distribution —
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(finetuned_bleu_scores, bins=20, color="mediumseagreen",
         edgecolor="black", alpha=0.7)
ax1.axvline(finetuned_stats["mean"],   color="red",   linestyle="--", linewidth=2,
            label=f"Mean: {finetuned_stats['mean']:.2f}")
ax1.axvline(finetuned_stats["median"], color="navy",  linestyle="--", linewidth=2,
            label=f"Median: {finetuned_stats['median']:.2f}")
ax1.set_xlabel("BLEU Score"); ax1.set_ylabel("Frequency")
ax1.set_title("Fine-Tuned BLEU Distribution (Part B checkpoint)", fontweight="bold")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.boxplot(finetuned_bleu_scores, vert=True, patch_artist=True,
            boxprops    = dict(facecolor="mediumseagreen", alpha=0.7),
            medianprops = dict(color="red", linewidth=2))
ax2.set_ylabel("BLEU Score")
ax2.set_title("Fine-Tuned BLEU Box Plot (Part B checkpoint)", fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("outputs/partC_finetuned_bleu_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fine-tuned BLEU distribution saved to outputs/partC_finetuned_bleu_distribution.png")

# — Zero-Shot vs Fine-Tuned comparison (using scalar zero-shot mean as reference line) —
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sorted(finetuned_bleu_scores), label="Fine-Tuned CodeT5+ (Part B)",
        color="mediumseagreen", linewidth=2)
ax.axhline(zeroshot_mean, color="steelblue", linestyle="--", linewidth=2,
           label=f"Zero-Shot Mean (Part B): {zeroshot_mean:.2f}")
ax.axhline(finetuned_stats["mean"], color="mediumseagreen", linestyle="--",
           linewidth=2, alpha=0.6,
           label=f"Fine-Tuned Mean: {finetuned_stats['mean']:.2f}")
ax.set_xlabel("Test Example (sorted by BLEU)"); ax.set_ylabel("BLEU Score")
ax.set_title("Zero-Shot vs Fine-Tuned CodeT5+ (Part B checkpoint) — BLEU",
             fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/partC_zeroshot_vs_finetuned_bleu.png", dpi=150, bbox_inches="tight")
plt.close()
print("Comparison plot saved to outputs/partC_zeroshot_vs_finetuned_bleu.png")


# ===========================================================================
# ## Done
# ===========================================================================
NOTEBOOK_END_TIME = datetime.now()
elapsed = NOTEBOOK_END_TIME - NOTEBOOK_START_TIME
total_mins, total_secs = divmod(int(elapsed.total_seconds()), 60)
print(f"\n✅ Part C complete in: {total_mins}m {total_secs}s")
print(f"   Finished at: {NOTEBOOK_END_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
