#!/usr/bin/env python
# coding: utf-8
"""
MSDS458 Assignment 3 — Part B: Exploring Pretrained Models for Text-to-Code Generation
                                — **PyTorch port**
"""
from datetime import datetime
NOTEBOOK_START_TIME = datetime.now()

# general dependencies
import os, sys, random, copy, urllib.request
from   tqdm import tqdm

# data science dependencies
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display required
import matplotlib.pyplot as plt

# deep learning dependencies
import torch
import torch.nn as nn
from   transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from   torch.utils.data import DataLoader, Dataset
import optuna

# hugging face dependencies
import evaluate
from   datasets import load_dataset

# utility imports
from dfs_utils_torch import (
    prepare_training_data,
    add_python_hint,
    compute_bleu_for_code,
    evaluate_model_with_bleu,
    analyze_bleu_results,
    batch_generate_codes,
)

# ── Environment ───────────────────────────────────────────────────────────────
os.environ["HF_TOKEN"]                       = os.getenv("HF_TOKEN", "")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]  = "1"
os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]  = "0"

# ── Seeds ─────────────────────────────────────────────────────────────────────
SEED = 702
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- HYPERPARAMETER TUNING CONFIGURATION ---
# Tuning ranges — objective() reads directly from these
config = {
    "max_input_length"  : {"type": "int", "low": 64,  "high": 256, "step": 64},
    "max_target_length" : {"type": "int", "low": 128, "high": 512, "step": 128},
    "batch_size"        : {"type": "categorical", "choices": [4, 8, 16, 32, 64]},
}

tuning_config = {
    "learning_rate"        : {"type": "float", "low": 1e-5, "high": 5e-4, "log": True},
    "warmup_ratio"         : {"type": "float", "low": 0.05, "high": 0.30},
    "weight_decay"         : {"type": "float", "low": 0.0,  "high": 0.1},
    "num_beams"            : {"type": "int",   "low": 1,    "high": 8},
    "max_new_tokens"       : {"type": "int",   "low": 64,   "high": 512,  "step": 64},
    "no_repeat_ngram_size" : {"type": "int",   "low": 2,    "high": 5},
}

# Fixed training constants (not tuned)
EPOCHS   = 25
PATIENCE = 10

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

os.makedirs("outputs", exist_ok=True)
print(f"Running on: {DEVICE}")

# ── Model Name ────────────────────────────────────────────────────────────────
MODEL_NAME = "Salesforce/codet5p-220m-py"

# =============================================================================
# ## Step 1: Load Dataset
# =============================================================================
print("\nLoading MBPP dataset...")
data          = prepare_training_data("MBPP")
train_prompts = data["train"]["prompts"]
train_code    = data["train"]["code"]
val_prompts   = data["val"]["prompts"]   if "val"  in data else train_prompts[:50]
val_code      = data["val"]["code"]      if "val"  in data else train_code[:50]
test_prompts  = data["test"]["prompts"]
test_code     = data["test"]["code"]

print(f"   Train: {len(train_prompts)} examples")
print(f"   Val:   {len(val_prompts)}   examples")
print(f"   Test:  {len(test_prompts)}  examples")

# =============================================================================
# ## Step 2: Load Pretrained Model and Tokenizer
# =============================================================================
print(f"\nLoading pretrained model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token    = os.environ["HF_TOKEN"],
    use_fast = False,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    use_safetensors = False,
    token           = os.environ["HF_TOKEN"],
).to(DEVICE)

print(f"Model and tokenizer loaded on {DEVICE}!")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# =============================================================================
# ## Step 3: Zero-Shot Evaluation (Baseline before fine-tuning)
# =============================================================================
bleu_metric         = evaluate.load("bleu")
test_prompts_python = [add_python_hint(p) for p in test_prompts]

print("\nGenerating zero-shot test codes...")
zeroshot_generated = batch_generate_codes(
    test_prompts_python,
    model     = model,
    tokenizer = tokenizer,
    device    = DEVICE,
)

zeroshot_bleu_scores, zeroshot_syntax_valid, zeroshot_codes, zeroshot_stats = evaluate_model_with_bleu(
    test_prompts_python,
    test_code,
    None,
    bleu_metric,
    model_label    = "Zero-Shot CodeT5+",
    generated_codes = zeroshot_generated,
)

analyze_bleu_results(
    zeroshot_bleu_scores,
    zeroshot_syntax_valid,
    zeroshot_codes,
    test_prompts_python,
    test_code,
    zeroshot_stats,
    model_label = "Zero-Shot CodeT5+",
)

# Save zero-shot BLEU distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(zeroshot_bleu_scores, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
ax1.axvline(zeroshot_stats["mean"],   color="red",   linestyle="--", linewidth=2,
            label=f"Mean: {zeroshot_stats['mean']:.2f}")
ax1.axvline(zeroshot_stats["median"], color="green", linestyle="--", linewidth=2,
            label=f"Median: {zeroshot_stats['median']:.2f}")
ax1.set_xlabel("BLEU Score"); ax1.set_ylabel("Frequency")
ax1.set_title("Zero-Shot BLEU Distribution", fontweight="bold")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.boxplot(zeroshot_bleu_scores, vert=True, patch_artist=True,
            boxprops    = dict(facecolor="lightblue", alpha=0.7),
            medianprops = dict(color="red", linewidth=2))
ax2.set_ylabel("BLEU Score")
ax2.set_title("Zero-Shot BLEU Box Plot", fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("outputs/zeroshot_bleu_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Zero-shot BLEU distribution saved to outputs/zeroshot_bleu_distribution.png")

# =============================================================================
# ## Step 4: Fine-Tuning Dataset
# Wraps prompt→code pairs into a PyTorch Dataset for the DataLoader
# =============================================================================
class CodeT5Dataset(Dataset):
    """Tokenizes prompt/code pairs for CodeT5+ fine-tuning."""

    def __init__(self, prompts, codes, tokenizer, max_input_len, max_target_len):
        self.tokenizer      = tokenizer
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len
        self.prompts        = [add_python_hint(p) for p in prompts]
        self.codes          = codes

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.prompts[idx],
            max_length  = self.max_input_len,
            padding     = "max_length",
            truncation  = True,
            return_tensors = "pt",
        )
        with self.tokenizer.as_target_tokenizer():
            dec = self.tokenizer(
                self.codes[idx],
                max_length  = self.max_target_len,
                padding     = "max_length",
                truncation  = True,
                return_tensors = "pt",
            )

        labels = dec["input_ids"].squeeze()
        # Replace padding token id with -100 so CrossEntropyLoss ignores it
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids"      : enc["input_ids"].squeeze(),
            "attention_mask" : enc["attention_mask"].squeeze(),
            "labels"         : labels,
        }


# =============================================================================
# ## Step 5: Generation Helper
# Used inside the Optuna objective to evaluate BLEU after each trial
# =============================================================================
@torch.no_grad()
def generate_code(prompt, trial_model, num_beams, max_new_tokens,
                  no_repeat_ngram_size, max_input_length=128):
    inputs = tokenizer(
        add_python_hint(prompt),
        return_tensors = "pt",
        truncation     = True,
        max_length     = max_input_length,
    ).to(DEVICE)

    outputs = trial_model.generate(
        inputs["input_ids"],
        attention_mask       = inputs.get("attention_mask"),
        max_new_tokens       = max_new_tokens,
        num_beams            = num_beams,
        early_stopping       = True,
        no_repeat_ngram_size = no_repeat_ngram_size,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =============================================================================
# ## Step 6: Optuna Objective — Fine-Tuning Study
#
# Hyperparameters being tuned:
#   learning_rate        — how fast to update pretrained weights (most critical)
#   warmup_ratio         — fraction of steps used for LR warmup
#   weight_decay         — L2 regularization strength
#   num_beams            — beam search width at generation time
#   max_new_tokens       — length budget for generated code
#   no_repeat_ngram_size — n-gram blocking to reduce repetitive output
#
# Objective: MAXIMIZE mean BLEU on the validation set
# =============================================================================
def objective(trial):

    # 1. Suggest hyperparameters for this trial (ranges defined in config and tuning_config)
    c  = config
    tc = tuning_config

    # from config
    max_input_length = trial.suggest_int(
        "max_input_length", c["max_input_length"]["low"], c["max_input_length"]["high"],
        step=c["max_input_length"].get("step", 1),
    )
    max_target_length = trial.suggest_int(
        "max_target_length", c["max_target_length"]["low"], c["max_target_length"]["high"],
        step=c["max_target_length"].get("step", 1),
    )
    batch_size = trial.suggest_categorical(
        "batch_size", c["batch_size"]["choices"],
    )
    epochs   = EPOCHS
    patience = PATIENCE

    # from tuning_config
    learning_rate = trial.suggest_float(
        "learning_rate", tc["learning_rate"]["low"], tc["learning_rate"]["high"],
        log=tc["learning_rate"].get("log", False),
    )
    warmup_ratio = trial.suggest_float(
        "warmup_ratio", tc["warmup_ratio"]["low"], tc["warmup_ratio"]["high"],
    )
    weight_decay = trial.suggest_float(
        "weight_decay", tc["weight_decay"]["low"], tc["weight_decay"]["high"],
    )
    num_beams = trial.suggest_int(
        "num_beams", tc["num_beams"]["low"], tc["num_beams"]["high"],
    )
    max_new_tokens = trial.suggest_int(
        "max_new_tokens", tc["max_new_tokens"]["low"], tc["max_new_tokens"]["high"],
        step=tc["max_new_tokens"].get("step", 1),
    )
    no_repeat_ngram_size = trial.suggest_int(
        "no_repeat_ngram_size", tc["no_repeat_ngram_size"]["low"], tc["no_repeat_ngram_size"]["high"],
    )

    # 2. Deep-copy the pretrained model so each trial starts from the same weights
    trial_model = copy.deepcopy(model).to(DEVICE)
    trial_model.train()

    # 3. Build DataLoaders
    train_dataset = CodeT5Dataset(
        train_prompts, train_code, tokenizer,
        max_input_length, max_target_length,
    )
    val_dataset = CodeT5Dataset(
        val_prompts, val_code, tokenizer,
        max_input_length, max_target_length,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size = batch_size,
                              shuffle    = True)

    val_loader   = DataLoader(val_dataset,
                              batch_size = batch_size,
                              shuffle    = False)

    # 4. Optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        trial_model.parameters(),
        lr           = learning_rate,
        weight_decay = weight_decay,
    )
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    # 5. Fine-tuning loop with early stopping on BLEU
    best_bleu        = -1.0
    best_model_state = None
    patience_ctr     = 0

    for epoch in range(1, epochs + 1):

        # — Training pass —
        trial_model.train()
        for batch in tqdm(train_loader, desc=f"  Trial {trial.number} | Epoch {epoch}", leave=False):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = trial_model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels,
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trial_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        # — Validation pass: compute mean BLEU on val set —
        trial_model.eval()
        val_bleu_scores = []
        for vp, vc in zip(val_prompts, val_code):
            generated = generate_code(vp, trial_model, num_beams,
                                       max_new_tokens, no_repeat_ngram_size,
                                       max_input_length)
            try:
                result = bleu_metric.compute(
                    predictions = [generated],
                    references  = [[vc]],
                )
                val_bleu_scores.append(result["bleu"] * 100)
            except Exception:
                val_bleu_scores.append(0.0)

        mean_val_bleu = float(np.mean(val_bleu_scores))

        # Report to Optuna for pruning
        trial.report(mean_val_bleu, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Early stopping + best-weight restoration
        if mean_val_bleu > best_bleu:
            best_bleu        = mean_val_bleu
            best_model_state = copy.deepcopy(trial_model.state_dict())
            patience_ctr     = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    # Restore best weights from this trial before returning
    if best_model_state is not None:
        trial_model.load_state_dict(best_model_state)

    trial.set_user_attr("best_val_bleu", best_bleu)
    return best_bleu   # Optuna will MAXIMIZE this


# =============================================================================
# ## Step 7: Run Optuna Study
# =============================================================================
study = optuna.create_study(
    direction  = "maximize",           # we want higher BLEU
    pruner     = optuna.pruners.MedianPruner(n_startup_trials = 3,
                                             n_warmup_steps   = 2),
    storage    = "sqlite:///optuna_partB.db",
    study_name = "partB_codet5_finetune",
    load_if_exists = True,
)

def trial_callback(study, trial):
    print(f"  Trial {trial.number:>2d} complete | "
          f"val_bleu: {trial.value:.4f} | "
          f"params: { {k: round(v, 6) if isinstance(v, float) else v for k, v in trial.params.items()} }")
    print(f"  Best so far → Trial {study.best_trial.number} | val_bleu: {study.best_value:.4f}")
    print("-" * 80)

study.optimize(objective, n_trials=20, callbacks=[trial_callback])

# =============================================================================
# ## Step 8: Retrieve Best Trial and Fine-Tune Final Model
# =============================================================================
best_trial  = study.best_trial
best_params = best_trial.params

print("\n" + "=" * 80)
print(f"Best Trial: {best_trial.number} | val_bleu: {study.best_value:.4f}")
print(f"   Params: {best_params}")
print("=" * 80)

# Fine-tune a fresh copy of the pretrained model using the best hyperparameters
print("\nFine-tuning final model with best hyperparameters...")
final_model = copy.deepcopy(model).to(DEVICE)

train_dataset_full = CodeT5Dataset(
    train_prompts, train_code, tokenizer,
    best_params["max_input_length"], best_params["max_target_length"],
)
train_loader_full = DataLoader(train_dataset_full,
                    batch_size = best_params["batch_size"], shuffle=True
)

final_optimizer  = torch.optim.AdamW(
    final_model.parameters(),
    lr           = best_params["learning_rate"],
    weight_decay = best_params["weight_decay"],
)
total_steps_final  = len(train_loader_full) * EPOCHS
warmup_steps_final = int(total_steps_final * best_params["warmup_ratio"])
final_scheduler    = get_linear_schedule_with_warmup(
    final_optimizer,
    num_warmup_steps   = warmup_steps_final,
    num_training_steps = total_steps_final,
)

best_final_loss  = float("inf")
best_final_state = None
patience_ctr     = 0
train_losses     = []

for epoch in range(1, EPOCHS + 1):
    final_model.train()
    epoch_loss = 0.0
    n_batches  = 0

    for batch in tqdm(train_loader_full,
                      desc = f"Final training | Epoch {epoch}"):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        outputs = final_model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels,
        )
        loss = outputs.loss
        final_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
        final_optimizer.step()
        final_scheduler.step()

        epoch_loss += loss.item()
        n_batches  += 1

    mean_epoch_loss = epoch_loss / max(1, n_batches)
    train_losses.append(mean_epoch_loss)
    print(f"  Epoch {epoch}/{EPOCHS} | train_loss: {mean_epoch_loss:.4f}")

    if mean_epoch_loss < best_final_loss:
        best_final_loss  = mean_epoch_loss
        best_final_state = copy.deepcopy(final_model.state_dict())
        patience_ctr     = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"  Early stopping triggered at epoch {epoch}")
            break

# Restore best weights and save
if best_final_state is not None:
    final_model.load_state_dict(best_final_state)

torch.save(final_model.state_dict(), "outputs/codet5_finetuned_mbpp.pt")
print("Fine-tuned model saved to outputs/codet5_finetuned_mbpp.pt")

# Plot training loss curve
fig, ax = plt.subplots(figsize = (9, 5))
ax.plot(range(1, len(train_losses) + 1), train_losses,
        marker = "o", color = "steelblue", linewidth = 2)
ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
ax.set_title("Final Model — Training Loss", fontweight = "bold")
ax.grid(True, alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/finetuned_training_loss.png", dpi=150, bbox_inches="tight")
plt.close()
print("Training loss curve saved to outputs/finetuned_training_loss.png")

# =============================================================================
# ## Step 9: Post-Fine-Tuning BLEU Evaluation on Test Set
# =============================================================================
print("\nGenerating fine-tuned test codes...")

def generate_finetuned(prompt):
    return generate_code(
        prompt, final_model,
        num_beams            = best_params["num_beams"],
        max_new_tokens       = best_params["max_new_tokens"],
        no_repeat_ngram_size = best_params["no_repeat_ngram_size"],
        max_input_length     = best_params["max_input_length"],
    )

finetuned_generated = [generate_finetuned(p) for p in
                       tqdm(test_prompts_python,
                            desc = "Generating")]

finetuned_bleu_scores, finetuned_syntax_valid, finetuned_codes, finetuned_stats = evaluate_model_with_bleu(
    test_prompts_python,
    test_code,
    None,
    bleu_metric,
    model_label     = "Fine-Tuned CodeT5+",
    generated_codes = finetuned_generated,
)

analyze_bleu_results(
    finetuned_bleu_scores,
    finetuned_syntax_valid,
    finetuned_codes,
    test_prompts_python,
    test_code,
    finetuned_stats,
    model_label = "Fine-Tuned CodeT5+",
)

# Save fine-tuned BLEU distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(finetuned_bleu_scores, bins=20, color="mediumseagreen",
         edgecolor="black", alpha=0.7)
ax1.axvline(finetuned_stats["mean"],   color="red",   linestyle="--", linewidth=2,
            label=f"Mean: {finetuned_stats['mean']:.2f}")
ax1.axvline(finetuned_stats["median"], color="navy",  linestyle="--", linewidth=2,
            label=f"Median: {finetuned_stats['median']:.2f}")
ax1.set_xlabel("BLEU Score"); ax1.set_ylabel("Frequency")
ax1.set_title("Fine-Tuned BLEU Distribution", fontweight="bold")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.boxplot(finetuned_bleu_scores, vert=True, patch_artist=True,
            boxprops    = dict(facecolor="mediumseagreen", alpha=0.7),
            medianprops = dict(color="red", linewidth=2))
ax2.set_ylabel("BLEU Score")
ax2.set_title("Fine-Tuned BLEU Box Plot", fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("outputs/finetuned_bleu_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fine-tuned BLEU distribution saved to outputs/finetuned_bleu_distribution.png")

# =============================================================================
# ## Step 10: Zero-Shot vs Fine-Tuned Comparison Plot
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sorted(zeroshot_bleu_scores),   label="Zero-Shot CodeT5+",   color="steelblue",     linewidth=2)
ax.plot(sorted(finetuned_bleu_scores),  label="Fine-Tuned CodeT5+",  color="mediumseagreen", linewidth=2)
ax.axhline(zeroshot_stats["mean"],  color="steelblue",      linestyle="--", alpha=0.6,
           label=f"Zero-Shot Mean: {zeroshot_stats['mean']:.2f}")
ax.axhline(finetuned_stats["mean"], color="mediumseagreen",  linestyle="--", alpha=0.6,
           label=f"Fine-Tuned Mean: {finetuned_stats['mean']:.2f}")
ax.set_xlabel("Test Example (sorted by BLEU)"); ax.set_ylabel("BLEU Score")
ax.set_title("Zero-Shot vs Fine-Tuned CodeT5+ — BLEU Comparison", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/zeroshot_vs_finetuned_bleu.png", dpi=150, bbox_inches="tight")
plt.close()
print("Comparison plot saved to outputs/zeroshot_vs_finetuned_bleu.png")

# =============================================================================
# ## Step 11: Export Optuna Results to CSV
# =============================================================================
df_results         = study.trials_dataframe()
df_results.columns = [
    c.replace("params_", "").replace("user_attrs_", "") for c in df_results.columns
]

export_filename = os.path.join(os.getcwd(), "outputs", "tuning_results_partB.csv")
df_results.to_csv(export_filename,
                  index = False)

print(f"\nTuning complete. Results exported to {export_filename}")
print(f"Best Trial: {study.best_trial.number} | Best Val BLEU: {study.best_value:.4f}")
print(f"\nZero-Shot  → Mean BLEU: {zeroshot_stats['mean']:.2f}")
print(f"Fine-Tuned → Mean BLEU: {finetuned_stats['mean']:.2f}")
print(f"Improvement: +{finetuned_stats['mean'] - zeroshot_stats['mean']:.2f} BLEU points")

# compiling time
NOTEBOOK_END_TIME = datetime.now()
elapsed = NOTEBOOK_END_TIME - NOTEBOOK_START_TIME
total_mins, total_secs = divmod(int(elapsed.total_seconds()), 60)
print(f"   Total time:   {total_mins}m {total_secs}s")
print(f"   Current time: {NOTEBOOK_END_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
