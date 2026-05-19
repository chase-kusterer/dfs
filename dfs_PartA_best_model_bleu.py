#!/usr/bin/env python3
"""
dfs_PartA_best_model_bleu.py
============================
Rebuilds the best-performing model from part_a_results.xlsx using its exact
hyperparameter settings, trains it on MBPP, evaluates BLEU on the test split
(IDs 11-510), and produces a histogram + box plot matching the style of the
original dfs_PartA_tuning.py visualisation.

Best model hyperparameters (Trial 19, val_accuracy ≈ 0.5020):
  num_layers   = 5
  d_model      = 512
  num_heads    = 4
  d_ff         = 2048
  dropout_rate = 0.221838
  learning_rate= 0.000254

Fixed config:
  max_input_tokens  = 12000
  max_target_tokens = 20000
  max_input_length  = 128
  max_target_length = 256
  batch_size        = 32
  epochs            = 25
  patience          = 5
"""

import os
import copy
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import evaluate

sys.path.insert(0, "/home/claude")
from dfs_utils_torch import (
    prepare_training_data,
    TextVectorizer,
    add_python_hint,
    build_transformer_model,
    strip_special_tokens_from_target,
    generate_best,
    sequential_generate_codes,
    evaluate_model_with_bleu,
)

# ── Seeds ────────────────────────────────────────────────────────────────────
SEED = 702
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"Device: {DEVICE}")

# ── Config ───────────────────────────────────────────────────────────────────
config = {
    # fixed
    "max_input_tokens" : 12000,
    "max_target_tokens": 20000,
    "max_input_length" : 128,
    "max_target_length": 256,
    "batch_size"       : 32,
    "epochs"           : 25,
    "patience"         : 5,
    # best model hyperparameters from trial 19
    "num_layers"   : 5,
    "d_model"      : 512,
    "num_heads"    : 4,
    "d_ff"         : 2048,
    "dropout_rate" : 0.221838,
    "learning_rate": 0.000254,
}

START_TOKEN = "[START]"
END_TOKEN   = "[END]"

os.makedirs("outputs", exist_ok=True)

# ── Step 1: Load data ─────────────────────────────────────────────────────────
print("\n--- Step 1: Loading MBPP dataset ---")
data = prepare_training_data()

train_inputs  = data["train"]["prompts"]
train_targets = [f"{START_TOKEN} {c} {END_TOKEN}" for c in data["train"]["code"]]

val_inputs    = data["val"]["prompts"]
val_targets   = [f"{START_TOKEN} {c} {END_TOKEN}" for c in data["val"]["code"]]

test_inputs   = data["test"]["prompts"]
test_targets  = data["test"]["code"]   # raw, no special tokens (for BLEU)

# ── Step 2: Build vocabularies ────────────────────────────────────────────────
print("\n--- Step 2: Building vocabularies ---")
input_vectorizer = TextVectorizer(
    max_tokens=config["max_input_tokens"],
    output_sequence_length=config["max_input_length"],
)
input_vectorizer.adapt(train_inputs)

target_vectorizer = TextVectorizer(
    max_tokens=config["max_target_tokens"],
    output_sequence_length=config["max_target_length"],
)
target_vectorizer.adapt(train_targets)

print(f"  Input vocab size : {len(input_vectorizer.get_vocabulary())}")
print(f"  Target vocab size: {len(target_vectorizer.get_vocabulary())}")

# ── Step 3: Vectorize splits ──────────────────────────────────────────────────
print("\n--- Step 3: Vectorizing ---")
def make_loader(inputs, targets, shuffle=True):
    enc = input_vectorizer(inputs).long()
    tgt = target_vectorizer(targets).long()
    dec_in  = tgt[:, :-1]
    dec_tgt = tgt[:, 1:]
    ds = TensorDataset(enc, dec_in, dec_tgt)
    return DataLoader(ds, batch_size=config["batch_size"], shuffle=shuffle)

train_loader = make_loader(train_inputs, train_targets, shuffle=True)
val_loader   = make_loader(val_inputs,   val_targets,   shuffle=False)

# ── Step 4: Build model ───────────────────────────────────────────────────────
print("\n--- Step 4: Building transformer ---")
transformer = build_transformer_model(
    input_vocab_size  = len(input_vectorizer.get_vocabulary()),
    target_vocab_size = len(target_vectorizer.get_vocabulary()),
    num_layers        = config["num_layers"],
    d_model           = config["d_model"],
    num_heads         = config["num_heads"],
    d_ff              = config["d_ff"],
    dropout_rate      = config["dropout_rate"],
    max_input_len     = config["max_input_length"],
    max_target_len    = config["max_target_length"] - 1,
).to(DEVICE)

n_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
print(f"  Trainable parameters: {n_params:,}")

loss_fn   = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(
    transformer.parameters(),
    lr    = config["learning_rate"],
    betas = (0.9, 0.98),
    eps   = 1e-9,
)

# ── Step 5: Training loop with early stopping ─────────────────────────────────
def _token_accuracy(logits, targets):
    preds   = logits.argmax(dim=-1)
    mask    = targets != 0
    correct = ((preds == targets) & mask).sum().item()
    total   = mask.sum().item()
    return correct / max(1, total)

def run_epoch(model, loader, train: bool):
    model.train(train)
    total_loss, total_acc, n = 0.0, 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for enc_in, dec_in, dec_tgt in loader:
            enc_in  = enc_in.to(DEVICE)
            dec_in  = dec_in.to(DEVICE)
            dec_tgt = dec_tgt.to(DEVICE)
            logits  = model(enc_in, dec_in)
            loss    = loss_fn(logits.reshape(-1, logits.size(-1)), dec_tgt.reshape(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_acc  += _token_accuracy(logits.detach(), dec_tgt)
            n += 1
    return total_loss / max(1, n), total_acc / max(1, n)

print("\n--- Step 5: Training ---")
best_val_loss  = float("inf")
best_state     = None
patience_ctr   = 0
history        = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

for epoch in range(1, config["epochs"] + 1):
    tr_loss, tr_acc   = run_epoch(transformer, train_loader, train=True)
    val_loss, val_acc = run_epoch(transformer, val_loader,   train=False)

    history["loss"].append(tr_loss)
    history["val_loss"].append(val_loss)
    history["accuracy"].append(tr_acc)
    history["val_accuracy"].append(val_acc)

    print(f"  Epoch {epoch:2d}/{config['epochs']} | "
          f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f} | "
          f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = copy.deepcopy(transformer.state_dict())
        patience_ctr  = 0
    else:
        patience_ctr += 1
        if patience_ctr >= config["patience"]:
            print(f"  Early stopping at epoch {epoch} (patience={config['patience']})")
            break

# Restore best weights
if best_state is not None:
    transformer.load_state_dict(best_state)
print(f"  Best val_loss: {best_val_loss:.4f}")

# ── Step 6: BLEU evaluation on test split ─────────────────────────────────────
print("\n--- Step 6: BLEU evaluation (test split) ---")

target_vocab = target_vectorizer.get_vocabulary()
id_to_token  = {i: tok for i, tok in enumerate(target_vocab)}
token_to_id  = {tok: i for i, tok in enumerate(target_vocab)}

def get_token_id(token_str):
    return token_to_id.get(token_str) or token_to_id.get(token_str.lower())

start_id = get_token_id(START_TOKEN)
end_id   = get_token_id(END_TOKEN)
assert start_id is not None and end_id is not None, "START/END token ids not found!"

def generate_code_wrapper(prompt_text):
    return generate_best(
        prompt_text,
        model              = transformer,
        input_vectorizer   = input_vectorizer,
        start_id           = start_id,
        end_id             = end_id,
        id_to_token        = id_to_token,
        max_len            = config["max_target_length"] - 1,
        temperature        = 0.5,
        repetition_penalty = 1.5,
        device             = DEVICE,
    )

test_prompts_python = [add_python_hint(p) for p in test_inputs]
test_targets_clean  = [strip_special_tokens_from_target(t) for t in test_targets]

bleu_metric = evaluate.load("bleu")

print("  Generating code for test prompts (generate_best, temp=0.5, rep_penalty=1.5)...")
generated_codes = sequential_generate_codes(
    test_prompts_python,
    generate_fn=generate_code_wrapper,
    desc="Part A Best Model",
)

bleu_scores, syntax_valid, _, stats = evaluate_model_with_bleu(
    test_prompts_python,
    test_targets_clean,
    None,
    bleu_metric,
    model_label="Part A Best Model",
    generated_codes=generated_codes,
)

# ── Step 7: Plot — mirrors dfs_PartA_tuning.py ───────────────────────────────
print("\n--- Step 7: Generating BLEU distribution plot ---")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(bleu_scores, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
ax1.axvline(stats["mean"],   color="red",   linestyle="--", linewidth=2,
            label=f"Mean: {stats['mean']:.2f}")
ax1.axvline(stats["median"], color="green", linestyle="--", linewidth=2,
            label=f"Median: {stats['median']:.2f}")
ax1.set_xlabel("BLEU Score", fontsize=12)
ax1.set_ylabel("Frequency",  fontsize=12)
ax1.set_title("Distribution of BLEU Scores", fontsize=14, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.boxplot(bleu_scores, vert=True, patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
            medianprops=dict(color="red", linewidth=2))
ax2.set_ylabel("BLEU Score", fontsize=12)
ax2.set_title("Box Plot of BLEU Scores", fontsize=14, fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plot_path = "outputs/bleu_distribution_best_model.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Plot saved → {plot_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY — Part A Best Model")
print("=" * 60)
print(f"  num_layers    : {config['num_layers']}")
print(f"  d_model       : {config['d_model']}")
print(f"  num_heads     : {config['num_heads']}")
print(f"  d_ff          : {config['d_ff']}")
print(f"  dropout_rate  : {config['dropout_rate']:.6f}")
print(f"  learning_rate : {config['learning_rate']:.6f}")
print(f"  Best val_loss : {best_val_loss:.4f}")
print(f"  BLEU mean     : {stats['mean']:.2f}")
print(f"  BLEU median   : {stats['median']:.2f}")
print(f"  BLEU std      : {stats['std']:.2f}")
print(f"  BLEU min/max  : {stats['min']:.2f} / {stats['max']:.2f}")
print(f"  Syntax valid  : {stats['syntax_valid_count']}/{len(syntax_valid)} "
      f"({stats['syntax_valid_pct']:.1f}%)")
print("=" * 60)
