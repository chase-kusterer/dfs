
#!/usr/bin/env python
# coding: utf-8
"""
MSDS458 Assignment 3 — Part C: Fine-tuning Pretrained Models for Text-to-Code
                                Generation — **PyTorch port**
"""

# In[ ]:

NOTEBOOK_VERSION = "1.0-pytorch-fixed"
QUARTER = "Spring 2026"

from datetime import datetime
NOTEBOOK_START_TIME = datetime.now()
print(f"Notebook Version: {NOTEBOOK_VERSION} | {QUARTER}")


# ## 🧩 Step 1: Setup and Imports

# In[ ]:

print("✅ Packages already installed!")

import os, sys, random, copy
import numpy as np
import pandas as pd
import evaluate
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, RobertaTokenizer, get_linear_schedule_with_warmup

from datasets import load_dataset

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Set HF Token for authentication
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

print("✅ Libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")


# ## 📦 Step 1B: Import Shared Utilities

# In[ ]:

import os, sys, urllib.request

print("Using local dfs_utils_torch.py")

from dfs_utils_torch import (
    prepare_training_data,
    add_python_hint,
    compute_bleu_for_code,
    evaluate_model_with_bleu,
    analyze_bleu_results,
    batch_generate_codes,
)

print("✅ Utilities imported successfully")


# ## 🗂️ Step 2: Load and Process MBPP Dataset

# In[ ]:

print("Loading and processing MBPP dataset...")
print("=" * 60)

data = prepare_training_data("MBPP")
train_prompts = data["train"]["prompts"]
train_code    = data["train"]["code"]
val_prompts   = data["val"]["prompts"]
val_code      = data["val"]["code"]
test_prompts  = data["test"]["prompts"]
test_code     = data["test"]["code"]

print("=" * 60)
print(f"📊 Dataset sizes:")
print(f"  Train: {len(train_prompts)}")
print(f"  Val:   {len(val_prompts)}")
print(f"  Test:  {len(test_prompts)}")
print("✅ Dataset loaded and processed successfully!")


# ## 🤖 Step 4: Load Pretrained CodeT5+ Model

# In[ ]:

MODEL_NAME = "Salesforce/codet5p-220m-py"

print(f"Loading pretrained model: {MODEL_NAME}")
print("Note: CodeT5+ is an improved version of CodeT5 with better pretraining")

# FIX 1: Use RobertaTokenizer with base-vocab to avoid TypeError: Input must be a List
tokenizer = RobertaTokenizer.from_pretrained(
    "roberta-base", 
    use_fast=False, 
    add_prefix_space=True
)
tokenizer.add_tokens(["<pad>", "<s>", "</s>", "<unk>", "<mask>"], special_tokens=True)

# FIX 2: Use use_safetensors=False to bypass CVE-2025-32434 security block in PyTorch 2.6
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, 
    use_safetensors=False
).to(DEVICE)

print(f"\n✅ Model and tokenizer loaded!")
print(f"Model type: {type(model).__name__}")
print(f"Tokenizer vocab size: {tokenizer.vocab_size:,}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# In[ ]:

# Test the tokenizer on a sample.
sample_prompt = train_prompts[0]
sample_code   = train_code[0]

inputs  = tokenizer(sample_prompt, return_tensors="pt", truncation=True, max_length=128)
targets = tokenizer(sample_code,   return_tensors="pt", truncation=True, max_length=256)

print(f"\nTokenized input shape:  {tuple(inputs['input_ids'].shape)}")
print(f"Decoded back: {tokenizer.decode(inputs['input_ids'][0, :20], skip_special_tokens=False)}")


# ## 🔧 Step 5: Prepare Data for Fine-tuning

# In[ ]:

MAX_INPUT_LENGTH  = 128
MAX_TARGET_LENGTH = 256
BATCH_SIZE        = 8 # Reduce to 4 if you hit memory issues on M1

def tokenize_function(prompts, codes):
    model_inputs = tokenizer(
        prompts,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    labels = tokenizer(
        codes,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    labels_ids = labels["input_ids"]
    labels_ids = np.where(labels_ids == tokenizer.pad_token_id, -100, labels_ids)

    return {
        "input_ids":      model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels":         labels_ids,
    }

print("🔄 Tokenizing all datasets...")
train_dataset = tokenize_function(train_prompts, train_code)
val_dataset   = tokenize_function(val_prompts, val_code)
test_dataset  = tokenize_function(test_prompts, test_code)

class Seq2SeqDataset(Dataset):
    def __init__(self, encoded):
        self.input_ids      = torch.from_numpy(encoded["input_ids"]).long()
        self.attention_mask = torch.from_numpy(encoded["attention_mask"]).long()
        self.labels         = torch.from_numpy(encoded["labels"]).long()
    def __len__(self): return self.input_ids.size(0)
    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }

train_loader = DataLoader(Seq2SeqDataset(train_dataset), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(Seq2SeqDataset(val_dataset),   batch_size=BATCH_SIZE, shuffle=False)
print("✅ DataLoaders created")


# ## 🚀 Step 6: Fine-tune CodeT5

# In[ ]:

LEARNING_RATE = 5e-5
EPOCHS        = 5
WARMUP_STEPS  = 100

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

initial_weights = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

model_save_path = "codet5_finetuned_mbpp.pt"
early_stop_patience = 2
best_val_loss = float("inf")
best_state_dict = None
patience_ctr = 0

def run_train_epoch():
    model.train()
    total = 0.0
    for batch in tqdm(train_loader, desc="train", leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / len(train_loader)

@torch.no_grad()
def run_val_epoch():
    model.eval()
    total = 0.0
    for batch in tqdm(val_loader, desc="val", leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        total += outputs.loss.item()
    return total / len(val_loader)

print(f"\n🚀 Starting fine-tuning...")
history = {"loss": [], "val_loss": []}

for epoch in range(1, EPOCHS + 1):
    train_loss = run_train_epoch()
    val_loss   = run_val_epoch()
    history["loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    print(f"Epoch {epoch:2d} | loss {train_loss:.4f} | val_loss {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state_dict = copy.deepcopy(model.state_dict())
        torch.save(best_state_dict, model_save_path)
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= early_stop_patience:
            print("  ↳ early stopping.")
            break

if best_state_dict:
    model.load_state_dict(best_state_dict)
    print("✅ Best weights restored.")


# ## 🧪 Step 7: Evaluate Fine-Tuned Model

# In[ ]:

@torch.no_grad()
def generate_code(prompt, max_length=256, num_beams=5):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# BLEU Evaluation
bleu_metric = evaluate.load("bleu")
test_prompts_python = [add_python_hint(p) for p in test_prompts]

print("⚡ Generating test codes...")
finetuned_generated = batch_generate_codes(test_prompts_python, model=model, tokenizer=tokenizer, device=DEVICE)

finetuned_bleu_scores, finetuned_syntax_valid, finetuned_codes, finetuned_stats = evaluate_model_with_bleu(
    test_prompts_python, test_code, None, bleu_metric, model_label="Fine-Tuned CodeT5+", generated_codes=finetuned_generated
)

# Step 8 Comparison (Placeholders - Update with your actual Part B results!)
PART_B_ZEROSHOT_BLEU_MEAN = 4.24 
PART_B_ZEROSHOT_SYNTAX_PCT = 44.0

print(f"\n📈 Final Improvement Factor (FT vs Zero-Shot): {finetuned_stats['mean']/PART_B_ZEROSHOT_BLEU_MEAN:.1f}x")

NOTEBOOK_END_TIME = datetime.now()
print(f"✅ Notebook complete in: {NOTEBOOK_END_TIME - NOTEBOOK_START_TIME}")