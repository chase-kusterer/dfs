#!/usr/bin/env python
# coding: utf-8
"""
MSDS458 Assignment 3 — Part B: Exploring Pretrained Models for Text-to-Code Generation
                                — **PyTorch port**
"""

NOTEBOOK_VERSION = "1.0-pytorch-fixed"
QUARTER = "Spring 2026"

from datetime import datetime
NOTEBOOK_START_TIME = datetime.now()
print(f"Notebook Version: {NOTEBOOK_VERSION} | {QUARTER}")

# ## 🧩 Step 1: Setup and Imports
import os, sys, random, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import evaluate
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, RobertaTokenizer, AutoTokenizer
from datasets import load_dataset

# Set HF Token for authentication
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

# Reproducibility
SEED = 702 
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

print("✅ Libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")

# ## 📦 Step 1B: Import Shared Utilities
from dfs_utils_torch import (
    prepare_training_data,
    add_python_hint,
    compute_bleu_for_code,
    evaluate_model_with_bleu,
    analyze_bleu_results,
    batch_generate_codes,
)

# ## 🗂️ Step 2: Load and Process MBPP Dataset
data = prepare_training_data("MBPP")
train_prompts = data["train"]["prompts"]
test_prompts  = data["test"]["prompts"]
test_code     = data["test"]["code"]

# ## 🤖 Step 4: Load Pretrained CodeT5+ Model
MODEL_NAME = "Salesforce/codet5p-220m-py"

print(f"Loading pretrained model: {MODEL_NAME}")

# FIX 1: Manually load clean RobertaTokenizer to avoid metadata corruption TypeError
#MODEL_PATH = "/home/user1/.cache/huggingface/hub/models--Salesforce--codet5p-220m-py/snapshots/8844b8a8b0600ffce926b71880003f8b21dfd5e6"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=os.environ["HF_TOKEN"],
    use_fast=False
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    use_safetensors=False,
    token=os.environ["HF_TOKEN"]
).to(DEVICE)

print(f"✅ Model and tokenizer loaded on {DEVICE}!")

# ## 🧪 Step 5: Zero-Shot Evaluation
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

# ## 📊 Step 5B: BLEU Evaluation
bleu_metric = evaluate.load("bleu")
test_prompts_python = [add_python_hint(p) for p in test_prompts]

print("⚡ Generating zero-shot test codes...")
zeroshot_generated = batch_generate_codes(
    test_prompts_python,
    model=model,
    tokenizer=tokenizer,
    device=DEVICE,
)

zeroshot_bleu_scores, zeroshot_syntax_valid, zeroshot_codes, zeroshot_stats = evaluate_model_with_bleu(
    test_prompts_python,
    test_code,
    None,
    bleu_metric,
    model_label="Zero-Shot CodeT5+",
    generated_codes=zeroshot_generated,
)

analyze_bleu_results(
    zeroshot_bleu_scores,
    zeroshot_syntax_valid,
    zeroshot_codes,
    test_prompts_python,
    test_code,
    zeroshot_stats,
    model_label="Zero-Shot CodeT5+",
)

NOTEBOOK_END_TIME = datetime.now()
print(f"✅ Part B Complete in: {NOTEBOOK_END_TIME - NOTEBOOK_START_TIME}")