#!/usr/bin/env python
# coding: utf-8
"""
MSDS458 Assignment 3 — Part B: Exploring Pretrained Models for Text-to-Code Generation
                                — **PyTorch port**
"""
from datetime import datetime
NOTEBOOK_START_TIME = datetime.now()

# general dependencies
import os, sys, random, urllib.request
from   tqdm import tqdm

# data science dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# deep learning dependencies
import torch
from   transformers import AutoModelForSeq2SeqLM, RobertaTokenizer, AutoTokenizer
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

# setting HF Token for authentication
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]  = "1"
os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]   = "0"

# setting seeds
SEED = 702
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

os.makedirs("outputs", exist_ok=True)

## importing dataset ##
data          = prepare_training_data("MBPP")
train_prompts = data["train"]["prompts"]
test_prompts  = data["test"]["prompts"]
test_code     = data["test"]["code"]

## importing pretrained codeT5+ model ##
MODEL_NAME = "Salesforce/codet5p-220m-py"

print(f"Loading pretrained model: {MODEL_NAME}")

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

## zero-shot evaluation ##
@torch.no_grad()
def generate_code(prompt,
                  max_length = 256,
                  num_beams  = 5):

    inputs = tokenizer(prompt,
                       return_tensors = "pt",
                       truncation     = True,
                       max_length     = 128).to(DEVICE)

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask       = inputs.get("attention_mask"),
        max_length           = max_length,
        num_beams            = num_beams,
        early_stopping       = True,
        no_repeat_ngram_size = 2,
    )

    return tokenizer.decode(outputs[0],
                            skip_special_tokens = True)

## BLEU evaluation ##
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