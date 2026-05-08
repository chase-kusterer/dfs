#!/usr/bin/env python
# coding: utf-8
"""
MSDS458 Assignment 3 — Part A: Building an Encoder–Decoder Transformer from Scratch
                                (Text → Code) — **PyTorch port**

This file is a one-to-one PyTorch translation of the original Keras notebook.
The structure, comments and pedagogical narration of the original are preserved
so that the two versions can be compared side-by-side.

Notes on the conversion
-----------------------
- TensorFlow / Keras has been replaced with PyTorch (`torch`, `torch.nn`).
- `tf.keras.layers.TextVectorization` has been replaced with the small
  `TextVectorizer` class in `dfs_utils_torch.py`.
- `tf.data.Dataset` pipelines were replaced with `torch.utils.data.DataLoader`s.
- The Keras `model.fit(...)` / `model.evaluate(...)` calls have been replaced
  with explicit PyTorch training / eval loops, plus a tiny `EarlyStopping`
  helper that mirrors the Keras callback's behaviour.
- All the Transformer building blocks live in `dfs_utils_torch.py`
  (the PyTorch counterpart to `assignment3_utils.py`).
"""

# In[1]:

NOTEBOOK_VERSION = "1.0-pytorch"
QUARTER = "Spring 2026"

from datetime import datetime
NOTEBOOK_START_TIME = datetime.now()
print(f"Notebook Version: {NOTEBOOK_VERSION} | {QUARTER}")


# ## 🧩 Step 1: Setup and Imports
#
# We are using:
# - **PyTorch** for model construction and training
# - **Hugging Face Datasets** to load the MBPP dataset
# - **NumPy / Pandas** for data preprocessing
# - **Matplotlib / Seaborn** for visualizations
# - **Evaluate / SacreBLEU** for evaluation metrics

# In[2]:

# Uncomment the line below if running on Colab or if packages are not installed
# !pip install -q torch datasets evaluate
print("✅ Packages ready!")


# In[3]:

# Core dependencies
import os
import random
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Hugging Face datasets & evaluation libraries
from datasets import load_dataset
import evaluate
from tqdm import tqdm

# For reproducibility
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

print("✅ Libraries imported successfully!")
print("PyTorch version:", torch.__version__)
print("Device:", DEVICE)


# ## 📦 Step 1B: Import Shared Utilities
#
# We use the PyTorch utilities module `dfs_utils_torch.py` shipped
# alongside this script. It is the PyTorch counterpart of the original
# `assignment3_utils.py` and provides the data-loading, Transformer building
# blocks, generation, and BLEU-evaluation helpers.

# In[4]:

import os, sys, urllib.request

IN_COLAB = "google.colab" in sys.modules
UTILS_BRANCH = "main"

def fetch_github_raw(user, repo, branch, file_path, local_path):
    url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{file_path}"
    urllib.request.urlretrieve(url, local_path)
    print(f"Fetched {file_path} from branch '{branch}'.")

# (The original notebook fetched assignment3_utils.py from GitHub. The PyTorch
#  port ships its own utils file; we just import it directly.)
print("Using local dfs_utils_torch.py")

from dfs_utils_torch import (
    # Data loading & processing
    prepare_training_data,
    # Tokenization
    TextVectorizer,
    # BLEU evaluation
    add_python_hint,
    compute_bleu_for_code,
    evaluate_model_with_bleu,
    analyze_bleu_results,
    # Transformer building blocks
    PositionalEncoding,
    ScaledDotProductAttention,
    MultiHeadAttention,
    create_padding_mask,
    create_look_ahead_mask,
    create_decoder_mask,
    point_wise_feed_forward_network,
    EncoderBlock,
    DecoderBlock,
    build_transformer_model,
    # Training utilities
    plot_training_history,
    TrainingHistory,
    # Decoding & generation
    ids_to_code_text,
    strip_special_tokens_from_target,
    generate_code_for_prompt,
    generate_best,
    sequential_generate_codes,
)

print("✅ Utilities imported successfully")


# ## 🗂️ Step 2: Load and Explore the MBPP Dataset

# In[5]:

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


# In[6]:

print("📋 Sample training examples:")
print("=" * 80)
for i in range(3):
    print(f"\nExample {i+1}:")
    print("-" * 80)
    print(f"📝 Prompt: {train_prompts[i][:200]}{'...' if len(train_prompts[i]) > 200 else ''}")
    print(f"💻 Code:   {train_code[i][:200]}{'...' if len(train_code[i]) > 200 else ''}")
print("=" * 80)


# ## 🧮 Step 3: Tokenization and Vocabulary (Text & Code)
#
# We use a small `TextVectorizer` (PyTorch port of Keras' TextVectorization)
# that lowercases input, builds a capped integer vocabulary, and pads/truncates
# every sequence to a fixed length. Index 0 is reserved for padding and index 1
# for OOV — same as Keras' default.

# In[7]:

# 🔧 Hyperparameters for tokenization
MAX_INPUT_TOKENS  = 12000   # vocabulary size for prompts
MAX_TARGET_TOKENS = 20000   # vocabulary size for code
MAX_INPUT_LENGTH  = 128
MAX_TARGET_LENGTH = 256

START_TOKEN = "[START]"
END_TOKEN   = "[END]"

print("✅ Hyperparameters set:")
print(f"   Input vocab:  {MAX_INPUT_TOKENS}")
print(f"   Target vocab: {MAX_TARGET_TOKENS}")


# In[8]:

# Add START/END tokens to target sequences.
train_inputs  = train_prompts
train_targets = [f"{START_TOKEN} {c} {END_TOKEN}" for c in train_code]

val_inputs  = val_prompts
val_targets = [f"{START_TOKEN} {c} {END_TOKEN}" for c in val_code]

test_inputs  = test_prompts
test_targets = [f"{START_TOKEN} {c} {END_TOKEN}" for c in test_code]

print("✅ Added START and END tokens to target sequences")
print(f"   Example target (first 100 chars): {train_targets[0][:100]}...")


# In[9]:

input_texts  = train_inputs  + val_inputs  + test_inputs
target_texts = train_targets + val_targets + test_targets

print(f"Total examples for vocabulary: {len(input_texts)}")
print("\nSample description:\n", input_texts[0][:200], "...")
print("\nSample target with tokens:\n", target_texts[0][:200], "...")


# In[10]:

# (Standardization is built into TextVectorizer — lowercasing only.)
# The PyTorch TextVectorizer always lowercases by default, matching the
# original `simple_standardize_text` / `simple_standardize_code`.

# In[11]:

# Input (natural language prompt) vectorizer
input_vectorizer = TextVectorizer(
    max_tokens=MAX_INPUT_TOKENS,
    output_sequence_length=MAX_INPUT_LENGTH,
)

# Target (code) vectorizer
target_vectorizer = TextVectorizer(
    max_tokens=MAX_TARGET_TOKENS,
    output_sequence_length=MAX_TARGET_LENGTH,
)

input_vectorizer.adapt(input_texts)
target_vectorizer.adapt(target_texts)

print("✅ Vectorizers adapted on MBPP text and code.")


# In[12]:

# 🔹 Inspect a few tokenized sequences
example_idx = 0
print("Original prompt:")
print(input_texts[example_idx])
print("\nVectorized prompt (first 40 ids):")
print(input_vectorizer([input_texts[example_idx]])[0][:40].numpy())

print("\nOriginal target with [START]/[END]:")
print(target_texts[example_idx])
print("\nVectorized target (first 40 ids):")
print(target_vectorizer([target_texts[example_idx]])[0][:40].numpy())


# ## 🧵 Step 4: Build the Encoder–Decoder Training Dataset

# In[13]:

BATCH_SIZE = 32
DECODER_SEQ_LEN = MAX_TARGET_LENGTH - 1  # because we shift by 1


# In[14]:

# Vectorize all input and target texts
encoder_input_data = input_vectorizer(input_texts)         # (N, MAX_INPUT_LENGTH)
full_target_data   = target_vectorizer(target_texts)       # (N, MAX_TARGET_LENGTH)

print("Encoder input shape:", tuple(encoder_input_data.shape))
print("Full target shape:",   tuple(full_target_data.shape))


# In[15]:

decoder_input_data  = full_target_data[:, :-1]
decoder_target_data = full_target_data[:, 1:]
print("Decoder input shape:",  tuple(decoder_input_data.shape))
print("Decoder target shape:", tuple(decoder_target_data.shape))


# In[16]:

# Vectorize the already-split datasets
train_encoder_in  = input_vectorizer(train_inputs)
train_full_target = target_vectorizer(train_targets)
train_decoder_in  = train_full_target[:, :-1]
train_decoder_tgt = train_full_target[:, 1:]

val_encoder_in    = input_vectorizer(val_inputs)
val_full_target   = target_vectorizer(val_targets)
val_decoder_in    = val_full_target[:, :-1]
val_decoder_tgt   = val_full_target[:, 1:]

test_encoder_in   = input_vectorizer(test_inputs)
test_full_target  = target_vectorizer(test_targets)
test_decoder_in   = test_full_target[:, :-1]
test_decoder_tgt  = test_full_target[:, 1:]

print("✅ Data vectorized and split")
print(f"Train encoder shape:        {tuple(train_encoder_in.shape)}")
print(f"Train decoder input shape:  {tuple(train_decoder_in.shape)}")
print(f"Train decoder target shape: {tuple(train_decoder_tgt.shape)}")
print(f"\nVal encoder shape:  {tuple(val_encoder_in.shape)}")
print(f"Test encoder shape: {tuple(test_encoder_in.shape)}")


# In[17]:

def make_loader(encoder_inputs, decoder_inputs, decoder_targets,
                batch_size=BATCH_SIZE, shuffle=True):
    """PyTorch counterpart of the original `make_transformer_dataset` helper."""
    ds = TensorDataset(encoder_inputs.long(),
                       decoder_inputs.long(),
                       decoder_targets.long())
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )

train_loader = make_loader(train_encoder_in, train_decoder_in, train_decoder_tgt, shuffle=True)
val_loader   = make_loader(val_encoder_in,   val_decoder_in,   val_decoder_tgt,   shuffle=False)
test_loader  = make_loader(test_encoder_in,  test_decoder_in,  test_decoder_tgt,  shuffle=False)

print(f"Train batches: {len(train_loader)} | "
      f"Val batches: {len(val_loader)} | "
      f"Test batches: {len(test_loader)}")


# In[18]:

# 🔹 Peek at one batch to verify shapes
for enc_in_b, dec_in_b, dec_tgt_b in train_loader:
    print("encoder_inputs shape:",  tuple(enc_in_b.shape))
    print("decoder_inputs shape:",  tuple(dec_in_b.shape))
    print("decoder_targets shape:", tuple(dec_tgt_b.shape))
    break


# ## 🔍 Step 5: Positional Encoding and Attention Blocks

# In[19]:

# Core Transformer hyperparameters
D_MODEL     = 256
NUM_HEADS   = 4
D_FF        = 512
DROPOUT_RATE = 0.1


# ### 5.1 Positional Encoding

# In[20]:

sample_pe_layer = PositionalEncoding(max_length=MAX_TARGET_LENGTH, d_model=D_MODEL)
test_input  = torch.zeros(2, 10, D_MODEL)
test_output = sample_pe_layer(test_input)
print("Positional encoding output shape:", tuple(test_output.shape))


# ### 5.2 Scaled Dot-Product Attention

# In[21]:

temp_attn = ScaledDotProductAttention()
dummy_q = torch.rand(2, 4, 10, 64)
dummy_k = torch.rand(2, 4, 10, 64)
dummy_v = torch.rand(2, 4, 10, 64)
out, weights = temp_attn(dummy_q, dummy_k, dummy_v, mask=None)
print("Attention output shape:", tuple(out.shape))
print("Attention weights shape:", tuple(weights.shape))


# ### 5.3 Multi-Head Attention

# In[22]:

temp_mha = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
dummy_q = torch.rand(2, 10, D_MODEL)
dummy_k = torch.rand(2, 10, D_MODEL)
dummy_v = torch.rand(2, 10, D_MODEL)
out, attn_w = temp_mha(v=dummy_v, k=dummy_k, q=dummy_q, mask=None)
print("Multi-head output shape:", tuple(out.shape))
print("Attention weights shape:", tuple(attn_w.shape))


# ### 5.4 Attention Masks

# In[23]:

dummy_seq = torch.tensor([[7, 3, 0, 0, 0], [5, 2, 8, 0, 0]])
pad_mask = create_padding_mask(dummy_seq)
la_mask  = create_look_ahead_mask(5)
dec_mask = create_decoder_mask(dummy_seq)
print("Padding mask shape:",     tuple(pad_mask.shape))
print("Look-ahead mask shape:",  tuple(la_mask.shape))
print("Decoder mask shape:",     tuple(dec_mask.shape))


# ## 🏗️ Step 6: Encoder & Decoder Blocks + Full Transformer Model

# ### 6.2 Encoder Block

# In[24]:

enc_block = EncoderBlock(d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF)
dummy_enc_input = torch.rand(2, 10, D_MODEL)
enc_block.eval()
enc_out = enc_block(dummy_enc_input, None)
print("Encoder block output shape:", tuple(enc_out.shape))


# ### 6.3 Decoder Block

# In[25]:

dec_block = DecoderBlock(d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF)
dummy_dec_input  = torch.rand(2, 8, D_MODEL)
dummy_enc_output = torch.rand(2, 10, D_MODEL)
dec_block.eval()
dec_out, _, _ = dec_block(dummy_dec_input, dummy_enc_output,
                          look_ahead_mask=None, padding_mask=None)
print("Decoder block output shape:", tuple(dec_out.shape))


# ### 6.5 Build and Summarize

# In[26]:

input_vocab_size  = len(input_vectorizer.get_vocabulary())
target_vocab_size = len(target_vectorizer.get_vocabulary())

transformer = build_transformer_model(
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    num_layers=2,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    dropout_rate=DROPOUT_RATE,
    max_input_len=MAX_INPUT_LENGTH,
    max_target_len=MAX_TARGET_LENGTH - 1,
).to(DEVICE)

# PyTorch equivalent of `model.summary(...)`
total_params     = sum(p.numel() for p in transformer.parameters())
trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
print(transformer)
print(f"\nTotal parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# ### 6.7 Compile the Transformer
#
# In PyTorch we don't `compile` — instead we instantiate an optimizer and
# loss function. We use `CrossEntropyLoss` (operates on raw logits, the
# PyTorch equivalent of `SparseCategoricalCrossentropy(from_logits=True)`)
# and ignore padding (index 0) when computing the loss.

# In[27]:

LEARNING_RATE = 1e-4

loss_fn   = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE)

print("✅ Transformer ready to train!")


# ## 🚂 Step 7: Training the Transformer

# In[28]:

EPOCHS = 20


# In[29]:

# We re-implement Keras-style EarlyStopping + ModelCheckpoint behaviour by hand.
checkpoint_path = "transformer_mbpp.pt"


def _token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Token-level accuracy that ignores padding (id 0) — same as Keras default."""
    preds = logits.argmax(dim=-1)
    mask  = targets != 0
    if mask.sum() == 0:
        return 0.0
    return ((preds == targets) & mask).float().sum().item() / mask.float().sum().item()


def run_epoch(model, loader, train: bool):
    """One pass over `loader`. Returns (mean_loss, mean_token_accuracy)."""
    model.train(train)
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for enc_in_b, dec_in_b, dec_tgt_b in loader:
            enc_in_b  = enc_in_b.to(DEVICE)
            dec_in_b  = dec_in_b.to(DEVICE)
            dec_tgt_b = dec_tgt_b.to(DEVICE)

            logits = model(enc_in_b, dec_in_b)                                # (B, T, V)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)),
                           dec_tgt_b.reshape(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_acc  += _token_accuracy(logits.detach(), dec_tgt_b)
            n_batches  += 1

    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


print(f"✅ Model will be saved to: {checkpoint_path}")
print("✅ Training loop helpers defined. Ready to train!")


# ### 7.1 Fit the Model

# In[30]:

history = TrainingHistory()
best_val_loss = float("inf")
best_state    = None
patience      = 3
patience_ctr  = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(transformer, train_loader, train=True)
    val_loss,   val_acc   = run_epoch(transformer, val_loader,   train=False)

    history.history["loss"].append(train_loss)
    history.history["val_loss"].append(val_loss)
    history.history["accuracy"].append(train_acc)
    history.history["val_accuracy"].append(val_acc)

    print(f"Epoch {epoch:2d} | "
          f"loss {train_loss:.4f} acc {train_acc:.4f} | "
          f"val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

    # ModelCheckpoint(save_best_only=True, monitor='val_loss')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = copy.deepcopy(transformer.state_dict())
        torch.save(best_state, checkpoint_path)
        patience_ctr = 0
        print(f"  ↳ val_loss improved to {val_loss:.4f}, saved {checkpoint_path}")
    else:
        patience_ctr += 1
        if patience_ctr >= patience:
            # EarlyStopping(restore_best_weights=True)
            print(f"  ↳ no improvement for {patience} epochs — early stopping.")
            break

# Restore best weights, just like Keras' `restore_best_weights=True`.
if best_state is not None:
    transformer.load_state_dict(best_state)
    print("✅ Best weights restored.")


# ### 7.2 Plot Training and Validation Curves

# In[31]:

plot_training_history(history)


# ### 7.3 Evaluate on the Test Set

# In[32]:

test_loss, test_accuracy = run_epoch(transformer, test_loader, train=False)
print(f"🧪 Test Loss:                 {test_loss:.4f}")
print(f"🧪 Test Accuracy (token-level): {test_accuracy:.4f}")


# ## 🧪 Step 8: Generate Code from Prompts (Greedy Decoding)

# In[33]:

target_vocab = target_vectorizer.get_vocabulary()
id_to_token  = {i: tok for i, tok in enumerate(target_vocab)}
token_to_id  = {tok: i for i, tok in enumerate(target_vocab)}


def get_token_id(token_str):
    return token_to_id.get(token_str) or token_to_id.get(token_str.lower())

start_id = get_token_id(START_TOKEN)
end_id   = get_token_id(END_TOKEN)

print("START_TOKEN id:", start_id)
print("END_TOKEN id:",   end_id)
assert start_id is not None and end_id is not None, \
    "Failed to find START/END token ids in vocab!"


# ### 8.2 Greedy Decoding for a Single Prompt

# In[34]:

def generate_greedy(prompt_text):
    """Greedy decoding: always picks the most likely next token."""
    return generate_code_for_prompt(
        prompt_text,
        model=transformer,
        input_vectorizer=input_vectorizer,
        start_id=start_id,
        end_id=end_id,
        id_to_token=id_to_token,
        max_len=MAX_TARGET_LENGTH - 1,
        device=DEVICE,
    )


# ### 8.3 Compare Ground Truth vs Generated Code

# In[35]:

test_prompts = test_inputs
test_target_strings = test_targets
print(f"Test set size: {len(test_prompts)}")
print(f"Test targets size: {len(test_target_strings)}")


# ## 📊 Step 9: BLEU Score Evaluation

# In[36]:

bleu_metric = evaluate.load("bleu")

print("✅ BLEU metric loaded successfully!")


# In[37]:

# Apply Python-language hints to test prompts.
test_prompts_python = [add_python_hint(p) for p in test_prompts]
print("✅ Python language hints added to test prompts")
print(f"   Total prompts: {len(test_prompts_python)}")
print(f"\n📝 Example transformation:")
print(f"   Original:  {test_prompts[0][:70]}...")
print(f"   With hint: {test_prompts_python[0][:70]}...")


# In[38] — Test on a single example:

def generate_code_wrapper(prompt):
    """Wrapper around `generate_best` with Part-A specific hyperparameters."""
    return generate_best(
        prompt,
        model=transformer,
        input_vectorizer=input_vectorizer,
        start_id=start_id,
        end_id=end_id,
        id_to_token=id_to_token,
        max_len=MAX_TARGET_LENGTH - 1,
        temperature=0.5,
        repetition_penalty=1.5,
        device=DEVICE,
    )


sample_prompt_original = test_prompts[0]
sample_prompt_python   = add_python_hint(sample_prompt_original)
sample_reference       = strip_special_tokens_from_target(test_target_strings[0])

print("=" * 80)
print("🧪 TESTING BLEU COMPUTATION ON SINGLE EXAMPLE")
print("=" * 80)
print(f"\n📝 Prompt (with Python hint):\n   {sample_prompt_python[:100]}...")
print(f"\n🎯 Reference code:\n   {sample_reference[:100]}...")

bleu_score, is_valid_syntax, generated_code = compute_bleu_for_code(
    sample_prompt_python,
    sample_reference,
    generate_code_wrapper,
    bleu_metric,
    model_label="Part A Transformer",
)
print(f"\n🤖 Generated code:\n   {generated_code[:100]}...")
print(f"\n📊 Results:")
print(f"   BLEU Score:   {bleu_score:.2f}")
print(f"   Syntax Valid: {is_valid_syntax}")
print("\n" + "=" * 80)


# In[40] — Evaluate the full test set with BLEU:

print("⚡ Generating all test code (sequential decoding)...")
parta_generated = sequential_generate_codes(
    test_prompts_python,
    generate_fn=generate_code_wrapper,
    desc="Part A generation",
)

# Strip [START]/[END] tokens from references before scoring.
test_target_clean = [strip_special_tokens_from_target(s) for s in test_target_strings]

bleu_scores, syntax_valid, generated_code, stats = evaluate_model_with_bleu(
    test_prompts_python,
    test_target_clean,
    None,
    bleu_metric,
    model_label="Part A Transformer",
    generated_codes=parta_generated,
)

examples_with_scores = [
    {
        "idx": i,
        "prompt":       test_prompts_python[i],
        "reference":    test_target_clean[i],
        "generated":    generated_code[i],
        "bleu":         bleu_scores[i],
        "syntax_valid": syntax_valid[i],
    }
    for i in range(len(test_prompts_python))
]

print(f"\n💾 Results stored:")
print(f"   - bleu_scores: list of {len(bleu_scores)} BLEU scores")
print(f"   - syntax_valid: list of {len(syntax_valid)} validity booleans")
print(f"   - generated_code: list of {len(generated_code)} generated code samples")
print(f"   - stats: dict with mean, median, std, min, max, syntax stats")
print(f"   - examples_with_scores: list of {len(examples_with_scores)} example dicts")


# In[42] — Best/worst analysis:

analyze_bleu_results(
    bleu_scores,
    syntax_valid,
    generated_code,
    test_prompts_python,
    test_target_clean,
    stats,
    model_label="Part A Transformer",
)


# In[44]:

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(bleu_scores, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
ax1.axvline(stats["mean"],   color="red",   linestyle="--", linewidth=2,
            label=f"Mean: {stats['mean']:.2f}")
ax1.axvline(stats["median"], color="green", linestyle="--", linewidth=2,
            label=f"Median: {stats['median']:.2f}")
ax1.set_xlabel("BLEU Score", fontsize=12)
ax1.set_ylabel("Frequency",  fontsize=12)
ax1.set_title("Distribution of BLEU Scores", fontsize=14, fontweight="bold")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.boxplot(bleu_scores, vert=True, patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
            medianprops=dict(color="red", linewidth=2))
ax2.set_ylabel("BLEU Score", fontsize=12)
ax2.set_title("Box Plot of BLEU Scores", fontsize=14, fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

print(f"\n📊 From the visualization:")
print(f"   - Most scores cluster around {stats['median']:.2f}")
print(f"   - Range: {stats['min']:.2f} to {stats['max']:.2f}")
print(f"   - Standard deviation: {stats['std']:.2f}")


# ## 🎓 Summary
#
# This is a faithful PyTorch port of the original Keras Part A notebook.
# Functionally everything matches: same data, same architecture
# (encoder–decoder Transformer with PositionalEncoding + MHA + masking),
# same teacher-forcing training loop, same greedy / nucleus-sampling
# generation algorithm (`generate_best` mirrors the original's
# repetition-counts + temperature + softmax + top-p sampling exactly),
# same BLEU + syntax-validity evaluation.
#
# Two small fixes applied during the port:
#   - References are stripped of `[START]`/`[END]` markers before BLEU
#     scoring (the original notebook scored against unstripped strings).
#   - `add_python_hint` matches the original's "skip if 'python' is in the
#     first 20 chars" rule rather than always prepending.
#
# The only other behavioural differences come from framework defaults:
# PyTorch's Adam initialisation, dropout RNG, and weight-init schemes differ
# slightly from Keras', so absolute numbers can shift a little, but the
# qualitative story (small from-scratch models can't really do code
# generation) is the same.

# In[45]:

NOTEBOOK_END_TIME = datetime.now()
elapsed = NOTEBOOK_END_TIME - NOTEBOOK_START_TIME
total_mins, total_secs = divmod(int(elapsed.total_seconds()), 60)
print(f"✅ Notebook complete")
print(f"Total time:  {total_mins}m {total_secs}s")
print(f"Current time: {NOTEBOOK_END_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
