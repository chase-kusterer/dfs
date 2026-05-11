#!/usr/bin/env python
# coding: utf-8
"""
MSDS458 Assignment 3 — Part A: Building an Encoder–Decoder Transformer from Scratch
                                (Text → Code) — **PyTorch port**
"""

# In[1]:

from datetime import datetime
NOTEBOOK_START_TIME = datetime.now()

# ## 🧩 Step 1: Setup and Imports
#
# We are using:
# - **PyTorch** for model construction and training
# - **Hugging Face Datasets** to load the MBPP dataset
# - **NumPy / Pandas** for data preprocessing
# - **Matplotlib / Seaborn** for visualizations
# - **Evaluate / SacreBLEU** for evaluation metrics

# (Standardization is built into TextVectorizer — lowercasing only.)

# general dependencies
import os, sys, urllib.request
import random
import copy
from   tqdm import tqdm

# data science dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# deep learning dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.utils.data import DataLoader, TensorDataset
import optuna

# hugging face dependencies
from   datasets import load_dataset
import evaluate

from dfs_utils_torch import (
    prepare_training_data, # pulls data from hugging face
    TextVectorizer,        # lowercasing, max tokens, vectorization
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

print("Packages ready!")


# --- HYPERPARAMETER TUNING CONFIGURATION ---
config = {
    # data & Vocabulary
    "max_input_tokens" : 12000, # Range: 5000-15000 (Larger = more diverse vocab, but heavier)
    "max_target_tokens": 20000, # Range: 10000-30000 (Code usually needs higher vocab)
    "max_input_length" : 128,   # Range: 64-256 (Depends on prompt complexity)
    "max_target_length": 256,   # Range: 128-512 (Depends on expected code length)
    
    # model Architecture
    "num_layers"  : 2,   # Range: 2-6 (Depth: start small to avoid overfitting)
    "d_model"     : 256, # Range: 128-512 (Width: must be divisible by num_heads)
    "num_heads"   : 4,   # Range: 4-8 (Attention heads)
    "d_ff"        : 512, # Range: 2x to 4x d_model (Size of feed-forward layers)
    "dropout_rate": 0.1, # Range: 0.1-0.3 (Higher = more regularization)
    
    # training Loop
    "batch_size"   : 32,   # Range: 16-64 (Limited by GPU memory)
    "learning_rate": 1e-4, # Range: 5e-5 to 5e-4 (The most critical parameter)
    "epochs"       : 25,   # Range: 10-50 (Combined with EarlyStopping)
    "patience"     : 5     # Range: 3-7 (How long to wait for val_loss improvement)
}

# setting seeds
SEED = 702
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# setting processing parameters 
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


# Step 1B: Import Shared Utilities

# disabling colab components
#IN_COLAB = "google.colab" in sys.modules
#UTILS_BRANCH = "main"

# just in case I need this as I build
def fetch_github_raw(user, repo, branch, file_path, local_path):
    url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{file_path}"
    urllib.request.urlretrieve(url, local_path)
    print(f"Fetched {file_path} from branch '{branch}'.")

# tuning study setup (optuna)
def objective(trial):
    # 1. Suggest values for the current run
    current_config = {
        "num_layers":    trial.suggest_int("num_layers", 2, 6),
        "d_model":       trial.suggest_categorical("d_model", [128, 256, 512]),
        "num_heads":     trial.suggest_categorical("num_heads", [4, 8]),
        "d_ff":          trial.suggest_int("d_ff", 512, 2048, step=512),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "dropout_rate":  trial.suggest_float("dropout_rate", 0.1, 0.3),
        "batch_size":    config["batch_size"], # keep static or tune if memory allows
        "epochs":        config["epochs"],
        "patience":      config["patience"]
    }

    # CRITICAL: Transformer architectural constraint
    if current_config["d_model"] % current_config["num_heads"] != 0:
        raise optuna.exceptions.TrialPruned()

    # 2. Build model for this trial
    trial_transformer = build_transformer_model(
        input_vocab_size  = len(input_vectorizer.get_vocabulary()),
        target_vocab_size = len(target_vectorizer.get_vocabulary()),
        num_layers        = current_config["num_layers"],
        d_model           = current_config["d_model"],
        num_heads         = current_config["num_heads"],
        d_ff              = current_config["d_ff"],
        dropout_rate      = current_config["dropout_rate"],
        max_input_len     = config["max_input_length"],
        max_target_len    = config["max_target_length"] - 1,
    ).to(DEVICE)

    trial_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    trial_optimizer = torch.optim.Adam(
        trial_transformer.parameters(),
        lr    = current_config["learning_rate"],
        betas = (0.9, 0.98),
        eps   = 1e-9,
    )

    # 3. Training loop with early stopping + best-weight restoration
    best_trial_val_loss = float("inf")
    best_trial_val_acc  = 0.0
    best_trial_state    = None
    patience_ctr        = 0

    def run_trial_epoch(model, loader, train: bool):
        model.train(train)
        total_loss, total_acc, n_batches = 0.0, 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for enc_in_b, dec_in_b, dec_tgt_b in loader:
                enc_in_b  = enc_in_b.to(DEVICE)
                dec_in_b  = dec_in_b.to(DEVICE)
                dec_tgt_b = dec_tgt_b.to(DEVICE)
                logits = model(enc_in_b, dec_in_b)
                loss   = trial_loss_fn(logits.reshape(-1, logits.size(-1)),
                                       dec_tgt_b.reshape(-1))
                if train:
                    trial_optimizer.zero_grad()
                    loss.backward()
                    trial_optimizer.step()
                total_loss += loss.item()
                total_acc  += _token_accuracy(logits.detach(), dec_tgt_b)
                n_batches  += 1
        return total_loss / max(1, n_batches), total_acc / max(1, n_batches)

    for epoch in range(1, current_config["epochs"] + 1):
        run_trial_epoch(trial_transformer, train_loader, train=True)
        trial_val_loss, trial_val_acc = run_trial_epoch(trial_transformer, val_loader, train=False)

        trial.report(trial_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if trial_val_loss < best_trial_val_loss:
            best_trial_val_loss = trial_val_loss
            best_trial_val_acc  = trial_val_acc
            best_trial_state    = copy.deepcopy(trial_transformer.state_dict())
            patience_ctr        = 0
        else:
            patience_ctr += 1
            if patience_ctr >= current_config["patience"]:
                break

    # Restore best weights from this trial before reporting metrics
    if best_trial_state is not None:
        trial_transformer.load_state_dict(best_trial_state)

    # Store custom metrics in the trial's user attributes for the DataFrame
    trial.set_user_attr("val_accuracy", best_trial_val_acc)

    return best_trial_val_loss


## importing dataset ##
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
print("📋 Sample training examples:")
print("=" * 80)
for i in range(3):
    print(f"\nExample {i+1}:")
    print("-" * 80)
    print(f"📝 Prompt: {train_prompts[i][:200]}{'...' if len(train_prompts[i]) > 200 else ''}")
    print(f"💻 Code:   {train_code[i][:200]}{'...' if len(train_code[i]) > 200 else ''}")
print("=" * 80)


## Tokenization and Vocabulary (Text & Code) ##
#
# We use a small `TextVectorizer` (PyTorch port of Keras' TextVectorization)
# that lowercases input, builds a capped integer vocabulary, and pads/truncates
# every sequence to a fixed length. Index 0 is reserved for padding and index 1
# for OOV — same as Keras' default.

# 🔧 Hyperparameters for tokenization
START_TOKEN = "[START]"
END_TOKEN   = "[END]"

print("✅ Hyperparameters set:")
print(f"   Input vocab:  {config['max_input_tokens']}")
print(f"   Target vocab: {config['max_target_tokens']}")


# Add START/END tokens to target sequences.
train_inputs  = train_prompts
train_targets = [f"{START_TOKEN} {c} {END_TOKEN}" for c in train_code]

val_inputs  = val_prompts
val_targets = [f"{START_TOKEN} {c} {END_TOKEN}" for c in val_code]

test_inputs  = test_prompts
test_targets = [f"{START_TOKEN} {c} {END_TOKEN}" for c in test_code]

print("✅ Added START and END tokens to target sequences")
print(f"   Example target (first 100 chars): {train_targets[0][:100]}...")

input_texts  = train_inputs  + val_inputs  + test_inputs
target_texts = train_targets + val_targets + test_targets

print(f"Total examples for vocabulary: {len(input_texts)}")
print("\nSample description:\n", input_texts[0][:200], "...")
print("\nSample target with tokens:\n", target_texts[0][:200], "...")



# input (natural language prompt) vectorizer
input_vectorizer = TextVectorizer(
    max_tokens             = config["max_input_tokens"],
    output_sequence_length = config["max_input_length"],
)

# target (code) vectorizer
target_vectorizer = TextVectorizer(
    max_tokens             = config["max_target_tokens"],
    output_sequence_length = config["max_target_length"],
)


input_vectorizer .adapt(input_texts)
target_vectorizer.adapt(target_texts)

print("✅ Vectorizers adapted on MBPP text and code.")


# Vectorize all input and target texts
encoder_input_data = input_vectorizer(input_texts)         # (N, MAX_INPUT_LENGTH)
full_target_data   = target_vectorizer(target_texts)       # (N, MAX_TARGET_LENGTH)

decoder_input_data  = full_target_data[:, :-1]
decoder_target_data = full_target_data[:, 1:]

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

def make_loader(encoder_inputs, decoder_inputs, decoder_targets,
                batch_size = None, shuffle = False):
    """PyTorch counterpart of the original `make_transformer_dataset` helper."""
    if batch_size is None:
        batch_size = config["batch_size"]
    ds = TensorDataset(encoder_inputs.long(),
                       decoder_inputs.long(),
                       decoder_targets.long())
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )

train_loader = make_loader(train_encoder_in, train_decoder_in, train_decoder_tgt, 
                           batch_size = config["batch_size"],
                           shuffle    = False)

val_loader   = make_loader(val_encoder_in,   val_decoder_in,   val_decoder_tgt, 
                           batch_size = config["batch_size"],
                           shuffle    = False)

test_loader  = make_loader(test_encoder_in,  test_decoder_in,  test_decoder_tgt, 
                           batch_size = config["batch_size"],
                           shuffle    = False)

print(f"Train batches: {len(train_loader)} | "
      f"Val batches: {len(val_loader)} | "
      f"Test batches: {len(test_loader)}")


# 🔹 Peek at one batch to verify shapes
for enc_in_b, dec_in_b, dec_tgt_b in train_loader:
    print("encoder_inputs shape:",  tuple(enc_in_b.shape))
    print("decoder_inputs shape:",  tuple(dec_in_b.shape))
    print("decoder_targets shape:", tuple(dec_tgt_b.shape))
    break


## 🔍 Step 5: Positional Encoding and Attention Blocks
### 5.1 Positional Encoding
sample_pe_layer = PositionalEncoding(max_length = config["max_target_length"],
                                     d_model    = config["d_model"])

test_input  = torch.zeros(2, 10, config["d_model"])
test_output = sample_pe_layer(test_input)

# 5.2 Scaled Dot-Product Attention
temp_attn = ScaledDotProductAttention()
dummy_q = torch.rand(2, 4, 10, 64)
dummy_k = torch.rand(2, 4, 10, 64)
dummy_v = torch.rand(2, 4, 10, 64)
out, weights = temp_attn(dummy_q, dummy_k, dummy_v, mask=None)


# 5.3 Multi-Head Attention
temp_mha = MultiHeadAttention(d_model   = config["d_model"],
                              num_heads = config["num_heads"])

dummy_q = torch.rand(2, 10, config["d_model"])
dummy_k = torch.rand(2, 10, config["d_model"])
dummy_v = torch.rand(2, 10, config["d_model"])
out, attn_w = temp_mha(v=dummy_v, k=dummy_k, q=dummy_q, mask=None)


# 5.4 Attention Masks
dummy_seq = torch.tensor([[7, 3, 0, 0, 0], [5, 2, 8, 0, 0]])
pad_mask  = create_padding_mask(dummy_seq)
la_mask   = create_look_ahead_mask(5)
dec_mask  = create_decoder_mask(dummy_seq)


## Step 6: Encoder & Decoder Blocks + Full Transformer Model

# 6.2 Encoder Block
enc_block = EncoderBlock(d_model   = config["d_model"], 
                         num_heads = config["num_heads"], 
                         d_ff      = config["d_ff"])

dummy_enc_input = torch.rand(2, 10, config["d_model"])
enc_block.eval()
enc_out = enc_block(dummy_enc_input, None)

# 6.3 Decoder Block
dec_block = DecoderBlock(d_model   = config["d_model"], 
                         num_heads = config["num_heads"], 
                         d_ff      = config["d_ff"])

dummy_dec_input  = torch.rand(2, 8 , config["d_model"])
dummy_enc_output = torch.rand(2, 10, config["d_model"])
dec_block.eval()
dec_out, _, _    = dec_block(dummy_dec_input, dummy_enc_output,
                             look_ahead_mask = None,
                             padding_mask    = None)

# 6.5 Build and Summarize
input_vocab_size  = len(input_vectorizer.get_vocabulary())
target_vocab_size = len(target_vectorizer.get_vocabulary())

# passing hyperparameters from the config dictionary
transformer = build_transformer_model(
    input_vocab_size  = input_vocab_size,
    target_vocab_size = target_vocab_size,
    num_layers        = config["num_layers"],
    d_model           = config["d_model"],
    num_heads         = config["num_heads"],
    d_ff              = config["d_ff"],
    dropout_rate      = config["dropout_rate"],
    max_input_len     = config["max_input_length"],
    max_target_len    = config["max_target_length"] - 1,
).to(DEVICE)

# PyTorch equivalent of `model.summary(...)`
total_params     = sum(p.numel() for p in transformer.parameters())
trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
print(transformer)
print(f"\nTotal parameters:   {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# 6.7 Compile the Transformer
# defining the loss function
loss_fn = nn.CrossEntropyLoss(ignore_index = 0) 

# defining the optimizer
optimizer = torch.optim.Adam(
    transformer.parameters(),
    lr = config["learning_rate"], # Now pulls from your tuning_config
    betas = (0.9, 0.98),           # Optimized for Transformer stability
    eps = 1e-9
)


## 🚂 Step 7: Training the Transformer
# EarlyStopping + ModelCheckpoint.
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
history = TrainingHistory()
best_val_loss = float("inf")
best_state    = None
patience      = config["patience"]
EPOCHS        = config["epochs"]
patience_ctr  = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(transformer, train_loader, train=True)
    val_loss,   val_acc   = run_epoch(transformer, val_loader,   train=False)

    history.history["loss"]        .append(train_loss)
    history.history["val_loss"]    .append(val_loss)
    history.history["accuracy"]    .append(train_acc)
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
plot_training_history(history)


# ### 7.3 Evaluate on the Test Set
test_loss, test_accuracy = run_epoch(transformer, test_loader, train=False)
print(f"🧪 Test Loss:                 {test_loss:.4f}")
print(f"🧪 Test Accuracy (token-level): {test_accuracy:.4f}")


# ## 🧪 Step 8: Generate Code from Prompts (Greedy Decoding)
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
def generate_greedy(prompt_text):
    """Greedy decoding: always picks the most likely next token."""
    return generate_code_for_prompt(
        prompt_text,
        model            = transformer,
        input_vectorizer = input_vectorizer,
        start_id         = start_id,
        end_id           = end_id,
        id_to_token      = id_to_token,
        max_len          = config["max_target_length"] - 1,
        device           = DEVICE,
    )


# 8.3 Compare Ground Truth vs Generated Code
test_prompts = test_inputs
test_target_strings = test_targets
print(f"Test set size: {len(test_prompts)}")
print(f"Test targets size: {len(test_target_strings)}")


## Step 9: BLEU Score Evaluation
bleu_metric = evaluate.load("bleu")

# Apply Python-language hints to test prompts.
test_prompts_python = [add_python_hint(p) for p in test_prompts]
print("✅ Python language hints added to test prompts")
print(f"   Total prompts: {len(test_prompts_python)}")
print(f"\n📝 Example transformation:")
print(f"   Original:  {test_prompts[0][:70]}...")
print(f"   With hint: {test_prompts_python[0][:70]}...")


def generate_code_wrapper(prompt):
    """Wrapper around `generate_best` with Part-A specific hyperparameters."""
    return generate_best(
        prompt,
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


analyze_bleu_results(
    bleu_scores,
    syntax_valid,
    generated_code,
    test_prompts_python,
    test_target_clean,
    stats,
    model_label="Part A Transformer",
)



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

# Create study (minimizing validation loss)
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials = 5,
                                       n_warmup_steps   = 5),
    storage        = "sqlite:///optuna_mbpp.db",
    study_name     = "partA_base_run",
    load_if_exists = True,
)

def trial_callback(study, trial):
    print(f"  Trial {trial.number:>2d} complete | "
          f"val_loss: {trial.value:.4f} | "
          f"val_acc: {trial.user_attrs.get('val_accuracy', float('nan')):.4f} | "
          f"params: { {k: round(v, 6) if isinstance(v, float) else v for k, v in trial.params.items()} }")
    print(f"  Best so far → Trial {study.best_trial.number} | val_loss: {study.best_value:.4f}")
    print("-" * 80)

study.optimize(objective, n_trials=20, callbacks=[trial_callback])

# 3. Convert results to a DataFrame
# This includes one column for every param and every metric (user_attrs)
df_results = study.trials_dataframe()

# Clean up column names (optional: removes 'params_' and 'user_attrs_' prefixes)
df_results.columns = [c.replace('params_', '').replace('user_attrs_', '') for c in df_results.columns]

# 4. Export to CSV
export_filename = "tuning_results_mbpp.csv"
df_results.to_csv(export_filename, index = False)

print(f"✅ Tuning complete. Results exported to {export_filename}")
print(f"Best Trial: {study.best_trial.number} | Best Loss: {study.best_value:.4f}")

NOTEBOOK_END_TIME = datetime.now()
elapsed = NOTEBOOK_END_TIME - NOTEBOOK_START_TIME
total_mins, total_secs = divmod(int(elapsed.total_seconds()), 60)
print(f"✅ Notebook complete")
print(f"Total time:  {total_mins}m {total_secs}s")
print(f"Current time: {NOTEBOOK_END_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
