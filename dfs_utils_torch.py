"""
assignment3_utils_torch.py
==========================

PyTorch port of the Keras-based `assignment3_utils.py` used in MSDS 458 Assignment 3.

Provides:
- Data loading & processing for the MBPP dataset (`prepare_training_data`).
- A simple Keras-style `TextVectorization` replacement (`TextVectorizer`).
- All Transformer building blocks (positional encoding, scaled dot-product attention,
  multi-head attention, encoder/decoder blocks, full encoder–decoder model).
- Mask helpers (padding mask, look-ahead mask, decoder mask).
- Greedy and "best" (top-p + repetition penalty) generation utilities.
- BLEU + syntax-validity evaluation utilities used by Parts A, B, and C.
- Batch generation helpers for HuggingFace seq2seq models (used in Parts B & C).

This module replaces both the original `assignment3_utils.py` and the parts of
the Keras notebooks that built model components inline.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# =============================================================================
# 1. Dataset loading / preparation (MBPP)
# =============================================================================
#
# Faithful port of Section 1-3 of the original `assignment3_utils.py`:
# configuration, raw split loading, per-example extraction, and
# `prepare_training_data` which is the function the notebooks actually call.

DATASET = "MBPP"

DATASET_CONFIG = {
    "MBPP": {
        "max_train": 374,
        "max_val":   90,
        "max_test":  50,
        "name":         "mbpp",
        "config":       "full",
        "train_split":  "train",
        "val_split":    "validation",
        "test_split":   "test",
    },
    "CodeContests": {
        "max_train": 2000,
        "max_val":   200,
        "max_test":  100,
        "name":         "deepmind/code_contests",
        "config":       None,
        "train_split":  "train",
        "val_split":    "valid",
        "test_split":   "test",
    },
}


def get_dataset_config(dataset_name: Optional[str] = None) -> dict:
    """Look up the configuration block for a dataset (defaults to global DATASET)."""
    if dataset_name is None:
        dataset_name = DATASET
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Must be 'MBPP' or 'CodeContests'"
        )
    return DATASET_CONFIG[dataset_name]


def load_dataset_splits(dataset_name: Optional[str] = None) -> Dict[str, "object"]:
    """Load raw HuggingFace dataset and return `train` / `val` / `test` splits."""
    from datasets import load_dataset

    config = get_dataset_config(dataset_name)
    if config.get("config"):
        raw_dataset = load_dataset(config["name"], config["config"])
    else:
        raw_dataset = load_dataset(config["name"])

    return {
        "train": raw_dataset[config["train_split"]],
        "val":   raw_dataset[config["val_split"]],
        "test":  raw_dataset[config["test_split"]],
    }


def extract_mbpp_example(example: dict) -> Tuple[Optional[str], Optional[str]]:
    """Pull (prompt, code) out of an MBPP example dict."""
    try:
        return example["text"], example["code"]
    except (KeyError, TypeError):
        return None, None


def extract_example(example: dict, dataset_name: Optional[str] = None):
    """Dispatch to the right per-dataset extractor (MBPP only here)."""
    if dataset_name is None:
        dataset_name = DATASET
    if dataset_name == "MBPP":
        return extract_mbpp_example(example)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def process_dataset_split(split_data, dataset_name: str, max_examples: int) -> Dict[str, List[str]]:
    """Pull up to `max_examples` (prompt, code) pairs out of a split."""
    prompts: List[str] = []
    codes: List[str] = []
    count = 0
    for example in split_data:
        if count >= max_examples:
            break
        prompt, code = extract_example(example, dataset_name)
        if prompt is not None and code is not None:
            prompts.append(prompt)
            codes.append(code)
            count += 1
    return {"prompts": prompts, "code": codes}


def prepare_training_data(dataset_name: Optional[str] = None) -> Dict[str, Dict[str, List[str]]]:
    """Load + process MBPP / CodeContests into prompts/code lists for each split.

    Faithful port of `assignment3_utils.prepare_training_data` — same dataset
    caps (374 / 90 / 50 for MBPP), same return shape:

        {'train': {'prompts': [...], 'code': [...]},
         'val':   {'prompts': [...], 'code': [...]},
         'test':  {'prompts': [...], 'code': [...]}}
    """
    config = get_dataset_config(dataset_name)
    if dataset_name is None:
        dataset_name = DATASET

    dataset = load_dataset_splits(dataset_name)
    print(f"Processing {dataset_name} dataset...")

    train_data = process_dataset_split(dataset["train"], dataset_name, config["max_train"])
    print(f"  Train: {len(train_data['prompts'])} examples")

    val_data = process_dataset_split(dataset["val"], dataset_name, config["max_val"])
    print(f"  Val:   {len(val_data['prompts'])} examples")

    test_data = process_dataset_split(dataset["test"], dataset_name, config["max_test"])
    print(f"  Test:  {len(test_data['prompts'])} examples")

    return {"train": train_data, "val": val_data, "test": test_data}


# =============================================================================
# 2. TextVectorizer — minimal replacement for tf.keras.layers.TextVectorization
# =============================================================================

# Token regex matches Keras' default "whitespace + standalone punctuation" splitter
# closely enough for this assignment (we keep brackets/operators as their own tokens).
# We match `[start]` / `[end]` (case-insensitive) before falling back to the
# generic word/punctuation pattern so the special tokens survive lowercasing.
_TOKEN_PATTERN = re.compile(r"\[start\]|\[end\]|\w+|[^\s\w]", re.IGNORECASE)


def _default_split(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text)


class TextVectorizer:
    """
    Lightweight, Keras-compatible TextVectorization replacement.

    - Lowercases input by default.
    - Builds an integer vocabulary capped at `max_tokens` (reserving idx 0 for
      padding and idx 1 for OOV, matching Keras conventions).
    - Pads / truncates each sequence to `output_sequence_length`.

    Special tokens like `[START]` and `[END]` are preserved by the splitter.
    """

    PAD_IDX = 0
    OOV_IDX = 1

    def __init__(
        self,
        max_tokens: int,
        output_sequence_length: int,
        lowercase: bool = True,
    ) -> None:
        self.max_tokens = max_tokens
        self.output_sequence_length = output_sequence_length
        self.lowercase = lowercase
        self._vocab: List[str] = []
        self._token_to_id: Dict[str, int] = {}

    # ---- vocabulary construction --------------------------------------------------
    def adapt(self, texts: List[str]) -> None:
        counter: Counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))

        # Reserve 0 = "" (pad), 1 = "[UNK]" (OOV) to match Keras output.
        vocab = ["", "[UNK]"]
        # max_tokens caps the *total* vocab size including reserved tokens.
        remaining = max(0, self.max_tokens - len(vocab))
        for tok, _ in counter.most_common(remaining):
            vocab.append(tok)

        self._vocab = vocab
        self._token_to_id = {tok: i for i, tok in enumerate(vocab)}

    # ---- public API ---------------------------------------------------------------
    def get_vocabulary(self) -> List[str]:
        return list(self._vocab)

    def __call__(self, texts) -> torch.Tensor:
        """Vectorize a batch of strings (or a single string) to a (B, L) LongTensor."""
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()

        out = np.zeros((len(texts), self.output_sequence_length), dtype=np.int64)
        for i, text in enumerate(texts):
            ids = self._encode(text)
            ids = ids[: self.output_sequence_length]
            out[i, : len(ids)] = ids
        return torch.from_numpy(out)

    # ---- helpers ------------------------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return _default_split(text)

    def _encode(self, text: str) -> List[int]:
        return [self._token_to_id.get(t, self.OOV_IDX) for t in self._tokenize(text)]


# =============================================================================
# 3. Transformer building blocks (PyTorch)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, max_length: int, d_model: int) -> None:
        super().__init__()
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, L, d_model)
        return x + self.pe[:, : x.size(1)]


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention.

    Q, K, V shapes: (B, num_heads, seq_len, depth).
    `mask` is broadcastable to (B, num_heads, seq_len_q, seq_len_k); positions
    where the mask is 1 are masked out (set to -inf before softmax).
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.bool(), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention over (B, seq_len, d_model) inputs."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention()

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, d_model) -> (B, num_heads, L, depth)
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.depth).transpose(1, 2)

    def forward(
        self,
        v: torch.Tensor,
        k: torch.Tensor,
        q: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.wq(q))
        k = self._split_heads(self.wk(k))
        v = self._split_heads(self.wv(v))

        out, attn = self.attn(q, k, v, mask=mask)
        # (B, num_heads, L, depth) -> (B, L, d_model)
        b, _, l, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(b, l, self.d_model)
        return self.dense(out), attn


def point_wise_feed_forward_network(d_model: int, d_ff: int) -> nn.Module:
    """Two linear layers with ReLU in between, applied position-wise."""
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(),
        nn.Linear(d_ff, d_model),
    )


# ----- Mask helpers -----------------------------------------------------------------

def create_padding_mask(seq: torch.Tensor) -> torch.Tensor:
    """
    1 where token == 0 (padding), else 0.
    Returns shape (B, 1, 1, L) so it broadcasts over heads and query positions.
    """
    mask = (seq == 0).to(torch.float32)
    return mask.unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Upper-triangular mask of shape (size, size); 1 = mask out, 0 = keep."""
    return torch.triu(torch.ones(size, size, device=device), diagonal=1)


def create_decoder_mask(seq: torch.Tensor) -> torch.Tensor:
    """Combined look-ahead + padding mask for decoder self-attention."""
    pad_mask = create_padding_mask(seq)  # (B, 1, 1, L)
    la_mask = create_look_ahead_mask(seq.size(1), device=seq.device)  # (L, L)
    return torch.maximum(pad_mask, la_mask)


# ----- Encoder / Decoder blocks -----------------------------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_out, _ = self.mha(v=x, k=x, q=x, mask=mask)
        x = self.norm1(x + self.drop1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mha_self = MultiHeadAttention(d_model, num_heads)
        self.mha_cross = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        look_ahead_mask: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn1, w1 = self.mha_self(v=x, k=x, q=x, mask=look_ahead_mask)
        x = self.norm1(x + self.drop1(attn1))

        attn2, w2 = self.mha_cross(v=enc_output, k=enc_output, q=x, mask=padding_mask)
        x = self.norm2(x + self.drop2(attn2))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.drop3(ffn_out))
        return x, w1, w2


# ----- Full encoder / decoder / transformer -----------------------------------------

class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        input_vocab_size: int,
        max_input_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(input_vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(max_input_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        target_vocab_size: int,
        max_target_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(target_vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(max_target_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        look_ahead_mask: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x, _, _ = layer(x, enc_output, look_ahead_mask, padding_mask)
        return x


class Transformer(nn.Module):
    """
    Encoder–decoder Transformer (Vaswani et al., 2017).

    `forward(encoder_inputs, decoder_inputs)` returns logits of shape
    (batch, dec_len, target_vocab_size). The original Keras implementation
    returned softmax probabilities; here we return raw logits and use
    `nn.CrossEntropyLoss` (which expects logits) for training.
    """

    def __init__(
        self,
        input_vocab_size: int,
        target_vocab_size: int,
        num_layers: int = 2,
        d_model: int = 256,
        num_heads: int = 4,
        d_ff: int = 512,
        dropout_rate: float = 0.1,
        max_input_len: int = 128,
        max_target_len: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            num_layers, d_model, num_heads, d_ff,
            input_vocab_size, max_input_len, dropout_rate,
        )
        self.decoder = Decoder(
            num_layers, d_model, num_heads, d_ff,
            target_vocab_size, max_target_len, dropout_rate,
        )
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(
        self,
        encoder_inputs: torch.Tensor,
        decoder_inputs: torch.Tensor,
    ) -> torch.Tensor:
        enc_padding_mask = create_padding_mask(encoder_inputs)
        dec_padding_mask = create_padding_mask(encoder_inputs)
        look_ahead_mask = create_decoder_mask(decoder_inputs)

        enc_output = self.encoder(encoder_inputs, enc_padding_mask)
        dec_output = self.decoder(decoder_inputs, enc_output, look_ahead_mask, dec_padding_mask)
        return self.final_layer(dec_output)


def build_transformer_model(
    input_vocab_size: int,
    target_vocab_size: int,
    num_layers: int = 2,
    d_model: int = 256,
    num_heads: int = 4,
    d_ff: int = 512,
    dropout_rate: float = 0.1,
    max_input_len: int = 128,
    max_target_len: int = 256,
) -> Transformer:
    """Factory matching the signature of the original Keras helper."""
    return Transformer(
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout_rate=dropout_rate,
        max_input_len=max_input_len,
        max_target_len=max_target_len,
    )


# =============================================================================
# 4. Training helpers (history container + plotting)
# =============================================================================

class TrainingHistory:
    """Mimics the relevant subset of Keras' History object."""

    def __init__(self) -> None:
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
        }


def plot_training_history(history) -> None:
    """Plot training/validation loss + accuracy curves.

    Faithful port of `assignment3_utils.plot_training_history`. Accepts any
    object with a `.history` dict (Keras `History`, our `TrainingHistory`,
    or even a bare dict if you wrap it).
    """
    import matplotlib.pyplot as plt

    history_dict = history.history if hasattr(history, "history") else history

    loss     = history_dict.get("loss", [])
    val_loss = history_dict.get("val_loss")
    acc      = history_dict.get("accuracy")
    val_acc  = history_dict.get("val_accuracy")

    epochs_range = range(1, len(loss) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label="Train Loss")
    if val_loss is not None:
        plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()

    if acc is not None:
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, acc, label="Train Accuracy")
        if val_acc is not None:
            plt.plot(epochs_range, val_acc, label="Val Accuracy")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.title("Training vs. Validation Accuracy")
        plt.legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# 5. Decoding / generation utilities (Part A)
# =============================================================================

def ids_to_code_text(
    token_ids,
    id_to_token: Dict[int, str],
    start_token: str = "[START]",
    end_token: str = "[END]",
) -> str:
    """Convert a 1D array of token ids back to a code string.

    Skips padding (id 0), `[START]`, and `[END]` markers (case-insensitively).
    Mirrors the original `assignment3_utils.ids_to_code_text` exactly.
    """
    tokens: List[str] = []
    start_lower = start_token.lower()
    end_lower = end_token.lower()
    # Accept lists, tuples, numpy arrays, and torch tensors transparently.
    for tid in token_ids:
        tid_int = int(tid)
        if tid_int == 0:
            continue
        tok = id_to_token.get(tid_int, "")
        if tok.lower() in ("", start_lower, end_lower):
            continue
        tokens.append(tok)
    return " ".join(tokens)


def strip_special_tokens_from_target(
    raw_target_str: str,
    start_token: str = "[START]",
    end_token: str = "[END]",
) -> str:
    """Remove `[START]` / `[END]` markers from a raw target string."""
    return raw_target_str.replace(start_token, "").replace(end_token, "").strip()


@torch.no_grad()
def generate_code_for_prompt(
    prompt_text: str,
    model: nn.Module,
    input_vectorizer: "TextVectorizer",
    start_id: int,
    end_id: int,
    id_to_token: Dict[int, str],
    max_len: int = 99,
    start_token: str = "[START]",
    end_token: str = "[END]",
    device: Optional[torch.device] = None,
) -> str:
    """Greedy decoding (argmax at each step).

    Mirrors the original `generate_code_for_prompt` from `assignment3_utils.py`:
      - Uses a fixed-length decoder buffer of shape (1, max_len) initialised
        to zeros and writes generated tokens into it incrementally — this is
        the same shape the model was trained with under teacher forcing.
      - Stops on `[END]`.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    encoder_inputs = input_vectorizer([prompt_text]).long().to(device)        # (1, L_in)
    generated_tokens: List[int] = [start_id]

    for _ in range(1, max_len):
        current_length = len(generated_tokens)
        decoder_inputs = torch.zeros((1, max_len), dtype=torch.long, device=device)
        decoder_inputs[0, :current_length] = torch.tensor(
            generated_tokens, dtype=torch.long, device=device,
        )

        logits = model(encoder_inputs, decoder_inputs)                        # (1, max_len, V)
        next_token_logits = logits[0, current_length - 1]                     # (V,)
        next_token_id = int(torch.argmax(next_token_logits).item())
        generated_tokens.append(next_token_id)

        if next_token_id == end_id:
            break

    return ids_to_code_text(generated_tokens[1:], id_to_token, start_token, end_token)


@torch.no_grad()
def generate_best(
    prompt_text: str,
    model: nn.Module,
    input_vectorizer: "TextVectorizer",
    start_id: int,
    end_id: int,
    id_to_token: Dict[int, str],
    max_len: int = 99,
    temperature: float = 0.7,
    repetition_penalty: float = 1.3,
    top_p: float = 0.9,
    start_token: str = "[START]",
    end_token: str = "[END]",
    device: Optional[torch.device] = None,
) -> str:
    """Repetition-penalised + nucleus-(top-p)-sampled decoding.

    Faithful PyTorch port of `assignment3_utils.generate_best`:
      - Tracks a `token_counts` dict and divides each token's logit by
        ``repetition_penalty ** count`` (this is the original's exact rule —
        not the more common HuggingFace +/- rule).
      - Suppresses padding (id 0) by setting its logit to -1e9.
      - Applies temperature, then softmax, then top-p truncation, then samples.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    encoder_inputs = input_vectorizer([prompt_text]).long().to(device)
    decoder_inputs = torch.zeros((1, max_len), dtype=torch.long, device=device)
    decoder_inputs[0, 0] = start_id
    token_counts: Dict[int, int] = {}

    for t in range(1, max_len):
        logits = model(encoder_inputs, decoder_inputs)                       # (1, max_len, V)
        next_token_logits = logits[0, t - 1].detach().cpu().numpy()          # (V,)

        for token_id, count in token_counts.items():
            next_token_logits[token_id] = (
                next_token_logits[token_id] / (repetition_penalty ** count)
            )

        next_token_logits[0] = -1e9  # suppress padding
        next_token_logits = next_token_logits / max(temperature, 1e-8)
        # Numerically stable softmax in numpy.
        shifted = next_token_logits - next_token_logits.max()
        exps = np.exp(shifted)
        next_token_probs = exps / exps.sum()

        sorted_indices = np.argsort(next_token_probs)[::-1]
        sorted_probs = next_token_probs[sorted_indices]
        cumsum_probs = np.cumsum(sorted_probs)
        nucleus_size = int(np.searchsorted(cumsum_probs, top_p)) + 1
        nucleus_indices = sorted_indices[:nucleus_size]
        nucleus_probs = sorted_probs[:nucleus_size]
        nucleus_probs = nucleus_probs / nucleus_probs.sum()

        next_token_id = int(np.random.choice(nucleus_indices, p=nucleus_probs))
        token_counts[next_token_id] = token_counts.get(next_token_id, 0) + 1
        decoder_inputs[0, t] = next_token_id

        if next_token_id == end_id:
            break

    return ids_to_code_text(
        decoder_inputs[0, 1:].cpu().tolist(), id_to_token, start_token, end_token,
    )


def sequential_generate_codes(
    prompts: List[str],
    generate_fn: Callable[[str], str],
    desc: str = "Generating",
) -> List[str]:
    """Generate code one prompt at a time (used by Part A)."""
    out = []
    for p in tqdm(prompts, desc=desc):
        try:
            out.append(generate_fn(p))
        except Exception as e:  # pragma: no cover - defensive
            out.append(f"# generation error: {e}")
    return out


# =============================================================================
# 6. Batch generation for HuggingFace seq2seq models (Parts B & C)
# =============================================================================

def batch_generate_codes(
    prompts: List[str],
    model,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 128,
    num_beams: int = 1,
    device: Optional[torch.device] = None,
) -> List[str]:
    """Batched HuggingFace seq2seq generation (PartB / PartC).

    Faithful PyTorch port of `assignment3_utils.batch_generate_codes`:
      - Tokenizes inputs with `max_length=128, padding=True, truncation=True`.
      - Uses `early_stopping=True` and `no_repeat_ngram_size=2`.
      - `num_beams=1` => greedy decoding (default; fast for evaluation).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    all_generated: List[str] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Batch generation"):
            batch = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            ).to(device)
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
            for output in outputs:
                all_generated.append(tokenizer.decode(output, skip_special_tokens=True))
    return all_generated


# =============================================================================
# 7. BLEU evaluation utilities (used by Parts A, B, C)
# =============================================================================

def add_python_hint(prompt: str) -> str:
    """
    Add an explicit "Python:\\n" prefix unless the prompt already mentions
    Python in its first 20 characters.

    This matches the original `assignment3_utils.add_python_hint` exactly so
    that BLEU evaluations are comparable across the Keras and PyTorch ports.
    """
    if "python" in prompt.lower()[:20]:
        return prompt
    return f"Python:\n{prompt}"


def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def compute_bleu_for_code(
    prompt: str,
    reference_code: str,
    generate_fn: Callable[[str], str],
    bleu_metric,
    model_label: str = "model",
) -> Tuple[float, bool, str]:
    """Generate code for one prompt and score BLEU + Python syntax validity.

    Faithful port of the original — empty/whitespace-only generations get a
    BLEU of 0.0 and `is_valid_syntax=False`, BLEU is reported on a 0-100 scale,
    and we never raise on degenerate inputs.
    """
    generated_code = generate_fn(prompt)
    generated_clean = generated_code.strip()
    reference_clean = reference_code.strip()

    if not generated_clean:
        return 0.0, False, generated_code

    try:
        bleu_result = bleu_metric.compute(
            predictions=[generated_clean],
            references=[[reference_clean]],
        )
        bleu_score = float(bleu_result["bleu"]) * 100.0
    except (ZeroDivisionError, ValueError):
        bleu_score = 0.0

    is_valid_syntax = False
    try:
        ast.parse(generated_clean)
        is_valid_syntax = True
    except (SyntaxError, ValueError):
        is_valid_syntax = False

    return bleu_score, is_valid_syntax, generated_code


def evaluate_model_with_bleu(
    test_prompts: List[str],
    test_codes: List[str],
    generate_fn: Optional[Callable[[str], str]],
    bleu_metric,
    model_label: str = "model",
    generated_codes: Optional[List[str]] = None,
) -> Tuple[List[float], List[bool], List[str], Dict[str, float]]:
    """Evaluate BLEU + syntax validity on a full test set.

    Either pass `generate_fn` (called once per prompt) or `generated_codes`
    (pre-computed, e.g. via `batch_generate_codes`). Faithful port of the
    original `assignment3_utils.evaluate_model_with_bleu`, including the
    summary block printed at the end.
    """
    using_pregenerated = generated_codes is not None
    if using_pregenerated:
        print(f"🔬 Scoring {model_label.upper()} on {len(test_prompts)} test examples (pre-generated)...")
    else:
        print(f"🔬 Evaluating {model_label.upper()} model on {len(test_prompts)} test examples...")
        print("This will take ~2-3 minutes (generating code for each example)")
    print("=" * 80)

    bleu_scores: List[float] = []
    syntax_valid: List[bool] = []
    scored_codes: List[str] = []

    for i in tqdm(range(len(test_prompts)), desc=f"{model_label} BLEU Evaluation"):
        prompt = test_prompts[i]
        reference = test_codes[i]

        if using_pregenerated:
            generated = generated_codes[i]
            generated_clean = generated.strip()
            reference_clean = reference.strip()

            if not generated_clean:
                bleu_score = 0.0
                is_valid = False
            else:
                try:
                    bleu_result = bleu_metric.compute(
                        predictions=[generated_clean],
                        references=[[reference_clean]],
                    )
                    bleu_score = float(bleu_result["bleu"]) * 100.0
                except (ZeroDivisionError, ValueError):
                    bleu_score = 0.0
                try:
                    ast.parse(generated_clean)
                    is_valid = True
                except (SyntaxError, ValueError):
                    is_valid = False
        else:
            bleu_score, is_valid, generated = compute_bleu_for_code(
                prompt, reference, generate_fn, bleu_metric, model_label=model_label,
            )

        bleu_scores.append(bleu_score)
        syntax_valid.append(is_valid)
        scored_codes.append(generated)

    syntax_valid_count = int(sum(syntax_valid))
    syntax_valid_pct = (syntax_valid_count / max(1, len(syntax_valid))) * 100.0

    stats = {
        "mean":   float(np.mean(bleu_scores))   if bleu_scores else 0.0,
        "median": float(np.median(bleu_scores)) if bleu_scores else 0.0,
        "std":    float(np.std(bleu_scores))    if bleu_scores else 0.0,
        "min":    float(np.min(bleu_scores))    if bleu_scores else 0.0,
        "max":    float(np.max(bleu_scores))    if bleu_scores else 0.0,
        "syntax_valid_count": syntax_valid_count,
        "syntax_valid_pct":   syntax_valid_pct,
    }

    print("\n" + "=" * 80)
    print(f"📊 {model_label.upper()} EVALUATION RESULTS")
    print("=" * 80)
    print(f"BLEU Score (mean):        {stats['mean']:.2f}")
    print(f"BLEU Score (median):      {stats['median']:.2f}")
    print(f"BLEU Score (std dev):     {stats['std']:.2f}")
    print(f"BLEU Score (min):         {stats['min']:.2f}")
    print(f"BLEU Score (max):         {stats['max']:.2f}")
    print(f"Syntax Validity:          {syntax_valid_count}/{len(syntax_valid)} "
          f"({syntax_valid_pct:.1f}%)")
    print("=" * 80)
    print(f"\n✅ {model_label} evaluation complete!")

    return bleu_scores, syntax_valid, scored_codes, stats


def analyze_bleu_results(
    bleu_scores: List[float],
    syntax_valid: List[bool],
    generated_codes: List[str],
    test_prompts: List[str],
    test_codes: List[str],
    stats: Dict[str, float],
    model_label: str = "model",
) -> None:
    """Plot BLEU distribution + syntax-validity pie chart, and show the
    top-3 best and bottom-3 worst examples.

    Faithful port of `assignment3_utils.analyze_bleu_results`.
    """
    import matplotlib.pyplot as plt

    print("\n" + "=" * 80)
    print(f"📊 DETAILED ANALYSIS: {model_label.upper()}")
    print("=" * 80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: BLEU distribution.
    axes[0].hist(bleu_scores, bins=20, color="#3498db", alpha=0.7, edgecolor="black")
    axes[0].axvline(stats["mean"],   color="red",   linestyle="--", linewidth=2,
                    label=f"Mean: {stats['mean']:.2f}")
    axes[0].axvline(stats["median"], color="green", linestyle="--", linewidth=2,
                    label=f"Median: {stats['median']:.2f}")
    axes[0].set_xlabel("BLEU Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"{model_label.upper()} - BLEU Score Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: syntax validity pie chart.
    syntax_count = int(sum(syntax_valid))
    syntax_invalid_count = len(syntax_valid) - syntax_count
    axes[1].pie(
        [syntax_count, syntax_invalid_count],
        labels=["Valid Syntax", "Invalid Syntax"],
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
        startangle=90,
    )
    axes[1].set_title(
        f"{model_label.upper()} - Syntax Validity\n"
        f"({syntax_count}/{len(syntax_valid)} valid)"
    )

    plt.tight_layout()
    plt.show()

    # Best / worst tables.
    sorted_indices = list(np.argsort(bleu_scores)[::-1])

    print("\n" + "=" * 80)
    print(f"🏆 BEST {model_label.upper()} EXAMPLES (Highest BLEU)")
    print("=" * 80)
    for rank, idx in enumerate(sorted_indices[:3], 1):
        print(f"\n{'=' * 80}")
        print(f"Rank #{rank} - Example {idx + 1}")
        print("=" * 80)
        print(f"BLEU Score: {bleu_scores[idx]:.2f}")
        print(f"Syntax Valid: {syntax_valid[idx]}")
        print(f"\n📝 Prompt:")
        print(test_prompts[idx][:150] + ("..." if len(test_prompts[idx]) > 150 else ""))
        print(f"\n🎯 Reference:")
        print(test_codes[idx][:150] + ("..." if len(test_codes[idx]) > 150 else ""))
        print(f"\n🤖 Generated ({model_label}):")
        print(generated_codes[idx][:150] + ("..." if len(generated_codes[idx]) > 150 else ""))

    print("\n" + "=" * 80)
    print(f"💔 WORST {model_label.upper()} EXAMPLES (Lowest BLEU)")
    print("=" * 80)
    for rank, idx in enumerate(sorted_indices[-3:][::-1], 1):
        print(f"\n{'=' * 80}")
        print(f"Rank #{rank} from bottom - Example {idx + 1}")
        print("=" * 80)
        print(f"BLEU Score: {bleu_scores[idx]:.2f}")
        print(f"Syntax Valid: {syntax_valid[idx]}")
        print(f"\n📝 Prompt:")
        print(test_prompts[idx][:150] + ("..." if len(test_prompts[idx]) > 150 else ""))
        print(f"\n🎯 Reference:")
        print(test_codes[idx][:150] + ("..." if len(test_codes[idx]) > 150 else ""))
        print(f"\n🤖 Generated ({model_label}):")
        print(generated_codes[idx][:150] + ("..." if len(generated_codes[idx]) > 150 else ""))

    print("\n" + "=" * 80)
    print(f"✅ {model_label} analysis complete!")
    print("=" * 80)
