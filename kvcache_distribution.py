# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV Cache Distribution Analysis Script

Compare per-token-head vs per-channel INT8 quantization granularity.

Outputs (saved to OUTPUT_DIR):
  1. Per-layer KV value distribution histograms (grouped by head)
  2. Per-layer cosine similarity chart: per-token-head vs per-channel

Usage:
  pip install transformers datasets accelerate matplotlib
  python kvcache_distribution.py
"""

import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for remote GPU
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3.5-9B"
NUM_SAMPLES = 32  # text samples to analyze
MAX_SEQ_LEN = 256  # truncate length
DEVICE = "cuda"
OUTPUT_DIR = "kv_analysis_output"
# ────────────────────────────────────────────────────────────────────


# ====================================================================
# Part 1: Load model + Collect KV caches
# ====================================================================


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE,
    )
    model.eval()
    return model, tokenizer


def load_samples(tokenizer):
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for item in dataset:
        if len(item["text"].strip()) > 50:
            texts.append(item["text"])
        if len(texts) >= NUM_SAMPLES:
            break

    tokenized = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_SEQ_LEN,
            truncation=True,
        ).to(DEVICE)
        if inputs["input_ids"].shape[1] > 10:
            tokenized.append(inputs)
    return tokenized


def get_full_attention_indices(model_name):
    """Detect which layers are full_attention (have KV cache).

    For hybrid models like Qwen3.5, only a subset of layers use
    standard attention; the rest use linear attention (recurrent state).
    For pure-attention models, all layers are returned.
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    layer_types = getattr(text_config, "layer_types", None)

    if layer_types is None:
        # Pure attention model — all layers have KV cache
        return list(range(text_config.num_hidden_layers))

    indices = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
    print(
        f"  Hybrid model: {len(indices)}/{len(layer_types)} layers "
        f"are full_attention: {indices}"
    )
    return indices


def collect_kv_caches(model, samples, attn_layer_indices):
    """Run forward pass and collect KV caches from full_attention layers only.

    Returns list of dicts, one per sample:
        { "keys": [tensor, ...], "values": [tensor, ...] }

    Each tensor shape: [num_heads, seq_len, head_dim]
    (batch dim squeezed since batch_size=1)
    """
    all_kv = []
    with torch.no_grad():
        for i, inputs in enumerate(samples):
            outputs = model(**inputs, use_cache=True)
            cache = outputs.past_key_values

            keys, values = [], []
            for layer_idx in attn_layer_indices:
                layer_cache = cache.layers[layer_idx]
                # Attribute names vary across transformers versions
                k = getattr(
                    layer_cache,
                    "keys",
                    getattr(
                        layer_cache,
                        "key_cache",
                        getattr(layer_cache, "key_states", None),
                    ),
                )
                v = getattr(
                    layer_cache,
                    "values",
                    getattr(
                        layer_cache,
                        "value_cache",
                        getattr(layer_cache, "value_states", None),
                    ),
                )
                if k is None or v is None:
                    raise AttributeError(
                        f"Layer {layer_idx} ({type(layer_cache).__name__}) "
                        f"has no key/value cache. Attrs: "
                        f"{[a for a in dir(layer_cache) if not a.startswith('_')]}"
                    )
                keys.append(k.squeeze(0).cpu().float())
                values.append(v.squeeze(0).cpu().float())

            all_kv.append({"keys": keys, "values": values})
            print(f"  sample {i + 1}/{len(samples)} done (seq_len={keys[0].shape[1]})")
    return all_kv


# ====================================================================
# Part 2: Distribution Histograms
# ====================================================================


def plot_kv_histograms(all_kv, attn_layer_indices):
    """Plot KV value distribution per full_attention layer, heads overlaid."""
    sample = all_kv[0]
    num_attn_layers = len(attn_layer_indices)

    for kv_name in ["keys", "values"]:
        n_cols = 4
        n_rows = (num_attn_layers + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()
        fig.suptitle(
            f"{kv_name.upper()} Distribution by Head — full_attention layers only"
            f"\n(model: {MODEL_NAME})",
            fontsize=14,
        )

        for ax_idx in range(num_attn_layers):
            ax = axes[ax_idx]
            real_layer = attn_layer_indices[ax_idx]
            tensor = sample[kv_name][ax_idx]
            num_heads = tensor.shape[0]

            for head_idx in range(num_heads):
                head_data = tensor[head_idx].flatten().numpy()
                ax.hist(
                    head_data, bins=100, alpha=0.4, label=f"h{head_idx}", density=True
                )

            ax.set_title(f"Layer {real_layer}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend(fontsize=6)

        for idx in range(num_attn_layers, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"histogram_{kv_name}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")


# ====================================================================
# Part 3: Quantization Simulation + Cosine Similarity
# ====================================================================


def symmetric_int8_quantize_dequantize(tensor, scale):
    """Simulate INT8 symmetric quantization round-trip.

    Args:
        tensor: original float values
        scale: per-element scale (broadcastable to tensor shape)

    Returns:
        dequantized tensor (same shape as input)
    """
    quantized = torch.clamp(torch.round(tensor / scale), -128, 127)
    return quantized * scale


def quantize_per_token_head(tensor):
    """Per-token-head: one scale per (head, token), computed over head_dim.

    tensor shape: [num_heads, seq_len, head_dim]
    scale shape:  [num_heads, seq_len, 1]
    """
    absmax = tensor.abs().amax(dim=-1, keepdim=True)  # over head_dim
    scale = (absmax / 127.0).clamp(min=1e-6)
    return symmetric_int8_quantize_dequantize(tensor, scale)


def quantize_per_channel(tensor):
    """Per-channel: one scale per (head, channel), computed over seq_len.

    tensor shape: [num_heads, seq_len, head_dim]
    scale shape:  [num_heads, 1, head_dim]
    """
    absmax = tensor.abs().amax(dim=-2, keepdim=True)  # over seq_len
    scale = (absmax / 127.0).clamp(min=1e-6)
    return symmetric_int8_quantize_dequantize(tensor, scale)


def metrics_per_layer(original, dequantized):
    """Compute cosine similarity and MSE between original and dequantized.

    Both inputs shape: [num_heads, seq_len, head_dim]
    Returns: (cos_sim, mse) — both scalars
    """
    cos = F.cosine_similarity(original, dequantized, dim=-1)  # [num_heads, seq_len]
    mse = ((original - dequantized) ** 2).mean()
    return cos.mean().item(), mse.item()


def analyze_quantization(all_kv, attn_layer_indices):
    """For each full_attention layer, compare per-token-head vs per-channel.

    Averages cosine similarity and MSE across all samples.
    """
    metric_names = [
        "pth_cos_keys",
        "pc_cos_keys",
        "pth_cos_values",
        "pc_cos_values",
        "pth_mse_keys",
        "pc_mse_keys",
        "pth_mse_values",
        "pc_mse_values",
    ]
    results = {name: [] for name in metric_names}

    for ax_idx, real_layer in enumerate(attn_layer_indices):
        batch = {name: [] for name in metric_names}

        for sample in all_kv:
            key = sample["keys"][ax_idx]
            val = sample["values"][ax_idx]

            cos_k, mse_k = metrics_per_layer(key, quantize_per_token_head(key))
            cos_v, mse_v = metrics_per_layer(val, quantize_per_token_head(val))
            batch["pth_cos_keys"].append(cos_k)
            batch["pth_mse_keys"].append(mse_k)
            batch["pth_cos_values"].append(cos_v)
            batch["pth_mse_values"].append(mse_v)

            cos_k, mse_k = metrics_per_layer(key, quantize_per_channel(key))
            cos_v, mse_v = metrics_per_layer(val, quantize_per_channel(val))
            batch["pc_cos_keys"].append(cos_k)
            batch["pc_mse_keys"].append(mse_k)
            batch["pc_cos_values"].append(cos_v)
            batch["pc_mse_values"].append(mse_v)

        for name in metric_names:
            results[name].append(np.mean(batch[name]))

        print(
            f"  Layer {real_layer:2d} | "
            f"key cos: pth={results['pth_cos_keys'][-1]:.6f} "
            f"pc={results['pc_cos_keys'][-1]:.6f} | "
            f"key mse: pth={results['pth_mse_keys'][-1]:.6f} "
            f"pc={results['pc_mse_keys'][-1]:.6f}"
        )

    return results


def plot_metrics(results, attn_layer_indices):
    """Plot per-layer cosine similarity and MSE comparison."""
    layers = attn_layer_indices

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"INT8 Quantization: per-token-head vs per-channel\n"
        f"({MODEL_NAME}, {NUM_SAMPLES} samples)",
        fontsize=14,
    )

    # Row 1: Cosine Similarity (higher is better)
    for ax, kv, title in [
        (axes[0, 0], "keys", "Keys — Cosine Similarity (↑ better)"),
        (axes[0, 1], "values", "Values — Cosine Similarity (↑ better)"),
    ]:
        ax.plot(
            layers, results[f"pth_cos_{kv}"], "o-", label="per-token-head", markersize=3
        )
        ax.plot(
            layers, results[f"pc_cos_{kv}"], "s-", label="per-channel", markersize=3
        )
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Row 2: MSE (lower is better)
    for ax, kv, title in [
        (axes[1, 0], "keys", "Keys — MSE (↓ better)"),
        (axes[1, 1], "values", "Values — MSE (↓ better)"),
    ]:
        ax.plot(
            layers, results[f"pth_mse_{kv}"], "o-", label="per-token-head", markersize=3
        )
        ax.plot(
            layers, results[f"pc_mse_{kv}"], "s-", label="per-channel", markersize=3
        )
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("MSE")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "quantization_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Step 1: Loading model and collecting KV caches")
    print("=" * 60)
    attn_layer_indices = get_full_attention_indices(MODEL_NAME)
    model, tokenizer = load_model_and_tokenizer()
    samples = load_samples(tokenizer)
    print(f"  Got {len(samples)} samples")
    all_kv = collect_kv_caches(model, samples, attn_layer_indices)

    sample_shape = all_kv[0]["keys"][0].shape
    print(
        f"  {len(attn_layer_indices)} full_attention layers collected, "
        f"key shape per layer: {sample_shape} "
        f"(num_heads, seq_len, head_dim)"
    )

    # Free GPU memory — analysis runs on CPU
    del model
    torch.accelerator.empty_cache()

    print()
    print("=" * 60)
    print("Step 2: Plotting KV distribution histograms")
    print("=" * 60)
    plot_kv_histograms(all_kv, attn_layer_indices)

    print()
    print("=" * 60)
    print("Step 3: Quantization analysis (per-token-head vs per-channel)")
    print("=" * 60)
    results = analyze_quantization(all_kv, attn_layer_indices)

    print()
    print("=" * 60)
    print("Step 4: Plotting comparison charts")
    print("=" * 60)
    plot_metrics(results, attn_layer_indices)

    print()
    print("All done! Check output in:", OUTPUT_DIR)
