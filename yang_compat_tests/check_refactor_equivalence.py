# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bit-exact check that the readability refactor changed no numbers.

Loads the pre-refactor ``yang_attn.py`` (from a git ref) and the current one as
two standalone modules, feeds them identical random inputs, and requires
``torch.equal`` on every output. Anything short of bit-exact means the refactor
moved a number, which for this project would also break the guarantee that the
physical and static modes produce identical results.

Runs on CPU, no GPU needed, about a second::

    .venv/bin/python yang_compat_tests/check_refactor_equivalence.py
    .venv/bin/python yang_compat_tests/check_refactor_equivalence.py <baseline-ref>

The vllm imports in ``yang_attn.py`` are stubbed out, because the functions
being compared need nothing but torch.
"""

import subprocess
import sys
import types

import torch

BASELINE_REF = sys.argv[1] if len(sys.argv) > 1 else "feature/int8-kvcache"
MODULE_PATH = "vllm/v1/attention/backends/yang_attn.py"

STUB_HEADER = """
import os
from dataclasses import dataclass
import torch


class _StubLogger:
    def info(self, *args, **kwargs):
        pass


logger = _StubLogger()
FlashAttentionMetadata = object
"""


def repo_root() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def load_version(source_text: str, module_name: str) -> types.ModuleType:
    """Import a yang_attn.py body with its vllm dependencies replaced."""
    kept_lines = []
    for line in source_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import os", "import torch", "from dataclasses")):
            continue
        if stripped.startswith("from vllm"):
            continue
        if stripped.startswith("logger = init_logger"):
            continue
        kept_lines.append(line)
    module = types.ModuleType(module_name)
    source = STUB_HEADER + "\n".join(kept_lines)
    exec(compile(source, module_name, "exec"), module.__dict__)
    sys.modules[module_name] = module
    return module


root = repo_root()
baseline_source = subprocess.run(
    ["git", "-C", root, "show", f"{BASELINE_REF}:{MODULE_PATH}"],
    capture_output=True,
    text=True,
    check=True,
).stdout
with open(f"{root}/{MODULE_PATH}", encoding="utf-8") as current_file:
    current_source = current_file.read()

baseline = load_version(baseline_source, "yang_attn_baseline")
current = load_version(current_source, "yang_attn_current")
print(f"baseline = {BASELINE_REF}, current = working tree\n")

torch.manual_seed(0)

DTYPE = torch.bfloat16
NUM_BLOCKS, BLOCK_SIZE = 12, 16
NUM_KV_HEADS, HEAD_DIM = 2, 8
NUM_QUERY_HEADS = 4  # grouped-query attention: 2 query heads per kv head
QUERY_LENS = [3, 1, 5]  # prefill chunk, decode, prefill chunk
CONTEXT_LENS = [20, 17, 33]  # none a multiple of BLOCK_SIZE, so tails are ragged
ATTENTION_SCALE = 0.125

query = torch.randn(sum(QUERY_LENS), NUM_QUERY_HEADS, HEAD_DIM, dtype=DTYPE)
key_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE)
value_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE)
max_blocks_per_seq = (max(CONTEXT_LENS) + BLOCK_SIZE - 1) // BLOCK_SIZE
block_tables = torch.randint(
    0, NUM_BLOCKS, (len(QUERY_LENS), max_blocks_per_seq), dtype=torch.int32
)

failures: list[str] = []


def compare(label: str, baseline_out, current_out) -> None:
    identical = baseline_out.shape == current_out.shape and torch.equal(
        baseline_out, current_out
    )
    print(f"{'OK  ' if identical else 'FAIL'}  {label}")
    if identical:
        return
    failures.append(label)
    if baseline_out.shape == current_out.shape:
        largest_gap = (baseline_out.float() - current_out.float()).abs().max().item()
        print(f"        largest difference = {largest_gap}")
    else:
        print(f"        shapes {baseline_out.shape} vs {current_out.shape}")


# Scale helpers --------------------------------------------------------------
for granularity, reduce_dims in (("per_channel", (0,)), ("per_head", (0, 2))):
    compare(
        f"dynamic scale ({granularity})",
        baseline.compute_kv_scale(key_cache, reduce_dims, CONTEXT_LENS, block_tables),
        current.compute_dynamic_kv_scale(
            key_cache, reduce_dims, CONTEXT_LENS, block_tables
        ),
    )

baseline_scale, baseline_int8 = baseline.basic_quant(query, (0, 1, 2))
current_scale, current_int8 = current.quantize_per_tensor(query)
compare("per-tensor scale", baseline_scale, current_scale)
compare("per-tensor int8 values", baseline_int8, current_int8)

# Attention without quantization ---------------------------------------------
shared = dict(
    query=query,
    key_cache=key_cache,
    value_cache=value_cache,
    query_lens=QUERY_LENS,
    block_tables=block_tables,
    scale=ATTENTION_SCALE,
)
for label, extra in (
    ("bf16 attention", {}),
    ("bf16 attention, soft_cap", {"soft_cap": 30.0}),
    ("bf16 attention, sliding_window", {"sliding_window": 8}),
):
    compare(
        label,
        baseline.yang_paged_attn(kv_lens=CONTEXT_LENS, **extra, **shared),
        current.paged_attention_dequantize_first(
            context_lens=CONTEXT_LENS, **extra, **shared
        ),
    )

# Attention with int8 matmuls, every granularity and pool layout -------------
for granularity, reduce_dims, legacy_quant_type in (
    ("per_channel", (0,), 0),
    ("per_head", (0, 2), 1),
):
    key_scale = baseline.compute_kv_scale(
        key_cache, reduce_dims, CONTEXT_LENS, block_tables
    )
    value_scale = baseline.compute_kv_scale(
        value_cache, reduce_dims, CONTEXT_LENS, block_tables
    )

    for label, extra in (
        (f"int8 attention, bf16 pool ({granularity})", {}),
        (
            f"int8 attention, bf16 pool, sliding_window + soft_cap ({granularity})",
            {"sliding_window": 8, "soft_cap": 30.0},
        ),
    ):
        compare(
            label,
            baseline.yang_paged_attn_int8_accelerate(
                kv_lens=CONTEXT_LENS,
                k_scale=key_scale,
                v_scale=value_scale,
                kvquant_type=legacy_quant_type,
                **extra,
                **shared,
            ),
            current.paged_attention_int8_matmul(
                context_lens=CONTEXT_LENS,
                key_scale=key_scale,
                value_scale=value_scale,
                granularity=granularity,
                **extra,
                **shared,
            ),
        )

    # Physical modes: the pool itself already holds int8.
    int8_pool = dict(
        shared,
        key_cache=baseline.basic_quant(key_cache, (0, 1))[1],
        value_cache=baseline.basic_quant(value_cache, (0, 1))[1],
    )
    compare(
        f"int8 attention, int8 pool ({granularity})",
        baseline.yang_paged_attn_int8_accelerate(
            kv_lens=CONTEXT_LENS,
            k_scale=key_scale,
            v_scale=value_scale,
            kvquant_type=legacy_quant_type,
            pre_quantized=True,
            **int8_pool,
        ),
        current.paged_attention_int8_matmul(
            context_lens=CONTEXT_LENS,
            key_scale=key_scale,
            value_scale=value_scale,
            granularity=granularity,
            cache_holds_int8=True,
            **int8_pool,
        ),
    )

# Mode parsing ---------------------------------------------------------------
for mode_name in current.SUPPORTED_MODES:
    assert current.parse_mode(mode_name).name == mode_name
print(f"OK    parse_mode accepts all {len(current.SUPPORTED_MODES)} documented modes")

rejected = ("", "int8", "int8_per_tensor", "typo_per_head", "int8_phys_per_channel")
for malformed in rejected:
    try:
        current.parse_mode(malformed)
    except ValueError:
        continue
    failures.append(f"parse_mode accepted {malformed!r}")
    print(f"FAIL  parse_mode accepted {malformed!r}")
else:
    print("OK    parse_mode rejects malformed mode names")

print()
if failures:
    print(f"{len(failures)} mismatch(es): {failures}")
    sys.exit(1)
print("Every output is bit-identical to the pre-refactor implementation.")
