# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch paged attention with int8 KV cache quantization.

Which variant runs is chosen by the ``YANG_ATTN_MODE`` environment variable:

``bf16``
    This same PyTorch attention on the untouched cache. Exists to prove that
    the plumbing matches the native kernel before any quantization enters.

``calibrate``
    ``bf16`` plus the observation hook in ``flash_attn.py`` that records the
    largest magnitudes on the write path and produces the calibration file.

``int8_per_channel`` / ``int8_per_head``
    **Dynamic** quantization. The scale is recomputed every step from the
    tokens this batch references, the pool keeps bf16, and values are
    converted to int8 again on every read. Correct but expensive: the work in
    one decode step grows with the context length, so a whole run costs
    O(context^2). Saves no memory.

``int8_static_per_channel`` / ``int8_static_per_head``
    **Static** quantization. The scale was measured once offline (the
    calibration file), the pool itself is torch.int8 (requires
    ``kv_cache_dtype=int8``), and every value is quantized exactly once, when
    it is written. Halves cache memory, and reads cost nothing but a multiply.

The suffix picks the granularity: ``per_channel`` keeps one scale per
(kv head, channel), ``per_head`` one per kv head.

A static scale is what makes the int8 pool possible at all: a scale that
changes every step could not decode values stored under yesterday's scale,
which is why no dynamic int8-pool mode exists.

During development a third family sat between these two: calibrated scales
with the pool still in bf16, requantizing on every read. It existed to prove
the calibrated scales alone lose no accuracy, and then as a bit-exact
reference while the int8 pool was brought up (both were required to produce
identical outputs, and did). It lives on in the ``feature/int8-kvcache``
branch (there the int8-pool modes are named ``int8_phys_*``) and is removed
here, because the int8-pool mode is equal in accuracy and strictly better in
memory and speed.
"""

import os
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

logger = init_logger(__name__)

# Symmetric quantization: the zero point is fixed at 0, so the largest
# magnitude in a group maps onto +127 and no offset has to be stored.
INT8_MIN = -128
INT8_MAX = 127
LARGEST_INT8_MAGNITUDE = 127
# Floor for the scale, so an all-zero group cannot produce a zero divisor.
MIN_SCALE = 1e-6

# Which dimensions of a [token, kv head, channel] tensor collapse into a single
# scale. Collapsing tokens alone leaves one scale per (kv head, channel);
# collapsing tokens and channels leaves one scale per kv head.
REDUCE_DIMS_FOR_GRANULARITY = {
    "per_channel": (0,),
    "per_head": (0, 2),
}

SUPPORTED_MODES = (
    "bf16",
    "calibrate",
    "int8_per_channel",
    "int8_per_head",
    "int8_static_per_channel",
    "int8_static_per_head",
)

_last_logged_mode_name: str | None = None
# Calibration file contents, loaded once per process, and the per-layer scale
# tensors already moved onto the device.
_calibration_file_contents: dict | None = None
_scale_cache: dict[
    tuple[str, str, str, torch.dtype], tuple[torch.Tensor, torch.Tensor]
] = {}


@dataclass(frozen=True)
class QuantizationMode:
    """One ``YANG_ATTN_MODE`` string split into the axes it encodes."""

    name: str
    # False for "bf16" and "calibrate", which read the cache untouched.
    quantizes_kv: bool
    # "none", "dynamic" or "calibration_file".
    scale_source: str
    # True for the int8_static_* modes, where the pool is torch.int8.
    cache_holds_int8: bool
    # "per_channel", "per_head", or None when nothing is quantized.
    granularity: str | None
    # True only for "calibrate", which records statistics on the write path.
    observes_calibration: bool


def parse_mode(mode_name: str) -> QuantizationMode:
    """Decompose a ``YANG_ATTN_MODE`` string. See the module docstring."""
    if mode_name == "bf16":
        return QuantizationMode(
            name=mode_name,
            quantizes_kv=False,
            scale_source="none",
            cache_holds_int8=False,
            granularity=None,
            observes_calibration=False,
        )
    if mode_name == "calibrate":
        return QuantizationMode(
            name=mode_name,
            quantizes_kv=False,
            scale_source="none",
            cache_holds_int8=False,
            granularity=None,
            observes_calibration=True,
        )

    for granularity in REDUCE_DIMS_FOR_GRANULARITY:
        if mode_name.endswith("_" + granularity):
            prefix = mode_name[: -len(granularity) - 1]
            break
    else:
        raise ValueError(
            f"Unsupported YANG_ATTN_MODE {mode_name!r}; expected one of "
            f"{SUPPORTED_MODES}"
        )

    if prefix == "int8":
        scale_source, cache_holds_int8 = "dynamic", False
    elif prefix == "int8_static":
        scale_source, cache_holds_int8 = "calibration_file", True
    elif prefix == "int8_phys":
        raise ValueError(
            f"YANG_ATTN_MODE {mode_name!r} was renamed on this branch: the "
            "int8_phys_* modes are now called int8_static_*"
        )
    else:
        raise ValueError(
            f"Unsupported YANG_ATTN_MODE {mode_name!r}; expected one of "
            f"{SUPPORTED_MODES}"
        )

    return QuantizationMode(
        name=mode_name,
        quantizes_kv=True,
        scale_source=scale_source,
        cache_holds_int8=cache_holds_int8,
        granularity=granularity,
        observes_calibration=False,
    )


# --------------------------------------------------------------------------
# Quantization primitives
# --------------------------------------------------------------------------


def compute_symmetric_scale(
    values: torch.Tensor, reduce_dims: tuple[int, ...]
) -> torch.Tensor:
    """Largest magnitude along ``reduce_dims`` divided by 127, kept in fp32.

    ``keepdim=True`` leaves the collapsed dimensions in place with length 1 so
    the result broadcasts back over ``values`` in the division below.
    """
    largest_magnitude = values.abs().float().amax(dim=reduce_dims, keepdim=True)
    return largest_magnitude.clamp(min=MIN_SCALE) / LARGEST_INT8_MAGNITUDE


def quantize_to_int8(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``round(value / scale)`` clamped into the int8 range.

    The division runs in fp32 deliberately. bf16 carries only 8 mantissa bits,
    and an error far below the value of the last bit is enough to push
    ``round`` onto the neighbouring integer; that cost one gsm8k question
    during development.

    The clamp almost never fires with a dynamic scale, because the scale was
    derived from the largest value present. With a calibrated scale it does
    fire: any value larger than anything seen during calibration saturates
    at 127.
    """
    quotient = torch.round(values.float() / scale.float())
    return torch.clamp(quotient, INT8_MIN, INT8_MAX).to(torch.int8)


def quantize_per_tensor(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a whole tensor against a single scale.

    Returns the scale (in the input dtype) and the int8 values. Used for the
    query, which is small enough that one scale for everything is fine.
    """
    scale = compute_symmetric_scale(values, reduce_dims=(0, 1, 2))
    int8_values = quantize_to_int8(values, scale)
    return scale.to(values.dtype), int8_values


# --------------------------------------------------------------------------
# Reading one sequence out of the paged cache
# --------------------------------------------------------------------------


def gather_sequence_kv(
    cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_index: int,
    context_len: int,
) -> torch.Tensor:
    """Collect one sequence's cached K or V in logical token order.

    ``cache`` is [num_blocks, block_size, num_kv_heads, head_dim] and a
    sequence's blocks sit at unrelated positions in it, so the block table row
    is what turns "this sequence's n-th block" into "physical block p". Two
    kinds of memory must stay out of the result, and both bit this project
    before: blocks this sequence does not own, and the unwritten tail of its
    last block. Slicing to ``context_len`` handles the second.
    """
    _, block_size, num_kv_heads, head_dim = cache.shape
    blocks_used = (context_len + block_size - 1) // block_size
    physical_blocks = block_tables[seq_index, :blocks_used]
    in_token_order = cache[physical_blocks].view(-1, num_kv_heads, head_dim)
    return in_token_order[:context_len]


def compute_dynamic_kv_scale(
    cache: torch.Tensor,
    reduce_dims: tuple[int, ...],
    context_lens: list[int],
    block_tables: torch.Tensor,
) -> torch.Tensor:
    """Scale for the dynamic modes, measured over this step's own tokens.

    The statistic covers exactly the tokens this batch references and nothing
    else. Measuring over the whole pool instead reads blocks owned by other
    layers, which in a hybrid model means reading bytes that are not bf16 at
    all; the resulting NaN scale poisoned every later step.
    """
    per_sequence = [
        gather_sequence_kv(cache, block_tables, seq_index, context_len)
        for seq_index, context_len in enumerate(context_lens)
    ]
    referenced_tokens = torch.cat(per_sequence, dim=0)
    scale = compute_symmetric_scale(referenced_tokens, reduce_dims)
    return scale.to(cache.dtype)


# --------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------


def build_attention_mask(
    query_len: int,
    context_len: int,
    device: torch.device,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """True marks the positions a query row must not attend to.

    The query rows are the *last* ``query_len`` positions of a ``context_len``
    long history, which is where the ``context_len - query_len + 1`` diagonal
    offset comes from. Decode (query_len 1), whole-prompt prefill
    (query_len == context_len) and a chunked-prefill middle piece all fall out
    of the same formula.
    """
    ones = torch.ones(query_len, context_len, device=device)
    mask = torch.triu(ones, context_len - query_len + 1).bool()
    if sliding_window is not None:
        outside_window = (
            torch.triu(ones, diagonal=context_len - (query_len + sliding_window) + 1)
            .bool()
            .logical_not()
        )
        mask |= outside_window
    return mask


def paged_attention_dequantize_first(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    context_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    query_scale: torch.Tensor | None = None,
    key_scale: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attention that multiplies scales back in *before* the matmuls.

    Passing no scales gives plain bf16 attention, which is what the ``bf16``
    and ``calibrate`` modes use. Passing scales turns int8 back into floats
    first and then runs ordinary float attention: correct, and enough to show
    that int8 *storage* loses no accuracy, but it cannot reach int8 matmul
    hardware because both operands are floats again by then. That is what
    :func:`paged_attention_int8_matmul` is for.
    """
    block_tables = block_tables.cpu()
    num_kv_heads = key_cache.shape[2]
    outputs: list[torch.Tensor] = []
    first_query_index = 0

    for seq_index, query_len in enumerate(query_lens):
        context_len = context_lens[seq_index]
        q = query[first_query_index : first_query_index + query_len]
        if query_scale is not None:
            q = q.to(query_scale.dtype) * query_scale
        q = q * scale

        k = gather_sequence_kv(key_cache, block_tables, seq_index, context_len)
        v = gather_sequence_kv(value_cache, block_tables, seq_index, context_len)
        if key_scale is not None:
            k = k.to(key_scale.dtype) * key_scale.reshape(1, num_kv_heads, -1)
        if value_scale is not None:
            v = v.to(value_scale.dtype) * value_scale.reshape(1, num_kv_heads, -1)

        # Grouped-query attention: several query heads share one kv head, so
        # each kv head is repeated until the head counts line up.
        if q.shape[1] != k.shape[1]:
            queries_per_kv_head = q.shape[1] // k.shape[1]
            k = torch.repeat_interleave(k, queries_per_kv_head, dim=1)
            v = torch.repeat_interleave(v, queries_per_kv_head, dim=1)

        attn_score = torch.einsum("qnd,knd->nqk", q, k).float()
        if soft_cap is not None:
            attn_score = soft_cap * torch.tanh(attn_score / soft_cap)
        mask = build_attention_mask(
            query_len, context_len, query.device, sliding_window
        )
        attn_score.masked_fill_(mask, float("-inf"))
        attn_weights = torch.softmax(attn_score, dim=-1).to(v.dtype)

        outputs.append(torch.einsum("nqk,knd->qnd", attn_weights, v))
        first_query_index += query_len

    return torch.cat(outputs, dim=0)


def paged_attention_int8_matmul(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    context_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    granularity: str,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    cache_holds_int8: bool = False,
) -> torch.Tensor:
    """Attention whose matmuls take int8 on both sides; scales apply after.

    This is the shape a real int8 kernel has to have, because int8 tensor core
    instructions need both operands to already be int8. Writing it in PyTorch
    proves the rearrangement is exact before anyone writes CUDA. The einsums
    use fp32 only because PyTorch has no int8 matmul; fp32 represents these
    integer products and sums exactly, so it is a faithful stand-in for an
    int32 accumulator rather than an approximation.

    Where the key scale can be applied depends on the granularity, because the
    dot product sums over the channel dimension:

    * ``per_head`` keeps one scale for the whole head, so it is constant across
      that sum and can simply multiply the finished score.
    * ``per_channel`` has a different scale per channel, i.e. it varies along
      the very axis being summed, so a single factor afterwards cannot undo it.
      It is folded into the query first instead, which only regroups the
      multiplications and changes no value.

    The value side is easy at both granularities: its sum runs over tokens
    while the scale is attached to channels, so the scale is constant across
    that sum either way.

    ``cache_holds_int8`` says the pool already stores int8 (the static
    modes), so the conversion step is skipped entirely.
    """
    assert granularity in REDUCE_DIMS_FOR_GRANULARITY, granularity
    block_tables = block_tables.cpu()
    num_kv_heads = key_cache.shape[2]
    num_query_heads = query.shape[1]
    output_dtype = query.dtype

    # Scales as stored cover kv heads; expand them to query heads so they line
    # up with the post-GQA tensors used below.
    key_scale_per_query_head = key_scale.reshape(1, num_kv_heads, -1)
    value_scale_per_query_head = value_scale.reshape(1, num_kv_heads, -1)
    if num_query_heads != num_kv_heads:
        queries_per_kv_head = num_query_heads // num_kv_heads
        key_scale_per_query_head = torch.repeat_interleave(
            key_scale_per_query_head, queries_per_kv_head, dim=1
        )
        value_scale_per_query_head = torch.repeat_interleave(
            value_scale_per_query_head, queries_per_kv_head, dim=1
        )

    outputs: list[torch.Tensor] = []
    first_query_index = 0

    for seq_index, query_len in enumerate(query_lens):
        context_len = context_lens[seq_index]
        q = query[first_query_index : first_query_index + query_len] * scale

        k = gather_sequence_kv(key_cache, block_tables, seq_index, context_len)
        v = gather_sequence_kv(value_cache, block_tables, seq_index, context_len)
        if cache_holds_int8:
            k_int8, v_int8 = k, v
        else:
            k_int8 = quantize_to_int8(k, key_scale)
            v_int8 = quantize_to_int8(v, value_scale)

        if num_query_heads != k_int8.shape[1]:
            queries_per_kv_head = num_query_heads // k_int8.shape[1]
            k_int8 = torch.repeat_interleave(k_int8, queries_per_kv_head, dim=1)
            v_int8 = torch.repeat_interleave(v_int8, queries_per_kv_head, dim=1)

        if granularity == "per_channel":
            # Fold the key's per-channel scale into the query, then quantize
            # the query. Both operands of the einsum end up int8 and a single
            # per-tensor factor is all that is left to reapply.
            q_scaled_by_key = q * key_scale_per_query_head
            query_scale, q_int8 = quantize_per_tensor(q_scaled_by_key)
        else:
            query_scale, q_int8 = quantize_per_tensor(q)

        attn_score = torch.einsum("qnd,knd->nqk", q_int8.float(), k_int8.float())
        if granularity == "per_channel":
            attn_score *= query_scale
        else:
            attn_score = (
                attn_score
                * key_scale_per_query_head.reshape(num_query_heads, 1, -1)
                * query_scale
            )

        if soft_cap is not None:
            attn_score = soft_cap * torch.tanh(attn_score / soft_cap)
        mask = build_attention_mask(
            query_len, context_len, query.device, sliding_window
        )
        attn_score.masked_fill_(mask, float("-inf"))
        attn_weights = torch.softmax(attn_score, dim=-1)

        out = torch.einsum("nqk,knd->qnd", attn_weights, v_int8.float())
        out *= value_scale_per_query_head

        outputs.append(out.to(output_dtype))
        first_query_index += query_len

    return torch.cat(outputs, dim=0)


# --------------------------------------------------------------------------
# Calibrated scales and the int8-pool write path
# --------------------------------------------------------------------------


def load_calibrated_scales(
    layer_name: str,
    granularity: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read this layer's key and value scales from the calibration file.

    The file is produced by the observation hook in ``flash_attn.py`` and holds
    eight tables per layer: the measured largest magnitudes at both
    granularities, plus the scales already derived from them. Only the scales
    are read here; the magnitudes are kept so the scales can be re-derived with
    a different rule without rerunning calibration.

    Results are cached per (layer, granularity, device, dtype) because the file
    never changes while the process runs.
    """
    global _calibration_file_contents

    cache_key = (layer_name, granularity, str(device), dtype)
    cached = _scale_cache.get(cache_key)
    if cached is not None:
        return cached

    if _calibration_file_contents is None:
        path = os.environ.get("YANG_KV_SCALE_PATH")
        assert path and os.path.exists(path), (
            "the int8_static_* modes need YANG_KV_SCALE_PATH "
            "pointing at a calibration file"
        )
        _calibration_file_contents = torch.load(path, map_location="cpu")
        logger.info(
            "[yang_attn] loaded calibrated kv scales for %d layers from %s",
            len(_calibration_file_contents),
            path,
        )

    tables = _calibration_file_contents[layer_name]
    suffix = "_per_head" if granularity == "per_head" else ""
    key_scale = tables["k_scale" + suffix].to(device=device, dtype=dtype)
    value_scale = tables["v_scale" + suffix].to(device=device, dtype=dtype)

    _scale_cache[cache_key] = (key_scale, value_scale)
    return key_scale, value_scale


def write_int8_kv_cache(
    layer,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write path for the static (int8-pool) modes, replacing the native kernel.

    With ``kv_cache_dtype=int8`` the pool is allocated as torch.int8, so the
    new tokens are quantized here, once, and the int8 values are what the pool
    keeps. Every later read then costs nothing but a multiply.

    ``slot_mapping`` holds one destination per new token, already resolved to a
    physical position, so no block table lookup is needed on this side. Its
    length is also the number of real tokens: the key and value buffers handed
    in are reused across steps and their tails can still hold older data.
    """
    assert kv_cache.dtype == torch.int8
    mode = parse_mode(os.environ["YANG_ATTN_MODE"])
    assert mode.cache_holds_int8, mode.name
    # Every mode that stores int8 also names a granularity; see parse_mode.
    assert mode.granularity is not None

    key_scale, value_scale = load_calibrated_scales(
        layer.layer_name, mode.granularity, key.device, key.dtype
    )

    num_tokens = slot_mapping.shape[0]
    key_int8 = quantize_to_int8(key[:num_tokens], key_scale)
    value_int8 = quantize_to_int8(value[:num_tokens], value_scale)

    head_size = key.shape[-1]
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    block_size = key_cache.shape[1]
    block_index = slot_mapping // block_size
    offset_in_block = slot_mapping % block_size
    key_cache[block_index, offset_in_block] = key_int8
    value_cache[block_index, offset_in_block] = value_int8


# --------------------------------------------------------------------------
# Entry point called from FlashAttentionImpl.forward
# --------------------------------------------------------------------------


def _log_chunked_prefill_shapes(layer, query_lens: list[int], context_lens: list[int]):
    """Show which prefill shapes actually reach this operator.

    query_len == context_len is a whole-prompt prefill, 1 < query_len <
    context_len is a chunked-prefill middle piece, and query_len == 1 is
    decode (not logged, it would drown everything else). One layer only, so
    the same step is not printed eight times.
    """
    if os.environ.get("YANG_CHUNK_WITNESS") != "1":
        return
    if not any(query_len > 1 for query_len in query_lens):
        return
    if not layer.layer_name.endswith("layers.3.self_attn.attn"):
        return
    logger.info(
        "[yang_attn] chunk witness: query_lens=%s kv_lens=%s",
        query_lens[:8],
        context_lens[:8],
    )


def _log_pool_vs_referenced_magnitudes(key_cache, block_tables):
    """Probe that found the NaN: whole pool versus this batch's own blocks.

    A healthy referenced-block maximum next to a NaN pool maximum is what
    proved the bug was the statistic's range, not the quantization math.
    """
    if not os.environ.get("YANG_ATTN_DEBUG"):
        return
    referenced_blocks = torch.unique(block_tables)
    logger.info(
        "[yang_attn][debug] pool_blocks=%d used_blocks=%d "
        "K pool_absmax=%.2f used_absmax=%.2f",
        key_cache.shape[0],
        referenced_blocks.numel(),
        key_cache.abs().amax().item(),
        key_cache[referenced_blocks].abs().amax().item(),
    )


def yang_forward(
    yang_mode: str,
    self,
    layer,
    query: torch.Tensor,  # [num_tokens, num_query_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache: torch.Tensor,
    output: torch.Tensor,
    num_actual_tokens: int,
    attn_metadata: FlashAttentionMetadata,
) -> torch.Tensor:
    """Replacement for the native FlashAttention call. See the module docstring.

    ``output`` is preallocated by the caller and must be written in place: vLLM
    relies on that buffer's address staying put.
    """
    global _last_logged_mode_name

    mode = parse_mode(yang_mode)
    if _last_logged_mode_name != mode.name:
        logger.info("[yang_attn] active, mode=%s", mode.name)
        _last_logged_mode_name = mode.name

    # Features this implementation does not handle. Failing loudly beats
    # silently producing wrong numbers for a case that was never considered.
    # `causal` is also a tensor when the batch mixes causal and non-causal
    # sequences, and `is True` rejects that too.
    assert self.alibi_slopes is None
    assert self.sinks is None
    assert attn_metadata.causal is True
    assert self.sliding_window == (-1, -1)
    assert attn_metadata.mm_prefix_query_range_tensor is None
    assert attn_metadata.rswa_prefix_lens is None

    # The pool dtype and the mode have to agree, in both directions.
    if mode.cache_holds_int8:
        assert self.kv_cache_dtype == "int8", (
            f"{mode.name} needs the int8 pool: launch with kv_cache_dtype=int8"
        )
        assert key_cache.dtype == torch.int8
    else:
        assert self.kv_cache_dtype == "auto", (
            f"{mode.name} expects the default bf16 pool, got "
            f"kv_cache_dtype={self.kv_cache_dtype}"
        )

    # query_start_loc is a prefix sum over the flattened batch, so consecutive
    # differences give each sequence's token count for this step.
    query_lens = torch.diff(attn_metadata.query_start_loc).tolist()
    context_lens = attn_metadata.seq_lens.tolist()
    assert sum(query_lens) == num_actual_tokens
    block_tables = attn_metadata.block_table

    _log_chunked_prefill_shapes(layer, query_lens, context_lens)

    soft_cap = self.logits_soft_cap if self.logits_soft_cap else None

    if not mode.quantizes_kv:
        out = paged_attention_dequantize_first(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            context_lens=context_lens,
            block_tables=block_tables,
            scale=self.scale,
            soft_cap=soft_cap,
        )
    else:
        # Every quantizing mode names a granularity; see parse_mode.
        assert mode.granularity is not None
        if mode.scale_source == "calibration_file":
            key_scale, value_scale = load_calibrated_scales(
                layer.layer_name, mode.granularity, query.device, query.dtype
            )
        else:
            _log_pool_vs_referenced_magnitudes(key_cache, block_tables)
            reduce_dims = REDUCE_DIMS_FOR_GRANULARITY[mode.granularity]
            key_scale = compute_dynamic_kv_scale(
                key_cache, reduce_dims, context_lens, block_tables
            )
            value_scale = compute_dynamic_kv_scale(
                value_cache, reduce_dims, context_lens, block_tables
            )

        out = paged_attention_int8_matmul(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            context_lens=context_lens,
            block_tables=block_tables,
            scale=self.scale,
            key_scale=key_scale,
            value_scale=value_scale,
            granularity=mode.granularity,
            soft_cap=soft_cap,
            cache_holds_int8=mode.cache_holds_int8,
        )

    output[:num_actual_tokens].copy_(out.view_as(output[:num_actual_tokens]))
    return output
