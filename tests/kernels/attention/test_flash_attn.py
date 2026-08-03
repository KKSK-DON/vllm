# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

try:
    from vllm.vllm_flash_attn import (
        fa_version_unsupported_reason,
        flash_attn_varlen_func,
        is_fa_version_supported,
    )
except ImportError:
    if current_platform.is_rocm():
        pytest.skip(
            "vllm_flash_attn is not supported for vLLM on ROCm.",
            allow_module_level=True,
        )


NUM_HEADS = [(4, 4), (8, 2)]
HEAD_SIZES = [40, 72, 80, 128, 256]
BLOCK_SIZES = [16]
DTYPES = [torch.bfloat16]
QDTYPES = [None, torch.float8_e4m3fn, torch.int8]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]
SOFT_CAPS = [None]
SLIDING_WINDOWS = [None, 256]
# 0 for per-channel quantization, 1 for per-head quantization
KVQUANT_TYPES = [0, 1]


def quant(x: torch.Tensor, dims: tuple):
    x_descale = x.abs().float().amax(dim=dims, keepdim=True).clamp(min=1e-6) / 127
    x_int8 = torch.clamp(torch.round(x / x_descale), -128, 127).to(torch.int8)
    x_descale = x_descale.to(x.dtype)
    return x_descale, x_int8


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def yang_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    block_tables = block_tables.cpu()
    _, block_size, num_kv_heads, head_dim = key_cache.shape
    outputs: list[torch.Tensor] = []
    start_idx = 0

    for i in range(len(query_lens)):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : query_len + start_idx]
        # per tensor 1, 1, 1
        if q_scale is not None:
            q = q.to(q_scale.dtype) * q_scale
        q = q * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_idxs = block_tables[i, :num_kv_blocks]
        k = key_cache[block_idxs].view(-1, num_kv_heads, head_dim)
        k = k[:kv_len]
        if k_scale is not None:
            # per channel 1, 1, num_kv_heads, head_dim
            # per head 1, 1, num_kv_heads, 1
            k = k.to(k_scale.dtype) * k_scale.reshape(1, num_kv_heads, -1)
        v = value_cache[block_idxs].view(-1, num_kv_heads, head_dim)
        v = v[:kv_len]
        if v_scale is not None:
            v = v.to(v_scale.dtype) * v_scale.reshape(1, num_kv_heads, -1)

        if q.shape[1] != k.shape[1]:  # gqa
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn_score = torch.einsum("qnd,knd->nqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, kv_len - query_len + 1).bool()
        ########
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn_score = soft_cap * torch.tanh(attn_score / soft_cap)
        ########
        attn_score.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn_score, dim=-1).to(v.dtype)
        out = torch.einsum("nqk,knd->qnd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def yang_paged_attn_int8_accelerate(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    kvquant_type: int | None = None,
) -> torch.Tensor:
    block_tables = block_tables.cpu()
    _, block_size, num_kv_heads, head_dim = key_cache.shape
    outputs: list[torch.Tensor] = []
    start_idx = 0
    num_q_heads = query.shape[1]
    assert k_scale is not None and v_scale is not None
    k_scale_exp = k_scale.reshape(1, num_kv_heads, -1)
    v_scale_exp = v_scale.reshape(1, num_kv_heads, -1)
    if num_q_heads != num_kv_heads:
        k_scale_exp = torch.repeat_interleave(
            k_scale_exp, num_q_heads // num_kv_heads, dim=1
        )
        v_scale_exp = torch.repeat_interleave(
            v_scale_exp, num_q_heads // num_kv_heads, dim=1
        )
    for i in range(len(query_lens)):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : query_len + start_idx]
        # per tensor 1, 1, 1
        q = q * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_idxs = block_tables[i, :num_kv_blocks]
        k_int8 = key_cache[block_idxs].view(-1, num_kv_heads, head_dim)
        k_int8 = k_int8[:kv_len]
        v_int8 = value_cache[block_idxs].view(-1, num_kv_heads, head_dim)
        v_int8 = v_int8[:kv_len]
        # per channel 1, 1, num_kv_heads, head_dim
        # per head 1, 1, num_kv_heads, 1
        if q.shape[1] != k_int8.shape[1]:  # gqa
            k_int8 = torch.repeat_interleave(
                k_int8, q.shape[1] // k_int8.shape[1], dim=1
            )
            v_int8 = torch.repeat_interleave(
                v_int8, q.shape[1] // v_int8.shape[1], dim=1
            )
        # int8 accelerate
        # per channel scale [1, num_heads (8), head_dim (128)]
        # per head scale [1, num_heads (8), 1]
        if kvquant_type == 0:  # per channel
            q = q * k_scale_exp
            qk_descale, quantized_query = quant(q, (0, 1, 2))
            # pytorch does not support int8 so we use float32 for the einsum
            attn_score = torch.einsum(
                "qnd,knd->nqk", quantized_query.float(), k_int8.float()
            )
            attn_score *= qk_descale
            empty_mask = torch.ones(query_len, kv_len)
            mask = torch.triu(empty_mask, kv_len - query_len + 1).bool()
            ########
            if sliding_window is not None:
                sliding_window_mask = (
                    torch.triu(
                        empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                    )
                    .bool()
                    .logical_not()
                )
                mask |= sliding_window_mask
            if soft_cap is not None:
                attn_score = soft_cap * torch.tanh(attn_score / soft_cap)
            ########
            attn_score.masked_fill_(mask, float("-inf"))
            attn = torch.softmax(attn_score, dim=-1)
            out = torch.einsum("nqk,knd->qnd", attn, v_int8.float())
            out *= v_scale_exp

            outputs.append(out.to(q.dtype))
            start_idx += query_len
        else:  # kv per head
            q_descale, quantized_query = quant(q, (0, 1, 2))
            attn_score = torch.einsum(
                "qnd,knd->nqk", quantized_query.float(), k_int8.float()
            )
            attn_score = (
                attn_score * k_scale_exp.reshape(num_q_heads, 1, -1) * q_descale
            )
            empty_mask = torch.ones(query_len, kv_len)
            mask = torch.triu(empty_mask, kv_len - query_len + 1).bool()
            ########
            if sliding_window is not None:
                sliding_window_mask = (
                    torch.triu(
                        empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                    )
                    .bool()
                    .logical_not()
                )
                mask |= sliding_window_mask
            if soft_cap is not None:
                attn_score = soft_cap * torch.tanh(attn_score / soft_cap)
            ########
            attn_score.masked_fill_(mask, float("-inf"))
            attn = torch.softmax(attn_score, dim=-1)
            out = torch.einsum("nqk,knd->qnd", attn, v_int8.float())
            out *= v_scale_exp

            outputs.append(out.to(q.dtype))
            start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize("use_out", [True, False])
@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("fa_version", [2, 3])
@pytest.mark.parametrize("q_dtype", QDTYPES)
@pytest.mark.parametrize("kvquant_type", KVQUANT_TYPES)
@torch.inference_mode()
def test_varlen_with_paged_kv(
    use_out: bool,
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    fa_version: int,
    q_dtype: torch.dtype | None,
    kvquant_type: int,
) -> None:
    torch.set_default_device("cuda")
    if q_dtype != torch.int8:
        if kvquant_type != 0:
            pytest.skip("kvquant_type is only relevant for int8 quantization")
        if not is_fa_version_supported(fa_version):
            pytest.skip(
                f"Flash attention version {fa_version} not supported due "
                f'to: "{fa_version_unsupported_reason(fa_version)}"'
            )
        if q_dtype is not None and (dtype != torch.bfloat16 or fa_version == 2):
            pytest.skip(
                "Flash attention with quantized inputs is only "
                "supported on version 3 with bfloat16 base type"
            )
    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    out = torch.empty_like(query) if use_out else None

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None
    k_descale = None
    v_descale = None

    if q_dtype is not None and q_dtype != torch.int8:
        # QKV are drawn from N(0, 1): no need for a fp8 scaling factor
        maybe_quantized_query = query.to(q_dtype)
        maybe_quantized_key_cache = key_cache.to(q_dtype)
        maybe_quantized_value_cache = value_cache.to(q_dtype)

        scale_shape = (num_seqs, num_kv_heads)
        q_descale = torch.ones(scale_shape, dtype=torch.float32)
        k_descale = torch.ones(scale_shape, dtype=torch.float32)
        v_descale = torch.ones(scale_shape, dtype=torch.float32)
    elif q_dtype == torch.int8:
        # per tensor quant for query
        q_descale, maybe_quantized_query = quant(query, (0, 1, 2))
        if kvquant_type == 0:  # per channel
            k_descale, maybe_quantized_key_cache = quant(key_cache, (0, 1))
            v_descale, maybe_quantized_value_cache = quant(value_cache, (0, 1))
        else:  # quant_type == 1 per head
            k_descale, maybe_quantized_key_cache = quant(key_cache, (0, 1, 3))
            v_descale, maybe_quantized_value_cache = quant(value_cache, (0, 1, 3))

    if q_dtype != torch.int8:
        output = flash_attn_varlen_func(
            q=maybe_quantized_query,
            k=maybe_quantized_key_cache,
            v=maybe_quantized_value_cache,
            out=out,
            cu_seqlens_q=cu_query_lens,
            seqused_k=kv_lens,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=window_size,
            block_table=block_tables,
            softcap=soft_cap if soft_cap is not None else 0,
            fa_version=fa_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        output = output if not use_out else out

    if q_dtype == torch.int8:
        yang_q = maybe_quantized_query.clone()
        yang_k = maybe_quantized_key_cache.clone()
        yang_v = maybe_quantized_value_cache.clone()
        yang_q_scale, yang_k_scale, yang_v_scale = q_descale, k_descale, v_descale
        yang_int8_acc_q = query.clone()
        yang_int8_acc_k = maybe_quantized_key_cache.clone()
        yang_int8_acc_v = maybe_quantized_value_cache.clone()
        yang_int8_acc_k_scale, yang_int8_acc_v_scale = k_descale, v_descale
    else:
        # ref_paged_attn scales query in-place below; clone before it runs
        yang_q = query.clone()
        yang_k = key_cache
        yang_v = value_cache
        yang_q_scale = yang_k_scale = yang_v_scale = None

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )

    yang_out = yang_paged_attn(
        query=yang_q,
        key_cache=yang_k,
        value_cache=yang_v,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        q_scale=yang_q_scale,
        k_scale=yang_k_scale,
        v_scale=yang_v_scale,
    )
    if q_dtype == torch.int8:
        yang_int8_acc_out = yang_paged_attn_int8_accelerate(
            query=yang_int8_acc_q,
            key_cache=yang_int8_acc_k,
            value_cache=yang_int8_acc_v,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            k_scale=yang_int8_acc_k_scale,
            v_scale=yang_int8_acc_v_scale,
            kvquant_type=kvquant_type,
        )

    if q_dtype != torch.int8:
        atol, rtol = 1.5e-2, 1e-2
        if q_dtype is not None:
            atol, rtol = 1.5e-1, 1.5e-1
        (
            torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol),
            f"{torch.max(torch.abs(output - ref_output))}",
        )
    if q_dtype == torch.int8:
        atol, rtol = 1.5e-1, 1.5e-1
        torch.testing.assert_close(
            yang_int8_acc_out,
            ref_output,
            atol=atol,
            rtol=rtol,
            msg=f"{torch.max(torch.abs(ref_output - yang_int8_acc_out))}",
        )
    torch.testing.assert_close(
        yang_out,
        ref_output,
        atol=atol,
        rtol=rtol,
        msg=f"{torch.max(torch.abs(ref_output - yang_out))}",
    )
