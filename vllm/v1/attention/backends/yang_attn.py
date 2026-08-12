# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

logger = init_logger(__name__)

_logged_mode: str | None = None


def compute_kv_scale(
    x: torch.Tensor, dims: tuple, kv_lens: list[int], block_tables: torch.Tensor
) -> torch.Tensor:
    num_blks, blk_size, num_kv_heads, head_dim = x.shape
    xs = []
    for i in range(len(kv_lens)):
        kv_len = kv_lens[i]
        num_kv_blocks = (kv_len + blk_size - 1) // blk_size
        blk_idxs = block_tables[i, :num_kv_blocks]
        real_sub_x = x[blk_idxs].view(-1, num_kv_heads, head_dim)[:kv_len]
        xs.append(real_sub_x)
    real_x = torch.cat(xs, dim=0)

    x_descale = real_x.abs().float().amax(dim=dims, keepdim=True).clamp(min=1e-6) / 127
    # x_int8 = torch.clamp(torch.round(x / x_descale), -128, 127).to(torch.int8)
    x_descale = x_descale.to(x.dtype)
    return x_descale


def basic_quant(x: torch.Tensor, dims: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    x_descale = x.abs().float().amax(dim=dims, keepdim=True).clamp(min=1e-6) / 127
    x_int8 = torch.clamp(torch.round(x / x_descale), -128, 127).to(torch.int8)
    x_descale = x_descale.to(x.dtype)
    return x_descale, x_int8


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
        empty_mask = torch.ones(query_len, kv_len, device=query.device)
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
        k = key_cache[block_idxs].view(-1, num_kv_heads, head_dim)
        k = k[:kv_len]
        k_int8 = torch.clamp(torch.round(k.float() / k_scale.float()), -128, 127).to(
            torch.int8
        )
        v = value_cache[block_idxs].view(-1, num_kv_heads, head_dim)
        v = v[:kv_len]
        v_int8 = torch.clamp(torch.round(v.float() / v_scale.float()), -128, 127).to(
            torch.int8
        )
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
            qk_descale, quantized_query = basic_quant(q, (0, 1, 2))
            # pytorch does not support int8 so we use float32 for the einsum
            attn_score = torch.einsum(
                "qnd,knd->nqk", quantized_query.float(), k_int8.float()
            )
            attn_score *= qk_descale
            empty_mask = torch.ones(query_len, kv_len, device=query.device)
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
            q_descale, quantized_query = basic_quant(q, (0, 1, 2))
            attn_score = torch.einsum(
                "qnd,knd->nqk", quantized_query.float(), k_int8.float()
            )
            attn_score = (
                attn_score * k_scale_exp.reshape(num_q_heads, 1, -1) * q_descale
            )
            empty_mask = torch.ones(query_len, kv_len, device=query.device)
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


def yang_forward(
    yang_mode: str,  # bf16, int8_per_channel, int8_per_head
    self,
    query: torch.Tensor,  # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache: torch.Tensor,
    output: torch.Tensor,
    num_actual_tokens: int,
    attn_metadata: FlashAttentionMetadata,
) -> torch.Tensor:
    global _logged_mode
    if _logged_mode != yang_mode:
        logger.info("[yang_attn] active, mode=%s", yang_mode)
        _logged_mode = yang_mode

    assert self.alibi_slopes is None
    assert self.sinks is None
    assert attn_metadata.causal is True
    assert attn_metadata.sliding_window in (None, (-1, -1))
    assert self.sliding_window == (-1, -1)
    assert self.kv_cache_dtype == "auto"
    assert attn_metadata.mm_prefix_range_tensor is None
    assert attn_metadata.rswa_prefix_lens is None

    cu_seqlens_q = attn_metadata.query_start_loc  # prefix sum
    query_lens = torch.diff(cu_seqlens_q).tolist()
    kv_len = attn_metadata.seq_lens.tolist()
    assert sum(query_lens) == num_actual_tokens
    block_tables = attn_metadata.block_table

    if yang_mode == "bf16":
        out = yang_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_len,
            block_tables=block_tables,
            scale=self.scale,
            soft_cap=self.logits_soft_cap if self.logits_soft_cap else None,
        )
    elif yang_mode in ("int8_per_channel", "int8_per_head"):
        # kv_cache are [num_blocks, block_size, num_kv_heads, head_dim]
        # but use [len, num_kv_heads, head_dim] to compute the max value for
        # quantization, check quant method
        dims = (0,) if yang_mode == "int8_per_channel" else (0, 2)
        if os.environ.get("YANG_ATTN_DEBUG"):
            used_blocks = torch.unique(attn_metadata.block_table)
            pool_max = key_cache.abs().amax().item()
            used_max = key_cache[used_blocks].abs().amax().item()
            logger.info(
                "[yang_attn][debug] pool_blocks=%d used_blocks=%d "
                "K pool_absmax=%.2f used_absmax=%.2f",
                key_cache.shape[0],
                used_blocks.numel(),
                pool_max,
                used_max,
            )
        k_scale = compute_kv_scale(key_cache, dims, kv_len, block_tables)
        v_scale = compute_kv_scale(value_cache, dims, kv_len, block_tables)

        out = yang_paged_attn_int8_accelerate(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_len,
            block_tables=block_tables,
            scale=self.scale,
            soft_cap=self.logits_soft_cap if self.logits_soft_cap else None,
            k_scale=k_scale,
            v_scale=v_scale,
            kvquant_type=0 if yang_mode == "int8_per_channel" else 1,
        )
    else:
        raise ValueError(f"Unsupported yang_mode: {yang_mode}")
    output[:num_actual_tokens].copy_(out.view_as(output[:num_actual_tokens]))
    return output
