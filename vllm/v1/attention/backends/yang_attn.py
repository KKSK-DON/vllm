# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

logger = init_logger(__name__)

_logged_mode: str | None = None

# _calib_amax: dict[str, dict[str, torch.Tensor]] = {}
_static_scale_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
_static_scales: dict | None = None


# def _gather_valid_tokens(
#     x: torch.Tensor, kv_lens: list[int], block_tables: torch.Tensor
# ) -> torch.Tensor:
#     _, blk_size, num_kv_heads, head_dim = x.shape
#     xs = []
#     for i in range(len(kv_lens)):
#         kv_len = kv_lens[i]
#         num_kv_blocks = (kv_len + blk_size - 1) // blk_size
#         blk_idxs = block_tables[i, :num_kv_blocks]
#         xs.append(x[blk_idxs].view(-1, num_kv_heads, head_dim)[:kv_len])
#     return torch.cat(xs, dim=0)


# def _calibrate_observe(#
#     layer_name: str,
#     key_cache: torch.Tensor,
#     value_cache: torch.Tensor,
#     kv_lens: list[int],
#     block_tables: torch.Tensor,
# ) -> None:
#     path = os.environ.get("YANG_KV_SCALE_PATH")
#     assert path, "calibrate mode requires YANG_KV_SCALE_PATH"
#     entry = _calib_amax.setdefault(layer_name, {})
#     changed = False
#     for table_name, cache in (("k_amax", key_cache), ("v_amax", value_cache)):
#         real = _gather_valid_tokens(cache, kv_lens, block_tables)
#         new = real.abs().float().amax(dim=(0,), keepdim=True)
#         old = entry.get(table_name)
#         if old is None:
#             entry[table_name] = new
#             changed = True
#         else:
#             merged = torch.maximum(old, new)
#             if not torch.equal(merged, old):
#                 entry[table_name] = merged
#                 changed = True
#     if changed:
#         snapshot = {}
#         for layer, tables in _calib_amax.items():
#             saved = {table_name: t.cpu() for table_name, t in tables.items()}
#             # everything derivable is derived here, offline: per_head tables
#             # and ready-to-use scales, so eval runs purely read the file
#             saved["k_amax_per_head"] = saved["k_amax"].amax(dim=-1, keepdim=True)
#             saved["v_amax_per_head"] = saved["v_amax"].amax(dim=-1, keepdim=True)
#             for amax_name in list(saved):
#                 scale_name = amax_name.replace("amax", "scale")
#                 saved[scale_name] = saved[amax_name].clamp(min=1e-6) / 127
#             snapshot[layer] = saved
#         tmp = path + ".tmp"
#         torch.save(snapshot, tmp)
#         os.replace(tmp, path)


def _get_static_scales(  #
    layer_name: str, yang_mode: str, device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    global _static_scales
    cached = _static_scale_cache.get((layer_name, yang_mode))
    if cached is not None:
        return cached
    if _static_scales is None:
        path = os.environ.get("YANG_KV_SCALE_PATH")
        assert path and os.path.exists(path), (
            "int8_static modes require YANG_KV_SCALE_PATH "
            "pointing at a calibration file"
        )
        _static_scales = torch.load(path, map_location="cpu")
        logger.info(
            "[yang_attn] loaded static kv scales for %d layers from %s",
            len(_static_scales),
            path,
        )
    tables = _static_scales[layer_name]
    if yang_mode.endswith("per_head"):
        k_scale = tables["k_scale_per_head"]
        v_scale = tables["v_scale_per_head"]
    else:
        k_scale = tables["k_scale"]
        v_scale = tables["v_scale"]
    k_scale = k_scale.to(device=device, dtype=dtype)
    v_scale = v_scale.to(device=device, dtype=dtype)
    _static_scale_cache[(layer_name, yang_mode)] = (k_scale, v_scale)
    return k_scale, v_scale


def yang_static_write(
    layer,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Physical-int8 write path, replacing reshape_and_cache_flash.

    With kv_cache_dtype=int8 the pool is allocated as torch.int8, so this
    just quantizes the new K/V tokens with the calibrated static scales
    and scatters the codes in.
    """
    assert kv_cache.dtype == torch.int8
    mode = os.environ["YANG_ATTN_MODE"]
    head_size = key.shape[-1]
    k_scale, v_scale = _get_static_scales(layer.layer_name, mode, key.device, key.dtype)
    num_tokens = slot_mapping.shape[0]
    k = key[:num_tokens].float()
    v = value[:num_tokens].float()
    k_codes = torch.clamp(torch.round(k / k_scale), -128, 127).to(torch.int8)
    v_codes = torch.clamp(torch.round(v / v_scale), -128, 127).to(torch.int8)
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    block_size = key_cache.shape[1]
    block_idx = slot_mapping // block_size
    block_off = slot_mapping % block_size
    key_cache[block_idx, block_off] = k_codes
    value_cache[block_idx, block_off] = v_codes


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
    pre_quantized: bool = False,
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
        v = value_cache[block_idxs].view(-1, num_kv_heads, head_dim)
        v = v[:kv_len]
        if pre_quantized:
            # cache already holds int8 codes (physical static mode)
            k_int8, v_int8 = k, v
        else:
            k_int8 = torch.clamp(
                torch.round(k.float() / k_scale.float()), -128, 127
            ).to(torch.int8)
            v_int8 = torch.clamp(
                torch.round(v.float() / v_scale.float()), -128, 127
            ).to(torch.int8)
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
    yang_mode: str,  # bf16 / int8_per_channel / int8_per_head
    #              / calibrate / int8_static_per_channel / int8_static_per_head
    #              / int8_phys_per_channel / int8_phys_per_head
    self,
    layer,
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
    if yang_mode.startswith("int8_phys"):
        assert self.kv_cache_dtype == "int8", (
            "int8_phys modes need the int8 pool: launch with kv_cache_dtype=int8"
        )
    else:
        assert self.kv_cache_dtype == "auto"
    assert attn_metadata.mm_prefix_range_tensor is None
    assert attn_metadata.rswa_prefix_lens is None

    cu_seqlens_q = attn_metadata.query_start_loc  # prefix sum
    query_lens = torch.diff(cu_seqlens_q).tolist()
    kv_len = attn_metadata.seq_lens.tolist()
    assert sum(query_lens) == num_actual_tokens
    block_tables = attn_metadata.block_table

    # Chunk witness: prove which prefill shapes actually reach this operator.
    # query_len == kv_len -> whole-prompt prefill; 1 < query_len < kv_len ->
    # a chunked-prefill middle piece; query_len == 1 (silent) -> decode.
    if (
        os.environ.get("YANG_CHUNK_WITNESS") == "1"
        and any(qlen > 1 for qlen in query_lens)
        and layer.layer_name.endswith("layers.3.self_attn.attn")
    ):
        logger.info(
            "[yang_attn] chunk witness: query_lens=%s kv_lens=%s",
            query_lens[:8],
            kv_len[:8],
        )

    if yang_mode in ("bf16", "calibrate"):
        # if yang_mode == "calibrate":
        #     _calibrate_observe(
        #         layer.layer_name, key_cache, value_cache, kv_len, block_tables
        #     )
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
    elif yang_mode in ("int8_phys_per_channel", "int8_phys_per_head"):
        # with kv_cache_dtype=int8 the carved views arrive as torch.int8
        assert key_cache.dtype == torch.int8
        k_scale, v_scale = _get_static_scales(
            layer.layer_name, yang_mode, query.device, query.dtype
        )
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
            kvquant_type=0 if yang_mode.endswith("per_channel") else 1,
            pre_quantized=True,
        )
    elif yang_mode in ("int8_static_per_channel", "int8_static_per_head"):
        k_scale, v_scale = _get_static_scales(
            layer.layer_name, yang_mode, query.device, key_cache.dtype
        )
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
            kvquant_type=0 if yang_mode.endswith("per_channel") else 1,
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
