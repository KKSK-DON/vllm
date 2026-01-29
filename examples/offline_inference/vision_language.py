# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""

import itertools
import json
import os
import random
from contextlib import contextmanager
from dataclasses import asdict
from typing import NamedTuple

import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.lora.request import LoRARequest
from vllm.multimodal.image import convert_image_mode
from vllm.utils.argparse_utils import FlexibleArgumentParser

# ---------------------------------------------------------------------------
# GLM-4.1V position comparison: vLLM vs HuggingFace Transformers
# ---------------------------------------------------------------------------

# List capturing EVERY call to get_mrope_input_positions (including warmup).
# Each entry is {input_tokens, position_ids, mrope_delta}.
_vllm_all_captures: list[dict] = []


def _install_vllm_capture_hook():
    """Monkey-patch get_mrope_input_positions to capture input_tokens and
    position_ids produced inside vLLM.  All invocations (warmup + real
    inference) are recorded so we can match the right one later."""
    from vllm.model_executor.models.glm4_1v import Glm4vForConditionalGeneration

    _orig_fn = Glm4vForConditionalGeneration.get_mrope_input_positions

    def _hooked(self, input_tokens, mm_features):
        result = _orig_fn(self, input_tokens, mm_features)
        _vllm_all_captures.append(
            {
                "input_tokens": list(input_tokens),
                "position_ids": result[0].clone().cpu(),
                "mrope_delta": result[1],
            }
        )
        print(
            f"[capture hook] call #{len(_vllm_all_captures)}, "
            f"seq_len={len(input_tokens)}"
        )
        return result

    Glm4vForConditionalGeneration.get_mrope_input_positions = _hooked


def _find_matching_capture(prompt_token_ids: list[int]) -> dict | None:
    """Find the captured call whose input_tokens matches the real inference
    prompt_token_ids.  Falls back to the last capture if no exact match."""
    for cap in reversed(_vllm_all_captures):
        if cap["input_tokens"] == prompt_token_ids:
            return cap
    # No exact match — return the last non-warmup capture (longest sequence)
    if _vllm_all_captures:
        return max(_vllm_all_captures, key=lambda c: len(c["input_tokens"]))
    return None


def hf_get_rope_index(
    input_tokens: list[int],
    image_grid_thw: torch.LongTensor | None,
    video_grid_thw: torch.LongTensor | None,
    spatial_merge_size: int,
    image_token_id: int,
    video_start_token_id: int,
    video_end_token_id: int,
) -> tuple[torch.Tensor, int]:
    """Standalone re-implementation of HF Glm4vModel.get_rope_index for a
    single sequence (no batch dim).  Faithfully mirrors the HF transformers
    code so we can compare against the vLLM implementation without loading
    model weights."""

    llm_pos_ids_list: list[torch.Tensor] = []

    if image_grid_thw is not None or video_grid_thw is not None:
        input_token_type: list[str] = []
        video_check_flg = False
        for token in input_tokens:
            if token == video_start_token_id:
                video_check_flg = True
            elif token == video_end_token_id:
                video_check_flg = False

            if token == image_token_id and not video_check_flg:
                input_token_type.append("image")
            elif token == image_token_id and video_check_flg:
                input_token_type.append("video")
            else:
                input_token_type.append("text")

        input_type_group: list[tuple[str, int, int]] = []
        for key, group in itertools.groupby(
            enumerate(input_token_type), lambda x: x[1]
        ):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        image_index, video_index = 0, 0
        video_group_index = 0
        video_frame_num = 1

        for modality_type, start_idx, end_idx in input_type_group:
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0

            if modality_type == "image":
                t = image_grid_thw[image_index][0].item()
                h = image_grid_thw[image_index][1].item()
                w = image_grid_thw[image_index][2].item()
                llm_grid_t = t
                llm_grid_h = h // spatial_merge_size
                llm_grid_w = w // spatial_merge_size

                t_index = (
                    torch.arange(llm_grid_t)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + st_idx
                )
                image_index += 1
                video_frame_num = 1

            elif modality_type == "video":
                t = video_frame_num
                h = video_grid_thw[video_index][1].item()
                w = video_grid_thw[video_index][2].item()
                llm_grid_t = t
                llm_grid_h = h // spatial_merge_size
                llm_grid_w = w // spatial_merge_size

                for t_idx in range(llm_grid_t):
                    t_index = (
                        torch.tensor(t_idx)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(1, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(1, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + st_idx
                    )

                video_group_index += 1
                if video_group_index >= video_grid_thw[video_index][0].item():
                    video_index += 1
                    video_group_index = 0
                video_frame_num += 1

            else:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )
                video_frame_num = 1
    else:
        text_len = len(input_tokens)
        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1))

    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
    return llm_positions, mrope_position_delta


def _write_debug_file(
    filepath: str,
    label: str,
    input_ids: list[int],
    position_ids: torch.Tensor,
    tokenizer,
):
    """Write input_ids and position_ids (full, not truncated) to a file."""
    torch.set_printoptions(threshold=float("inf"), linewidth=200)
    with open(filepath, "w") as f:
        f.write(f"=== {label} ===\n\n")
        f.write(f"--- input_ids (len={len(input_ids)}) ---\n")
        f.write(json.dumps(input_ids))
        f.write("\n\n")
        f.write("--- input_tokens (decoded, one per line) ---\n")
        for idx, tid in enumerate(input_ids):
            token_str = tokenizer.decode([tid])
            f.write(f"[{idx:5d}] id={tid:6d}  token={repr(token_str)}\n")
        f.write("\n")
        f.write(f"--- position_ids (shape={list(position_ids.shape)}) ---\n")
        f.write(f"temporal: {position_ids[0].tolist()}\n")
        f.write(f"height:   {position_ids[1].tolist()}\n")
        f.write(f"width:    {position_ids[2].tolist()}\n")
    print(f"  Written to {filepath}")


def compare_glm4v_positions(
    model_name: str,
    prompt: str,
    video_data,
    video_metadata,
    mm_processor_kwargs: dict,
    vllm_prompt_token_ids: list[int],
    tokenizer,
):
    """Compare vLLM-internal vs HF-transformers input_ids and position_ids."""
    print("\n" + "=" * 70)
    print("COMPARING vLLM vs HF Transformers (GLM-4.1V video)")
    print("=" * 70)

    # ----- vLLM side (already captured via monkey-patch) -----
    print(
        f"\n  Total hook captures: {len(_vllm_all_captures)} "
        f"(includes warmup + real inference)"
    )
    cap = _find_matching_capture(vllm_prompt_token_ids)
    if cap is None:
        print("WARNING: No vLLM position_ids captured. Did the monkey-patch fire?")
        return
    vllm_input_ids = cap["input_tokens"]
    vllm_pos_ids = cap["position_ids"]
    vllm_mrope_delta = cap["mrope_delta"]
    matched = vllm_input_ids == vllm_prompt_token_ids
    print(f"  Using capture with seq_len={len(vllm_input_ids)}, exact match={matched}")

    # ----- HF side: run processor -----
    print("\nLoading HF Glm4vProcessor for comparison...")
    hf_processor = AutoProcessor.from_pretrained(model_name)

    # Build the same prompt text used by run_glm4_1v
    hf_text = prompt

    # The HF processor expects videos as list of video arrays
    hf_videos = [video_data]

    # Pass the same processing kwargs
    hf_kwargs = {}
    if "size" in mm_processor_kwargs:
        hf_kwargs["size"] = mm_processor_kwargs["size"]
    if "fps" in mm_processor_kwargs:
        hf_kwargs["fps"] = mm_processor_kwargs["fps"]

    hf_outputs = hf_processor(
        text=[hf_text],
        videos=hf_videos,
        return_tensors="pt",
        return_metadata=True,
        **hf_kwargs,
    )

    hf_input_ids = hf_outputs["input_ids"][0].tolist()
    hf_image_grid_thw = hf_outputs.get("image_grid_thw")
    hf_video_grid_thw = hf_outputs.get("video_grid_thw")

    # Get config values from the processor / model config
    hf_config = hf_processor.image_processor
    spatial_merge_size = hf_config.merge_size
    image_token_id = hf_processor.image_token_id
    # Get video boundary token IDs from tokenizer
    video_start_token_id = hf_processor.tokenizer.convert_tokens_to_ids(
        "<|begin_of_video|>"
    )
    video_end_token_id = hf_processor.tokenizer.convert_tokens_to_ids(
        "<|end_of_video|>"
    )

    print(f"  HF input_ids length: {len(hf_input_ids)}")
    print(f"  HF image_grid_thw: {hf_image_grid_thw}")
    print(f"  HF video_grid_thw: {hf_video_grid_thw}")
    print(f"  spatial_merge_size: {spatial_merge_size}")
    print(f"  image_token_id: {image_token_id}")
    print(f"  video_start_token_id: {video_start_token_id}")
    print(f"  video_end_token_id: {video_end_token_id}")

    # Compute HF position_ids
    hf_pos_ids, hf_delta = hf_get_rope_index(
        input_tokens=hf_input_ids,
        image_grid_thw=hf_image_grid_thw,
        video_grid_thw=hf_video_grid_thw,
        spatial_merge_size=spatial_merge_size,
        image_token_id=image_token_id,
        video_start_token_id=video_start_token_id,
        video_end_token_id=video_end_token_id,
    )

    # ----- Write full results to files -----
    hf_tokenizer = hf_processor.tokenizer
    out_dir = os.path.dirname(os.path.abspath(__file__))
    vllm_file = os.path.join(out_dir, "vllm_debug_output.txt")
    hf_file = os.path.join(out_dir, "hf_debug_output.txt")

    print("\nWriting full debug output...")
    _write_debug_file(vllm_file, "vLLM", vllm_input_ids, vllm_pos_ids, hf_tokenizer)
    _write_debug_file(
        hf_file, "HF Transformers", hf_input_ids, hf_pos_ids, hf_tokenizer
    )

    # ----- Compare input_ids -----
    print("\n" + "-" * 60)
    print("INPUT_IDS COMPARISON")
    print("-" * 60)
    print(f"  vLLM length: {len(vllm_input_ids)}")
    print(f"  HF   length: {len(hf_input_ids)}")

    max_len = max(len(vllm_input_ids), len(hf_input_ids))
    diff_count = 0
    for i in range(max_len):
        v_id = vllm_input_ids[i] if i < len(vllm_input_ids) else None
        h_id = hf_input_ids[i] if i < len(hf_input_ids) else None
        if v_id != h_id:
            v_tok = (
                repr(hf_tokenizer.decode([v_id])) if v_id is not None else "<MISSING>"
            )
            h_tok = (
                repr(hf_tokenizer.decode([h_id])) if h_id is not None else "<MISSING>"
            )
            print(
                f"  [{i:5d}] vLLM: id={v_id!s:>8s} {v_tok:>30s}  |  "
                f"HF: id={h_id!s:>8s} {h_tok}"
            )
            diff_count += 1
    if diff_count == 0:
        print("  ==> input_ids are IDENTICAL")
    else:
        print(f"  ==> {diff_count} positions differ")

    # ----- Compare position_ids -----
    print("\n" + "-" * 60)
    print("POSITION_IDS COMPARISON")
    print("-" * 60)
    print(f"  vLLM shape: {list(vllm_pos_ids.shape)},  mrope_delta={vllm_mrope_delta}")
    print(f"  HF   shape: {list(hf_pos_ids.shape)},  mrope_delta={hf_delta}")

    if vllm_pos_ids.shape == hf_pos_ids.shape:
        diff_mask = (vllm_pos_ids != hf_pos_ids).any(dim=0)
        num_pos_diffs = diff_mask.sum().item()
        if num_pos_diffs == 0:
            print("  ==> position_ids are IDENTICAL")
        else:
            print(f"  ==> {num_pos_diffs} columns differ")
            diff_indices = torch.nonzero(diff_mask, as_tuple=False).flatten()
            for idx in diff_indices.tolist():
                v_t, v_h, v_w = vllm_pos_ids[:, idx].tolist()
                h_t, h_h, h_w = hf_pos_ids[:, idx].tolist()
                # Show the token at this position (use vllm_input_ids if
                # available, else hf)
                tok_id = (
                    vllm_input_ids[idx]
                    if idx < len(vllm_input_ids)
                    else hf_input_ids[idx]
                )
                tok_str = repr(hf_tokenizer.decode([tok_id]))
                print(
                    f"  [{idx:5d}] token={tok_str:>20s}  "
                    f"vLLM(t={v_t},h={v_h},w={v_w})  "
                    f"HF(t={h_t},h={h_h},w={h_w})"
                )
    else:
        print("  ==> Shapes differ, cannot compare element-wise")
        print(f"      vLLM: {list(vllm_pos_ids.shape)}")
        print(f"      HF:   {list(hf_pos_ids.shape)}")

    print("=" * 70 + "\n")


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: list[int] | None = None
    lora_requests: list[LoRARequest] | None = None
    sampling_params: list[SamplingParams] | None = None


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.


# Aria
def run_aria(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "rhymes-ai/Aria"

    # NOTE: Need L40 (or equivalent) to avoid OOM
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        dtype="bfloat16",
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [
        (
            f"<|im_start|>user\n<fim_prefix><|img|><fim_suffix>{question}"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        for question in questions
    ]

    stop_token_ids = [93532, 93653, 944, 93421, 1019, 93653, 93519]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# Aya Vision
def run_aya_vision(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "CohereLabs/aya-vision-8b"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=2048,
        max_num_seqs=2,
        mm_processor_kwargs={"crop_to_patches": True},
        limit_mm_per_prompt={modality: 1},
    )
    prompts = [
        f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|><image>{question}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        for question in questions
    ]
    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Bee-8B
def run_bee(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "Open-Bee/Bee-8B-RL"

    prompts = [
        (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<image>\n{question}<|im_end|>"
            f"<|im_start|>assistant\n<think>\n"
        )
        for question in questions
    ]

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=16384,
        limit_mm_per_prompt={modality: 1},
        trust_remote_code=True,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


def run_bagel(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "ByteDance-Seed/BAGEL-7B-MoT"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [
        (
            f"<|im_start|>user\n<|image_pad|>\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# BLIP-2
def run_blip2(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa
    prompts = [f"Question: {question} Answer:" for question in questions]
    engine_args = EngineArgs(
        model="Salesforce/blip2-opt-2.7b",
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Chameleon
def run_chameleon(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [f"{question}<image>" for question in questions]
    engine_args = EngineArgs(
        model="facebook/chameleon-7b",
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


def run_command_a_vision(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "CohereLabs/command-a-vision-07-2025"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=32768,
        tensor_parallel_size=4,
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [
        f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|><|IMG_PATCH|>{question}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Deepseek-VL2
def run_deepseek_vl2(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "deepseek-ai/deepseek-vl2-tiny"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [
        f"<|User|>: <image>\n{question}\n\n<|Assistant|>:" for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


def run_deepseek_ocr(questions: list[str], modality: str) -> ModelRequestData:
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

    assert modality == "image"

    model_name = "deepseek-ai/DeepSeek-OCR"

    engine_args = EngineArgs(
        model=model_name,
        limit_mm_per_prompt={modality: 1},
        logits_processors=[NGramPerReqLogitsProcessor],
    )

    # deepseek-ocr use plain prompt template
    prompts = [f"<image>\n{question}" for question in questions]

    # The following sampling params config is taken from
    # the official Deepseek-OCR inference example.
    # (IMPORTANT) Use the custom logits processor and avoid skipping
    # special tokens for this model for the optimal OCR performance.
    sampling_params = [
        SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            # ngram logit processor args
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                # whitelist: <td>, </td>
                whitelist_token_ids={128821, 128822},
            ),
            skip_special_tokens=False,
        )
        for _ in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        sampling_params=sampling_params,
    )


# Dots-OCR
def run_dots_ocr(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [f"<|img|><|imgpad|><|endofimg|>{question}" for question in questions]
    engine_args = EngineArgs(
        model="rednote-hilab/dots.ocr",
        limit_mm_per_prompt={modality: 1},
        trust_remote_code=True,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Eagle2.5-VL
def run_eagle2_5(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "nvidia/Eagle2.5-8B"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        trust_remote_code=True,
        limit_mm_per_prompt={modality: 1},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"<image>\n{question}"}] for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for Eagle2.5 (Qwen2 based)
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# Ernie4.5-VL
def run_ernie45_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "baidu/ERNIE-4.5-VL-28B-A3B-PT"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        limit_mm_per_prompt={modality: 1},
        trust_remote_code=True,
    )

    if modality == "image":
        placeholder = "Picture 1:<|IMAGE_START|><|image@placeholder|><|IMAGE_END|>"
    elif modality == "video":
        placeholder = "Video 1:<|VIDEO_START|><|video@placeholder|><|VIDEO_END|>"

    prompts = [
        (
            f"<|begin_of_sentence|>User: {question}{placeholder}\n"
            "Assistant: <think></think>"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Fuyu
def run_fuyu(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [f"{question}\n" for question in questions]
    engine_args = EngineArgs(
        model="adept/fuyu-8b",
        max_model_len=2048,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Gemma 3
def run_gemma3(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "google/gemma-3-4b-it"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=2048,
        max_num_seqs=2,
        mm_processor_kwargs={"do_pan_and_scan": True},
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [
        (
            "<bos><start_of_turn>user\n"
            f"<start_of_image>{question}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        for question in questions
    ]
    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Gemma3N
def run_gemma3n(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "google/gemma-3n-E2B-it"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=2048,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
        enforce_eager=True,
    )

    prompts = [
        (
            "<start_of_turn>user\n"
            f"<image_soft_token>{question}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        for question in questions
    ]
    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# GLM-4v
def run_glm4v(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "zai-org/glm-4v-9b"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=2048,
        max_num_seqs=2,
        trust_remote_code=True,
        enforce_eager=True,
        hf_overrides={"architectures": ["GLM4VForCausalLM"]},
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [
        (
            "<|user|>\n<|begin_of_image|><|endoftext|><|end_of_image|>"
            f"{question}<|assistant|>"
        )
        for question in questions
    ]

    stop_token_ids = [151329, 151336, 151338]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# GLM-4.1V
def run_glm4_1v(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "zai-org/GLM-4.1V-9B-Thinking"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        mm_processor_kwargs={
            "size": {"shortest_edge": 12544, "longest_edge": 47040000},
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
        enforce_eager=True,
    )

    if modality == "image":
        placeholder = "<|begin_of_image|><|image|><|end_of_image|>"
    elif modality == "video":
        placeholder = "<|begin_of_video|><|video|><|end_of_video|>"

    prompts = [
        (
            "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n"
            f"{placeholder}"
            f"{question}<|assistant|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# GLM-4.5V
def run_glm4_5v(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "zai-org/GLM-4.5V"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        mm_processor_kwargs={
            "size": {"shortest_edge": 12544, "longest_edge": 47040000},
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
        enforce_eager=True,
        tensor_parallel_size=4,
    )

    if modality == "image":
        placeholder = "<|begin_of_image|><|image|><|end_of_image|>"
    elif modality == "video":
        placeholder = "<|begin_of_video|><|video|><|end_of_video|>"

    prompts = [
        (
            "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n"
            f"{placeholder}"
            f"{question}<|assistant|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# GLM-4.5V-FP8
def run_glm4_5v_fp8(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "zai-org/GLM-4.5V-FP8"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        mm_processor_kwargs={
            "size": {"shortest_edge": 12544, "longest_edge": 47040000},
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
        enforce_eager=True,
        tensor_parallel_size=4,
    )

    if modality == "image":
        placeholder = "<|begin_of_image|><|image|><|end_of_image|>"
    elif modality == "video":
        placeholder = "<|begin_of_video|><|video|><|end_of_video|>"

    prompts = [
        (
            "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n"
            f"{placeholder}"
            f"{question}<|assistant|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# H2OVL-Mississippi
def run_h2ovl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "h2oai/h2ovl-mississippi-800m"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={modality: 1},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"<image>\n{question}"}] for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for H2OVL-Mississippi
    # https://huggingface.co/h2oai/h2ovl-mississippi-800m
    stop_token_ids = [tokenizer.eos_token_id]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# HunyuanOCR
def run_hunyuan_vl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "tencent/HunyuanOCR"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        limit_mm_per_prompt={modality: 1},
    )

    placeholder = "<｜hy_place▁holder▁no▁100｜><｜hy_place▁holder▁no▁102｜><｜hy_place▁holder▁no▁101｜>"  # noqa: E501
    prompts = [
        f"<｜hy_begin▁of▁sentence｜>{placeholder}{question}<｜hy_User｜>"
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=None,
    )


# naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B
def run_hyperclovax_seed_vision(
    questions: list[str], modality: str
) -> ModelRequestData:
    model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192 if modality == "image" else 16384,
        limit_mm_per_prompt={modality: 1},
    )

    messages = list()
    for question in questions:
        if modality == "image":
            """
            ocr: List the words in the image in raster order.
                Even if the word order feels unnatural for reading,
                the model will handle it as long as it follows raster order.
                e.g. "Naver, CLOVA, bigshane"
            lens_keywords: List the entity names in the image.
                e.g. "iPhone"
            lens_local_keywords: List the entity names with quads in the image.
                e.g. "[0.07, 0.21, 0.92, 0.90] iPhone"
            """
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "ocr": "",
                                "lens_keywords": "",
                                "lens_local_keywords": "",
                            },
                            {
                                "type": "text",
                                "text": question,
                            },
                        ],
                    }
                ]
            )
        elif modality == "video":
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                            },
                            {
                                "type": "text",
                                "text": question,
                            },
                        ],
                    }
                ]
            )
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=None,
    )


# Idefics3-8B-Llama3
def run_idefics3(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "HuggingFaceM4/Idefics3-8B-Llama3"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        enforce_eager=True,
        # if you are running out of memory, you can reduce the "longest_edge".
        # see: https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3#model-optimizations
        mm_processor_kwargs={
            "size": {"longest_edge": 3 * 364},
        },
        limit_mm_per_prompt={modality: 1},
    )
    prompts = [
        (f"<|begin_of_text|>User:<image>{question}<end_of_utterance>\nAssistant:")
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Intern-S1
def run_interns1(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "internlm/Intern-S1-mini"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
        enforce_eager=True,
    )

    if modality == "image":
        placeholder = "<IMG_CONTEXT>"
    elif modality == "video":
        placeholder = "<video>"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"{placeholder}\n{question}"}]
        for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# InternVL
def run_internvl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "OpenGVLab/InternVL3-2B"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<image>"
    elif modality == "video":
        placeholder = "<video>"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"{placeholder}\n{question}"}]
        for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# Kanana-V
def run_kanana_v(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "kakaocorp/kanana-1.5-v-3b-instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        trust_remote_code=True,
        limit_mm_per_prompt={modality: 1},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"<image>\n{question}"}] for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Keye-VL
def run_keye_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "Kwai-Keye/Keye-VL-8B-Preview"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        trust_remote_code=True,
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        (
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Keye-VL-1.5
def run_keye_vl1_5(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "Kwai-Keye/Keye-VL-1.5-8B"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        trust_remote_code=True,
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        (
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Kimi-VL
def run_kimi_vl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [
        "<|im_user|>user<|im_middle|><|media_start|>image<|media_content|>"
        f"<|media_pad|><|media_end|>{question}<|im_end|>"
        "<|im_assistant|>assistant<|im_middle|>"
        for question in questions
    ]

    engine_args = EngineArgs(
        model="moonshotai/Kimi-VL-A3B-Instruct",
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# LightOnOCR
def run_lightonocr(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [
        "<|im_start|>system<|im_end|>\n<|im_start|>user\n<|image_pad|><|im_end|>\n<|im_start|>assistant\n"
        for _ in questions
    ]

    engine_args = EngineArgs(
        model="lightonai/LightOnOCR-1B",
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


def run_lfm2_vl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "LiquidAI/LFM2-VL-450M"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        limit_mm_per_prompt={modality: 1},
    )

    processor = AutoProcessor.from_pretrained(model_name)
    messages = [
        [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": question}],
            }
        ]
        for question in questions
    ]
    prompts = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


def run_llama4(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=4,
        tensor_parallel_size=8,
        gpu_memory_utilization=0.4,
        limit_mm_per_prompt={modality: 1},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [
        [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": f"{question}"}],
            }
        ]
        for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    stop_token_ids = None
    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# LLaVA-1.5
def run_llava(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [f"USER: <image>\n{question}\nASSISTANT:" for question in questions]

    engine_args = EngineArgs(
        model="llava-hf/llava-1.5-7b-hf",
        max_model_len=4096,
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# LLaVA-1.6/LLaVA-NeXT
def run_llava_next(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [f"[INST] <image>\n{question} [/INST]" for question in questions]
    engine_args = EngineArgs(
        model="llava-hf/llava-v1.6-mistral-7b-hf",
        max_model_len=8192,
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# LlaVA-NeXT-Video
# Currently only support for video input
def run_llava_next_video(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "video"

    prompts = [f"USER: <video>\n{question} ASSISTANT:" for question in questions]
    engine_args = EngineArgs(
        model="llava-hf/LLaVA-NeXT-Video-7B-hf",
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# LLaVA-OneVision
def run_llava_onevision(questions: list[str], modality: str) -> ModelRequestData:
    if modality == "video":
        prompts = [
            f"<|im_start|>user <video>\n{question}<|im_end|><|im_start|>assistant\n"
            for question in questions
        ]

    elif modality == "image":
        prompts = [
            f"<|im_start|>user <image>\n{question}<|im_end|><|im_start|>assistant\n"
            for question in questions
        ]

    engine_args = EngineArgs(
        model="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        max_model_len=16384,
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Mantis
def run_mantis(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    llama3_template = "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"  # noqa: E501
    prompts = [llama3_template.format(f"{question}\n<image>") for question in questions]

    engine_args = EngineArgs(
        model="TIGER-Lab/Mantis-8B-siglip-llama3",
        max_model_len=4096,
        hf_overrides={"architectures": ["MantisForConditionalGeneration"]},
        limit_mm_per_prompt={modality: 1},
    )
    stop_token_ids = [128009]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# MiniCPM-V
def run_minicpmv_base(questions: list[str], modality: str, model_name):
    assert modality in ["image", "video"]
    # If you want to use `MiniCPM-o-2_6` with audio inputs, check `audio_language.py` # noqa

    # 2.0
    # The official repo doesn't work yet, so we need to use a fork for now
    # For more details, please see: See: https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630 # noqa
    # model_name = "HwwwH/MiniCPM-V-2"

    # 2.5
    # model_name = "openbmb/MiniCPM-Llama3-V-2_5"

    # 2.6
    # model_name = "openbmb/MiniCPM-V-2_6"
    # o2.6

    # modality supports
    # 2.0: image
    # 2.5: image
    # 2.6: image, video
    # o2.6: image, video, audio
    # model_name = "openbmb/MiniCPM-o-2_6"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        trust_remote_code=True,
        limit_mm_per_prompt={modality: 1},
    )
    # NOTE The stop_token_ids are different for various versions of MiniCPM-V
    # 2.0
    # stop_token_ids = [tokenizer.eos_id]

    # 2.5
    # stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

    # 2.6 / o2.6
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    modality_placeholder = {
        "image": "(<image>./</image>)",
        "video": "(<video>./</video>)",
    }

    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"{modality_placeholder[modality]}\n{question}",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


def run_minicpmo(questions: list[str], modality: str) -> ModelRequestData:
    return run_minicpmv_base(questions, modality, "openbmb/MiniCPM-o-2_6")


def run_minicpmv(questions: list[str], modality: str) -> ModelRequestData:
    return run_minicpmv_base(questions, modality, "openbmb/MiniCPM-V-2_6")


def run_minimax_vl_01(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "MiniMaxAI/MiniMax-VL-01"

    engine_args = EngineArgs(
        model=model_name,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
        trust_remote_code=True,
        tensor_parallel_size=8,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [
        [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": question}],
            }
        ]
        for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Mistral-3 HF-format
def run_mistral3(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    # NOTE: Need L40 (or equivalent) to avoid OOM
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        tensor_parallel_size=2,
        limit_mm_per_prompt={modality: 1},
        ignore_patterns=["consolidated.safetensors"],
    )

    prompts = [f"<s>[INST]{question}\n[IMG][/INST]" for question in questions]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Molmo
def run_molmo(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "allenai/Molmo-7B-D-0924"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [
        f"<|im_start|>user <image>\n{question}<|im_end|><|im_start|>assistant\n"
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Molmo2
def run_molmo2(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "allenai/Molmo2-8B"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={modality: 1},
        max_num_batched_tokens=36864,
    )

    if modality == "image":
        placeholder = "<|image|>"
    elif modality == "video":
        placeholder = "<|video|>"
    else:
        raise ValueError(f"Unsupported modality for molmo2: {modality}")

    prompts = [
        f"{placeholder}<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Nemontron_VL
def run_nemotron_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={modality: 1},
    )

    assert modality == "image"
    placeholder = "<image>"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"{placeholder}\n{question}"}]
        for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# NVLM-D
def run_nvlm_d(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "nvidia/NVLM-D-72B"

    # Adjust this as necessary to fit in GPU
    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        tensor_parallel_size=4,
        limit_mm_per_prompt={modality: 1},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"<image>\n{question}"}] for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Ovis
def run_ovis(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "AIDC-AI/Ovis2-1B"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        trust_remote_code=True,
        dtype="half",
        limit_mm_per_prompt={modality: 1},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"<image>\n{question}"}] for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Ovis2_5
def run_ovis2_5(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "AIDC-AI/Ovis2.5-2B"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        trust_remote_code=True,
        dtype="half",
        limit_mm_per_prompt={modality: 1},
    )
    if modality == "image":
        placeholder = "<image>"
    elif modality == "video":
        placeholder = "<video>"

    prompts = [
        f"<|im_start|>user\n\n{placeholder}\n{question}<|im_end|>\n<|im_start|>assistant\n"
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# PaddleOCR-VL
def run_paddleocr_vl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "PaddlePaddle/PaddleOCR-VL"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
        trust_remote_code=True,
    )

    placeholder = "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>"
    prompts = [
        (f"<|begin_of_sentence|>User: {question}{placeholder}\nAssistant: ")
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# PaliGemma
def run_paligemma(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    # PaliGemma has special prompt format for VQA
    prompts = ["caption en" for _ in questions]
    engine_args = EngineArgs(
        model="google/paligemma-3b-mix-224",
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# PaliGemma 2
def run_paligemma2(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    # PaliGemma 2 has special prompt format for VQA
    prompts = ["caption en" for _ in questions]
    engine_args = EngineArgs(
        model="google/paligemma2-3b-ft-docci-448",
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Phi-3-Vision
def run_phi3v(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [
        f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"
        for question in questions
    ]

    # num_crops is an override kwarg to the multimodal image processor;
    # For some models, e.g., Phi-3.5-vision-instruct, it is recommended
    # to use 16 for single frame scenarios, and 4 for multi-frame.
    #
    # Generally speaking, a larger value for num_crops results in more
    # tokens per image instance, because it may scale the image more in
    # the image preprocessing. Some references in the model docs and the
    # formula for image tokens after the preprocessing
    # transform can be found below.
    #
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct#loading-the-model-locally
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/main/processing_phi3_v.py#L194
    engine_args = EngineArgs(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={"num_crops": 16},
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Phi-4-multimodal-instruct
def run_phi4mm(questions: list[str], modality: str) -> ModelRequestData:
    """
    Phi-4-multimodal-instruct supports both image and audio inputs. Here, we
    show how to process image inputs.
    """
    assert modality == "image"
    model_path = snapshot_download("microsoft/Phi-4-multimodal-instruct")
    # Since the vision-lora and speech-lora co-exist with the base model,
    # we have to manually specify the path of the lora weights.
    vision_lora_path = os.path.join(model_path, "vision-lora")
    prompts = [
        f"<|user|><|image_1|>{question}<|end|><|assistant|>" for question in questions
    ]
    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=5120,
        max_num_seqs=2,
        max_num_batched_tokens=12800,
        enable_lora=True,
        max_lora_rank=320,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={"dynamic_hd": 16},
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        lora_requests=[LoRARequest("vision", 1, vision_lora_path)],
    )


# Pixtral HF-format
def run_pixtral_hf(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "mistral-community/pixtral-12b"

    # NOTE: Need L40 (or equivalent) to avoid OOM
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=6144,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [f"<s>[INST]{question}\n[IMG][/INST]" for question in questions]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Qwen-VL
def run_qwen_vl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    engine_args = EngineArgs(
        model="Qwen/Qwen-VL",
        trust_remote_code=True,
        max_model_len=1024,
        max_num_seqs=2,
        hf_overrides={"architectures": ["QwenVLForConditionalGeneration"]},
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [f"{question}Picture 1: <img></img>\n" for question in questions]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Qwen2-VL
def run_qwen2_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "Qwen/Qwen2-VL-7B-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Qwen2.5-VL
def run_qwen2_5_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Qwen2.5-Omni
def run_qwen2_5_omni(questions: list[str], modality: str):
    model_name = "Qwen/Qwen2.5-Omni-7B"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|IMAGE|>"
    elif modality == "video":
        placeholder = "<|VIDEO|>"

    default_system = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
        "Group, capable of perceiving auditory and visual inputs, as well as "
        "generating text and speech."
    )

    prompts = [
        (
            f"<|im_start|>system\n{default_system}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_bos|>{placeholder}<|vision_eos|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]
    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Qwen3-VL-Dense
def run_qwen3_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "Qwen/Qwen3-VL-4B-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Qwen3-VL-MOE
def run_qwen3_vl_moe(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# R-4B
def run_r_vl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "YannQi/R-4B"

    prompts = [
        f"<|im_start|>user <image>\n{question}<|im_end|><|im_start|>assistant\n"
        for question in questions
    ]

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=16384,
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# SkyworkR1V
def run_skyworkr1v(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "Skywork/Skywork-R1V-38B"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={modality: 1},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"<image>\n{question}"}] for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for SkyworkR1V
    # https://huggingface.co/Skywork/Skywork-R1V-38B/blob/main/conversation.py
    stop_tokens = ["<｜end▁of▁sentence｜>", "<|endoftext|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# SmolVLM2-2.2B-Instruct
def run_smolvlm(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        enforce_eager=True,
        mm_processor_kwargs={
            "max_image_size": {"longest_edge": 384},
        },
        limit_mm_per_prompt={modality: 1},
    )
    prompts = [
        (f"<|im_start|>User:<image>{question}<end_of_utterance>\nAssistant:")
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Step3
def run_step3(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "stepfun-ai/step3-fp8"

    # NOTE: Below are verified configurations for step3-fp8
    # on 8xH100 GPUs.
    engine_args = EngineArgs(
        model=model_name,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=8,
        limit_mm_per_prompt={modality: 1},
        reasoning_parser="step3",
    )

    prompts = [
        "<｜begin▁of▁sentence｜> You are a helpful assistant. <|BOT|>user\n "
        f"<im_patch>{question} <|EOT|><|BOT|>assistant\n<think>\n"
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# omni-research/Tarsier-7b
def run_tarsier(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "omni-research/Tarsier-7b"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={modality: 1},
    )
    prompts = [(f"USER: <image>\n{question} ASSISTANT:") for question in questions]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


def run_tarsier2(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "omni-research/Tarsier2-Recap-7b"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        hf_overrides={
            "architectures": ["Tarsier2ForConditionalGeneration"],
            "model_type": "tarsier2",
        },
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


model_example_map = {
    "aria": run_aria,
    "aya_vision": run_aya_vision,
    "bagel": run_bagel,
    "bee": run_bee,
    "blip-2": run_blip2,
    "chameleon": run_chameleon,
    "command_a_vision": run_command_a_vision,
    "deepseek_vl_v2": run_deepseek_vl2,
    "deepseek_ocr": run_deepseek_ocr,
    "dots_ocr": run_dots_ocr,
    "eagle2_5": run_eagle2_5,
    "ernie45_vl": run_ernie45_vl,
    "fuyu": run_fuyu,
    "gemma3": run_gemma3,
    "gemma3n": run_gemma3n,
    "glm4v": run_glm4v,
    "glm4_1v": run_glm4_1v,
    "glm4_5v": run_glm4_5v,
    "glm4_5v_fp8": run_glm4_5v_fp8,
    "h2ovl_chat": run_h2ovl,
    "hunyuan_vl": run_hunyuan_vl,
    "hyperclovax_seed_vision": run_hyperclovax_seed_vision,
    "idefics3": run_idefics3,
    "interns1": run_interns1,
    "internvl_chat": run_internvl,
    "kanana_v": run_kanana_v,
    "keye_vl": run_keye_vl,
    "keye_vl1_5": run_keye_vl1_5,
    "kimi_vl": run_kimi_vl,
    "lightonocr": run_lightonocr,
    "lfm2_vl": run_lfm2_vl,
    "llama4": run_llama4,
    "llava": run_llava,
    "llava-next": run_llava_next,
    "llava-next-video": run_llava_next_video,
    "llava-onevision": run_llava_onevision,
    "mantis": run_mantis,
    "minicpmo": run_minicpmo,
    "minicpmv": run_minicpmv,
    "minimax_vl_01": run_minimax_vl_01,
    "mistral3": run_mistral3,
    "molmo": run_molmo,
    "molmo2": run_molmo2,
    "nemotron_vl": run_nemotron_vl,
    "NVLM_D": run_nvlm_d,
    "ovis": run_ovis,
    "ovis2_5": run_ovis2_5,
    "paddleocr_vl": run_paddleocr_vl,
    "paligemma": run_paligemma,
    "paligemma2": run_paligemma2,
    "phi3_v": run_phi3v,
    "phi4_mm": run_phi4mm,
    "pixtral_hf": run_pixtral_hf,
    "qwen_vl": run_qwen_vl,
    "qwen2_vl": run_qwen2_vl,
    "qwen2_5_vl": run_qwen2_5_vl,
    "qwen2_5_omni": run_qwen2_5_omni,
    "qwen3_vl": run_qwen3_vl,
    "qwen3_vl_moe": run_qwen3_vl_moe,
    "rvl": run_r_vl,
    "skywork_chat": run_skyworkr1v,
    "smolvlm": run_smolvlm,
    "step3": run_step3,
    "tarsier": run_tarsier,
    "tarsier2": run_tarsier2,
}


MODELS_NEED_VIDEO_METADATA = [
    "glm4_1v",
    "glm4_5v",
    "glm4_5v_fp8",
    "molmo2",
    "qwen3_vl",
    "qwen3_vl_moe",
]


def get_multi_modal_input(args):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if args.modality == "image":
        # Input image and question
        image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
        img_questions = [
            "What is the content of this image?",
            "Describe the content of this image in detail.",
            "What's in the image?",
            "Where is this image taken?",
        ]

        return {
            "data": image,
            "questions": img_questions,
        }

    if args.modality == "video":
        # Input video and question
        needs_metadata = args.model_type in MODELS_NEED_VIDEO_METADATA
        video = VideoAsset(name="baby_reading", num_frames=args.num_frames).np_ndarrays
        metadata = VideoAsset(name="baby_reading", num_frames=args.num_frames).metadata
        vid_questions = ["Why is this video funny?"]

        return {
            "data": ([(video, metadata)] if needs_metadata else video),
            "questions": vid_questions,
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)


def apply_image_repeat(
    image_repeat_prob, num_prompts, data, prompts: list[str], modality
):
    """Repeats images with provided probability of "image_repeat_prob".
    Used to simulate hit/miss for the MM preprocessor cache.
    """
    assert image_repeat_prob <= 1.0 and image_repeat_prob >= 0
    no_yes = [0, 1]
    probs = [1.0 - image_repeat_prob, image_repeat_prob]

    inputs = []
    inputs_with_empty_media = []
    cur_image = data
    for i in range(num_prompts):
        if image_repeat_prob is not None:
            res = random.choices(no_yes, probs)[0]
            if res == 0:
                # No repeat => Modify one pixel
                cur_image = cur_image.copy()
                new_val = (i // 256 // 256, i // 256, i % 256)
                cur_image.putpixel((0, 0), new_val)

        uuid = "uuid_{}".format(i)

        inputs.append(
            {
                "prompt": prompts[i % len(prompts)],
                "multi_modal_data": {modality: cur_image},
                "multi_modal_uuids": {modality: uuid},
            }
        )

        inputs_with_empty_media.append(
            {
                "prompt": prompts[i % len(prompts)],
                "multi_modal_data": {modality: None},
                "multi_modal_uuids": {modality: uuid},
            }
        )

    return inputs, inputs_with_empty_media


@contextmanager
def time_counter(enable: bool):
    if enable:
        import time

        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time
        print("-" * 50)
        print("-- generate time = {}".format(elapsed_time))
        print("-" * 50)
    else:
        yield


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models for text generation"
    )
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        default="llava",
        choices=model_example_map.keys(),
        help='Huggingface "model_type".',
    )
    parser.add_argument(
        "--num-prompts", type=int, default=4, help="Number of prompts to run."
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="image",
        choices=["image", "video"],
        help="Modality of the input.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to extract from the video.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed when initializing `vllm.LLM`.",
    )

    parser.add_argument(
        "--image-repeat-prob",
        type=float,
        default=None,
        help="Simulates the hit-ratio for multi-modal preprocessor cache (if enabled)",
    )

    parser.add_argument(
        "--disable-mm-processor-cache",
        action="store_true",
        help="If True, disables caching of multi-modal processor.",
    )

    parser.add_argument(
        "--time-generate",
        action="store_true",
        help="If True, then print the total generate() call time",
    )

    parser.add_argument(
        "--use-different-prompt-per-request",
        action="store_true",
        help="If True, then use different prompt (with the same multi-modal "
        "data) for each request.",
    )

    parser.add_argument(
        "--verify-mm-cache-hit-with-uuids",
        action="store_true",
        help="If True, will send all requests in a second batch with empty mm "
        "data to verify cache hits with UUIDs.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=None,
        help="Tensor parallel size to override the model's default setting. ",
    )
    return parser.parse_args()


def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    if args.tensor_parallel_size is not None and args.tensor_parallel_size < 1:
        raise ValueError(
            f"tensor_parallel_size must be a positive integer, "
            f"got {args.tensor_parallel_size}"
        )

    modality = args.modality
    mm_input = get_multi_modal_input(args)
    data = mm_input["data"]
    questions = mm_input["questions"]

    req_data = model_example_map[model](questions, modality)

    # Disable other modalities to save memory
    default_limits = {"image": 0, "video": 0, "audio": 0}
    req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
        req_data.engine_args.limit_mm_per_prompt or {}
    )

    engine_args = asdict(req_data.engine_args) | {
        "seed": args.seed,
        "mm_processor_cache_gb": 0 if args.disable_mm_processor_cache else 4,
    }
    if args.tensor_parallel_size is not None:
        engine_args["tensor_parallel_size"] = args.tensor_parallel_size

    # Install capture hook for GLM-4.1V position comparison
    is_glm4_compare = model in ("glm4_1v", "glm4_5v") and modality == "video"
    if is_glm4_compare:
        _install_vllm_capture_hook()

    llm = LLM(**engine_args)

    # Don't want to check the flag multiple times, so just hijack `prompts`.
    prompts = (
        req_data.prompts
        if args.use_different_prompt_per_request
        else [req_data.prompts[0]]
    )

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = (
        SamplingParams(
            temperature=0.2, max_tokens=64, stop_token_ids=req_data.stop_token_ids
        )
        if req_data.sampling_params is None
        else req_data.sampling_params
    )

    assert args.num_prompts > 0
    if args.num_prompts == 1:
        # Single inference
        uuid = "uuid_0"
        inputs = {
            "prompt": prompts[0],
            "multi_modal_data": {modality: data},
            "multi_modal_uuids": {modality: uuid},
        }
        inputs_with_empty_media = {
            "prompt": prompts[0],
            "multi_modal_data": {modality: None},
            "multi_modal_uuids": {modality: uuid},
        }
    else:
        # Batch inference
        if args.image_repeat_prob is not None:
            # Repeat images with specified probability of "image_repeat_prob"
            inputs, inputs_with_empty_media = apply_image_repeat(
                args.image_repeat_prob,
                args.num_prompts,
                data,
                prompts,
                modality,
            )
        else:
            # Use the same image for all prompts
            inputs = []
            inputs_with_empty_media = []
            for i in range(args.num_prompts):
                uuid = "uuid_{}".format(i)
                inputs.append(
                    {
                        "prompt": prompts[i % len(prompts)],
                        "multi_modal_data": {modality: data},
                        "multi_modal_uuids": {modality: uuid},
                    }
                )
                inputs_with_empty_media.append(
                    {
                        "prompt": prompts[i % len(prompts)],
                        "multi_modal_data": {modality: None},
                        "multi_modal_uuids": {modality: uuid},
                    }
                )

    # Add LoRA request if applicable
    lora_request = (
        req_data.lora_requests * args.num_prompts if req_data.lora_requests else None
    )

    with time_counter(args.time_generate):
        outputs = llm.generate(
            inputs,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

    print("-" * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)

    # --- GLM-4.1V / GLM-4.5V video position comparison ---
    if is_glm4_compare:
        vllm_prompt_token_ids = outputs[0].prompt_token_ids
        # Extract raw video frames for the HF processor
        # data is [(video_np, metadata)] for models needing metadata
        raw_video = data[0][0] if isinstance(data[0], tuple) else data
        raw_metadata = data[0][1] if isinstance(data[0], tuple) else None
        compare_glm4v_positions(
            model_name=req_data.engine_args.model,
            prompt=prompts[0],
            video_data=raw_video,
            video_metadata=raw_metadata,
            mm_processor_kwargs=dict(req_data.engine_args.mm_processor_kwargs or {}),
            vllm_prompt_token_ids=vllm_prompt_token_ids,
            tokenizer=llm.get_tokenizer(),
        )

    if args.verify_mm_cache_hit_with_uuids:
        try:
            # Verify cache hits with UUIDs
            print(
                "Sending a second batch of requests with empty media"
                " and matching UUIDs."
            )
            outputs = llm.generate(
                inputs_with_empty_media,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
            print("-" * 50)
            for o in outputs:
                generated_text = o.outputs[0].text
                print(generated_text)
                print("-" * 50)
        except Exception as e:
            print(f"Failed to verify cache hits with UUIDs. Error: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
