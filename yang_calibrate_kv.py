# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibrate per-channel KV amax on the LAST 5 aime25 questions.

The eval uses the FIRST 15 questions (--limit 15), so calibrating on the
last 5 gives zero overlap while staying on the doc-prescribed data (aime).
Dataset path/split are read from lm-eval's own aime25.yaml so calibration
and exam are guaranteed to load the same source.

Usage (on the GPU machine, repo root, env vars as usual):

    VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0 \
    HF_HOME=/root/autodl-tmp/hf \
    python yang_calibrate_kv.py --out /root/autodl-tmp/evals/kv_scales.pt
"""

import argparse
import os

BOXED_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
NUM_CALIB = 5


def load_calib_questions() -> list[str]:
    import lm_eval.tasks
    import yaml

    task_dir = os.path.join(os.path.dirname(lm_eval.tasks.__file__), "aime")
    with open(os.path.join(task_dir, "aime25.yaml")) as f:
        cfg = yaml.safe_load(f)

    from datasets import load_dataset

    ds = load_dataset(cfg["dataset_path"], split=cfg["test_split"])
    key = next(k for k in ds.column_names if k.lower() in ("problem", "question"))
    return [ds[i][key] for i in range(len(ds) - NUM_CALIB, len(ds))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    # Must be set before LLM() spawns the EngineCore subprocess (env inherits).
    os.environ["YANG_ATTN_MODE"] = "calibrate"
    os.environ["YANG_KV_SCALE_PATH"] = args.out

    questions = load_calib_questions()
    print(f"[calibrate] {len(questions)} aime25 questions (last {NUM_CALIB})")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    conversations = [
        [
            {"role": "system", "content": BOXED_SYSTEM},
            {"role": "user", "content": q},
        ]
        for q in questions
    ]
    llm.chat(conversations, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))

    import torch

    scales = torch.load(args.out, map_location="cpu")
    one = next(iter(scales.values()))
    print(
        f"[calibrate] done: {len(scales)} layers -> {args.out}; "
        f"k_amax shape {tuple(one['k_amax'].shape)}, "
        f"k_amax max {one['k_amax'].max():.2f}"
    )


if __name__ == "__main__":
    main()
