# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os


def main() -> None:
    from vllm import LLM, SamplingParams

    para = (
        "The quick brown fox jumps over the lazy dog. "
        "Energy conservation follows from time-translation symmetry. "
    )
    prompt = (
        "Read the following text carefully.\n\n"
        + para * 400
        + "\nSummarize the text in one sentence."
    )

    llm = LLM(
        model="Qwen/Qwen3.5-9B",
        max_model_len=16384,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        max_num_batched_tokens=int(os.environ.get("CHUNK_BUDGET", "1024")),
        kv_cache_dtype=os.environ.get("KVD", "auto"),
    )
    out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=64))
    print(out[0].outputs[0].text)


if __name__ == "__main__":
    main()
