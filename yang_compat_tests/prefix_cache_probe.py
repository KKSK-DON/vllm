# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time


def main() -> None:
    from vllm import LLM, SamplingParams

    para = (
        "The quick brown fox jumps over the lazy dog. "
        "Energy conservation follows from time-translation symmetry. "
    )
    prefix = "Background material:\n\n" + para * 300
    q1 = "\nQuestion: What animal is mentioned in the background? Answer briefly."
    q2 = "\nQuestion: What physical law is mentioned in the background? Answer briefly."

    llm = LLM(
        model="Qwen/Qwen3.5-9B",
        max_model_len=16384,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_prefix_caching=os.environ.get("PFX", "0") == "1",
        kv_cache_dtype=os.environ.get("KVD", "auto"),
    )
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    t0 = time.time()
    a = llm.generate([prefix + q1], sp)
    t1 = time.time()
    b = llm.generate([prefix + q2], sp)
    t2 = time.time()
    print("TIME_A %.2f" % (t1 - t0))
    print("TIME_B %.2f" % (t2 - t1))
    print("TEXT_A", repr(a[0].outputs[0].text))
    print("TEXT_B", repr(b[0].outputs[0].text))


if __name__ == "__main__":
    main()
