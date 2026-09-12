#!/bin/bash
# 真切块对拍: 10k token 长提示 + max_num_batched_tokens=1024 → 预填充被切成 ~10 块,
# 逐级对拍 native / yang bf16 / int8_static(int8 池) 的生成文本。
export HF_HOME=/root/autodl-tmp/hf UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
export YANG_KV_SCALE_PATH=/root/autodl-tmp/evals/kv_scales.pt
cd /root/autodl-tmp/vllm && source .venv/bin/activate
EV=/root/autodl-tmp/evals
rm -f $EV/.chunk.done $EV/.chunk.fail
echo "[chunk] native START $(date +%H:%M:%S)"
env -u YANG_ATTN_MODE python $EV/chunk_prefill_probe.py > $EV/CH_native.out 2> $EV/CH_native.log || { touch $EV/.chunk.fail; exit 1; }
echo "[chunk] yang_bf16 START $(date +%H:%M:%S)"
YANG_ATTN_MODE=bf16 python $EV/chunk_prefill_probe.py > $EV/CH_yang_bf16.out 2> $EV/CH_yang_bf16.log || { touch $EV/.chunk.fail; exit 1; }
echo "[chunk] static_pc START $(date +%H:%M:%S)"
YANG_ATTN_MODE=int8_static_per_channel KVD=int8 python $EV/chunk_prefill_probe.py > $EV/CH_static_pc.out 2> $EV/CH_static_pc.log || { touch $EV/.chunk.fail; exit 1; }
# vLLM 日志走 stdout,与生成文本同流;diff 前先抠出"引擎谢幕行之后"的纯文本
for f in native yang_bf16 static_pc; do
  sed -e "1,/exiting busy loop/d" $EV/CH_$f.out | sed -e "/./,\$!d" > $EV/T_$f.txt
done
echo "=== native vs yang_bf16 ==="; diff $EV/T_native.txt $EV/T_yang_bf16.txt && echo SAME_1
echo "=== yang_bf16 vs static_pc ==="; diff $EV/T_yang_bf16.txt $EV/T_static_pc.txt && echo SAME_2
touch $EV/.chunk.done
echo "[chunk] ALL DONE $(date +%H:%M:%S)"
