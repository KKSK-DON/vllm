#!/bin/bash
# prefix caching 对拍: 7.5k 公共前缀 + 两问, 开/关 enable_prefix_caching × 三模式。
# 验证 ①开关不改输出文本 ②开缓存后第二问耗时大跌(命中证据)。
export HF_HOME=/root/autodl-tmp/hf UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
export YANG_KV_SCALE_PATH=/root/autodl-tmp/evals/kv_scales.pt
cd /root/autodl-tmp/vllm && source .venv/bin/activate
EV=/root/autodl-tmp/evals
rm -f $EV/.pfx.done $EV/.pfx.fail
run() {
  local name=$1 mode=$2 kvd=$3 pfx=$4
  echo "[pfx] $name START $(date +%H:%M:%S)"
  if [ -z "$mode" ]; then
    PFX=$pfx KVD=$kvd env -u YANG_ATTN_MODE python $EV/prefix_cache_probe.py \
      > $EV/PC_$name.out 2> $EV/PC_$name.log \
      || { echo "[pfx] FAIL $name"; touch $EV/.pfx.fail; exit 1; }
  else
    PFX=$pfx KVD=$kvd YANG_ATTN_MODE=$mode python $EV/prefix_cache_probe.py \
      > $EV/PC_$name.out 2> $EV/PC_$name.log \
      || { echo "[pfx] FAIL $name"; touch $EV/.pfx.fail; exit 1; }
  fi
  echo "[pfx] $name DONE $(date +%H:%M:%S)"
}
run native_off "" auto 0
run native_on "" auto 1
run static_off int8_static_per_channel auto 0
run static_on int8_static_per_channel auto 1
run phys_off int8_phys_per_channel int8 0
run phys_on int8_phys_per_channel int8 1
echo "===== VERDICTS ====="
for m in native static phys; do
  grep -h "^TIME_" $EV/PC_${m}_off.out | sed "s/^/[$m off] /"
  grep -h "^TIME_" $EV/PC_${m}_on.out | sed "s/^/[$m on ] /"
  if diff <(grep "^TEXT_" $EV/PC_${m}_off.out) <(grep "^TEXT_" $EV/PC_${m}_on.out) >/dev/null 2>&1; then
    echo "VERDICT SAME_TEXT: $m on==off"
  else
    echo "VERDICT DIFF_TEXT: $m"
    diff <(grep "^TEXT_" $EV/PC_${m}_off.out) <(grep "^TEXT_" $EV/PC_${m}_on.out) | head -6
  fi
done
touch $EV/.pfx.done
echo "[pfx] ALL DONE $(date +%H:%M:%S)"
