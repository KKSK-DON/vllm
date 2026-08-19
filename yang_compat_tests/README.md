# yang_compat_tests — 调度特性兼容性对拍(chunked prefill / prefix caching)

步骤④(静态量化 + 物理 int8 存储)收官后的两场补测(2026-08-19):aime 七连测只覆盖了
"整段预填充 + 逐 token 解码"两种注意力形状,这里补上调度器的另外两个特性,验证
`int8_static_*` / `int8_phys_*` 在它们之下依然无损。详细结果记录在
`KV_CACHE_QUANTIZATION.md` 附录 F.6 / F.7;本目录是可复跑的测试本体。

## 文件清单

| 文件 | 角色 |
| --- | --- |
| `chunk_prefill_probe.py` | 切块探针:10k token 提示 + 贪心 64 token,旋钮全走环境变量 |
| `chunk_test.sh` | 切块跑批:四场阶梯 + 提取纯文本 + 三级 diff |
| `prefix_cache_probe.py` | 前缀缓存探针:7.5k 公共前缀 + 两个问题串行提交,打 `TIME_`/`TEXT_` 标记行 |
| `prefix_test.sh` | 前缀缓存跑批:开/关 × 三模式共 6 场 + 双判据裁决 |

## 测试一:chunked prefill(分块预填充)

**制造形状**:`max_num_batched_tokens=1024`(单步 token 预算)配 10k 提示,调度器被迫把
预填充切成 ~10 块——每块产生"本步新 token 数 1024 < 已缓存数"的中段形状,这是七连测
从未真实出现过的路径。

**四场阶梯**(相邻两场只差一个变量):

1. 原生(不设 `YANG_ATTN_MODE`)——基准
2. `YANG_ATTN_MODE=bf16`——考适配器数学
3. `int8_static_per_channel`——考量化
4. `int8_phys_per_channel` + `kv_cache_dtype=int8`——考物理存储

**判据**:贪心解码下生成文本 = 全部 KV 读写的指纹,逐字节 diff。

**实测结果**:②=③=④ 逐字节相同;①与②差一个词。归因加赛(`CHUNK_BUDGET=16384`
关闭切块重跑①②)判明:②切块=②不切块=①不切块,唯①切块是异类——漂移属于原生融合
内核自身的切块浮点抖动(分块改变瓦片式在线 softmax 的累加顺序,近平局处掀翻贪心);
pytorch 路径反而具备切块不变性(每行 softmax 对全前缀一次归约,与块边界无关)。

## 测试二:prefix caching(前缀缓存)

**设计**:同一段 ~7.5k token 前缀接两个不同问题,**串行**提交(第一问跑完,第二问的
前缀块必然已在缓存);`enable_prefix_caching` 开/关 × 原生/static/phys 共 6 场。
混合架构默认不开该特性,须显式传 True(上游注释:opt-in while the feature matures)。

**双判据**(缺一不可):

1. 不变性——开/关缓存的生成文本逐字相同;
2. 生效性——开缓存后第二问耗时显著下降(命中证据;没有这条,"文本相同"可能只是
   特性从未触发的空转假阳性)。

**实测结果**:三模式文本全部逐字相同;第二问耗时 原生 0.72→0.49s、static 0.93→0.54s、
phys 0.95→0.59s(提速 32-42%)。物理模式的 int8 块被跨请求复用零误差——
"scale 全局且时不变 ⇒ 缓存块可共享"的实测证明。

## 运行前提

- 脚本假定 gpuhub 机器布局:仓库+venv 在 `/root/autodl-tmp/vllm`,产物目录
  `/root/autodl-tmp/evals`,校准文件 `kv_scales.pt`(由 `yang_calibrate_kv.py` 生成)。
- 环境变量:`VLLM_ATTENTION_BACKEND=FLASH_ATTN`、`VLLM_USE_FLASHINFER_SAMPLER=0`、
  `YANG_KV_SCALE_PATH` 指向校准文件。
- 启动:`screen -dmS <名字> -L -Logfile <控制台日志> <跑批脚本>`,终态看
  `.chunk.done/.chunk.fail`、`.pfx.done/.pfx.fail` 标记文件。

## 两个踩过的工程坑(脚本里的形状由此而来)

1. **探针必须有 `if __name__ == "__main__":` 护栏**——vLLM 以 spawn 方式启动引擎
   子进程,子进程重新导入主模块;无护栏则顶层 `LLM(...)` 被重复执行,多进程库当场熔断。
2. **不可直接 diff 重定向文件**——vLLM 日志与生成文本同走标准输出,带时间戳/进程号的
   日志天然互不相同。`chunk_test.sh` 按"`exiting busy loop` 谢幕行之后"切出纯文本再比;
   `prefix_cache_probe.py` 改用 `TIME_`/`TEXT_` 标记行,判决直接 grep。
