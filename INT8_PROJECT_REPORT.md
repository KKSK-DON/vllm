# int8 KV Cache 动态量化项目 · 成果报告

> 2026-08-12 步骤③收官时整理。int8 kvcache 量化项目（按 Doc 执行）。
> 分支：`KKSK-DON/vllm` `feature/int8-kvcache`；模型：Qwen3.5-9B（混合架构）；机器：单卡 RTX 4090 48G。

---

## 一、一句话总结

在 vLLM 中以纯 PyTorch 实现了 decode 阶段 paged attention 的 **int8 KV cache 动态量化**（per_channel / per_head 两种粒度），端到端跑通真实模型评测：**humaneval + gsm8k 全量三组对照，精度损失全部小于一个标准误（≤0.15pp），统计意义上无损**。

背景动机（Doc 设定）：国产芯片无原生 fp8 算力 → 需要 int8 路线；KV cache 是 decode 显存与带宽的大头。

---

## 二、完成了什么（对照 Doc 五步）

| Doc 步骤 | 状态 | 产出 |
| --- | --- | --- |
| (1) kvcache 分布分析、定量化维度 | ✅ | `kvcache_distribution.py`：K 侧 outlier 呈"列状纹理"（固定 channel 跨 token 系统性偏大），per-channel 的 K 侧 MSE 低 3-4 倍 → 选定 per_channel/per_head 两粒度对比 |
| (2) PyTorch 量化 pageattention + 单测 | ✅ | `tests/kernels/attention/test_flash_attn.py`：`yang_paged_attn`（反量化先行版）+ `yang_paged_attn_int8_accelerate`（int8-first 加速版，含 per_channel 下 k_scale 吸收进 q 的数学等效技巧）。GPU 实测 **640/640 全过** |
| (3) 模型中动态量化验证精度 | ✅ **本阶段主体** | 详见下文三、四 |
| (4) 静态量化 | ✅ **完成并实测**（含物理 int8 存储,超出 Doc 要求） | 写入路径校准（aime25 末 5 题量尺,前 15 题考卷零重叠）→ `int8_static_*`（bf16 池+死尺）→ `int8_phys_*`（一等公民 `kv_cache_dtype=int8`,池子物理 int8）。四场 gsm8k 冒烟全部 0.875 与基线逐题持平;**KV 容量实测 1,567,690 → 2,707,828 token（×1.73,并发 191→331）**,同一 58.59 GiB 预算 |
| (5) 真 kernel 性能优化 | 可外包（文档原文"假装外包"） | 未做；aime 平方成本观察为其必要性提供了实证（见 4.3） |

### 步骤③的具体工程

1. **接线**：`flash_attn.py` forward 非 DCP 分支加 `YANG_ATTN_MODE` 环境变量岔路口（bf16 / int8_per_channel / int8_per_head 三档）→ 新文件 `yang_attn.py`。
2. **适配器四件套**：query_start_loc 前缀和差分出 query_lens；seq_lens/block_table 直取；现场量化；`copy_` + `view_as` 写回调用方预分配的输出缓冲（out-parameter 模式，CUDA graph 地址稳定性所需）。
3. **围栏**：8 道 assert（alibi/sinks/causal/sliding_window/fp8/mm-prefix/R-SWA/token 数一致性）——不理解的分支不硬啃，用 assert 拦住防静默走错路；激活日志（每模式打一次）保证"确实走了我的代码"可证。
4. **验证梯子**：先 bf16 纯 PyTorch 等效版与原生 flash 内核同题对照（8 题 gsm8k **逐题一致** 0.875 = 0.875）证明管道，再切 int8 证明量化数学。
5. **两轮迭代**（见四）：NaN 修复 → 提速 6.5 倍。

代码提交：`491e48c21`（接线 + NaN 修复）、`a728143de`（提速 + fp32 除法）。

---

## 三、最终数据

### 三组对照总表

| 基准 | 原始 bf16 | int8_per_channel | int8_per_head |
| --- | --- | --- | --- |
| **humaneval** pass@1（164 题，0-shot） | 0.7073 | **0.7134**（+0.6pp） | **0.7073**（±0） |
| **gsm8k** strict（1319 题全量，5-shot） | 0.8787 | **0.8772**（−0.15pp） | **0.8779**（−0.08pp） |
| gsm8k flexible | 0.8741 | 0.8704 | 0.8741 |
| aime25（15 题半卷 + \boxed 指令,0-shot） | 0.2667 (4/15) | 0.2000 (3/15) | 0.3333 (5/15) |

\* aime25 二次战役后补齐：首轮基线 0 分,经逐样本取证发现模型三题全对而判分器全判零——判分正则只认「回复即答案」或 \boxed{} 两种信封,长推理散文两样都没有。加系统提示「final answer within \boxed{}」后判分链路打通。最终三组差异 ±1 题,双向摆动,均在误差棒（±0.13）内。per_head 高于基线属小样本噪声,非量化增益。生成上限 20480 token（实测最长链 ~14k,留 40% 余量）。

### 结论与解读

- **所有差异 ≤0.6pp，小于一个标准误（±0.9pp）**——正确表述是"量化版与原版的差异小于随机波动"，不是"掉得少"。
- **per_channel 没有赢 per_head**：分布分析的 MSE 优势是真的，但 Qwen3.5-9B 的 outlier 病情未重到让 per_head 的 128 档刻度不够用——**量化方案的收益取决于数据分布的病重程度**；拉开差距需换 outlier 更凶的模型或降到 int4。
- 复现性：克隆实例上全量 gsm8k 基线 0.8787 vs 原机 0.8749（±0.9pp 内）；bf16 冒烟与原生逐题一致。
- **八连平**（同卷 gsm8k 8 题,变量逐一引入）：原生 / bf16 / 动态 pc / 动态 ph / 静态 pc / 静态 ph / 物理 pc / 物理 ph 全部 0.875——每个新变量（换算子→动态量化→死尺→物理存储）单独证明清白。静态尺以 aime25 末 5 题校准、考 gsm8k 仍持平,跨领域泛化成立。
- 速度（8 题 gsm8k 生成段）：原生 8s ≈ bf16 版 9s ≈ 动态 int8 版 9s（首版整池量化 59s）≈ 静态版 9s ≈ 物理版 8-11s（输出 81-99 toks/s）——短上下文下六种实现同速;静态/物理的结构性优势（免每步 findmax,线性 vs 平方）在长上下文才显形。
- **显存容量（c8 实测,物理版 vs auto,同机同 58.59 GiB 预算同 max_model_len=8192）**：KV cache 总容量 1,567,690 → **2,707,828 token（×1.727）**,最大并发 191.37× → **330.55×**。未达理论 2× 的原因是混合架构:8 层全注意力的每 token 字节减半,24 层 GDN 线性注意力的状态页不随 kv_cache_dtype 变,共享池摊平后得 1.73×（由此可反推 GDN 状态摊销占地 ≈ 注意力 bf16 占地的 19%）;纯 Transformer 架构下该比值应逼近 1.94-2.0×。

---

## 四、三次事故 → 三个发现（复盘素材）

### 4.1 NaN 事故：整池 findmax 读到无主内存

- **现象**：两种 int8 首跑全部 0/8，而单测 640 用例全过、bf16 模式满分。
- **取证**：探针打印显示第一次调用时"本批引用块 absmax=13.31（健康）、整池 absmax=NaN"；从第二次调用起引用块也 NaN——毒素沿"注意力输出→隐藏状态→写回 cache"逐层传染。
- **定罪**：混合架构模型所有层共用一块打包大内存（vLLM `attn_utils.py` 注释原话 "all packed tensors alias the same backing"），本层视角的"池子"横跨别层地盘；bf16 视角读任意字节必然撞上 NaN 位模式。
- **修法**：scale 统计范围从整池缩到"本批引用块的合法 token"（逐序列 gather + 掐掉最后一块未写满的尾巴——尾巴同样是无主字节）。
- **教训**：动态量化的统计范围 = 实际拥有的数据，一个字节都不能多。vLLM 自家 fp8 从构造上免疫（写入时量化 + 离线校准静态 scale，从不扫池）。传统单体架构模型上此 bug 不炸但仍错（过期数据默默撑大 scale）——Qwen3.5 的共享内存布局把静默错误放大成响亮失败，反而是帮忙。

### 4.2 bf16 除法掉一题：量化算术的中间精度是实变量

- 提速版把量化除法从 fp32 降到 bf16（bf16 尾数仅 8 位，取整前的商偶尔差一档），per_channel 8 题回归从 0.875 掉到 0.75。两版数学等价、种子固定，唯一变量就是除法精度——干净的 A/B 归因。抬回 fp32 后归位。
- **教训**：round 之前的算术在 fp32 里做；"数学等价"不等于"数值等价"。

### 4.3 aime 平方成本：读取侧动态量化的固有缺陷现形

- 32k 上下文下 int8 场次 GPU 99% 利用率却需 3.5h/场（基线 21min）：每生成一个 token 都要把该序列全部已有 KV 重新 gather + findmax + 量化，单步成本随上下文线性涨、整场平方级。gsm8k 的 1.4k 上下文把它藏住，32k 让它现形。
- **教训**：这正是工业界选"写入时量化（每值一生只处理一次）+ 静态 scale + 融合 kernel"的根本动机。实验数据完整走通了这条因果链——从"为什么我的方案慢"推出"为什么 vLLM 那样设计"。

---

## 五、关键设计决策记录

| 决策点 | 选择 | 理由 |
| --- | --- | --- |
| 量化位置 | 读取侧算子替换（pageattention 调用处），非写入侧 | Doc 步骤③原文；手术范围最小，精度实验最快出数 |
| scale 统计范围 | 本批引用块的合法 token（token 级精确，掐块尾） | NaN 事故教训；块级不够（尾巴是无主字节） |
| 统计与量化分离 | `compute_kv_scale` 只算 scale；量化在注意力循环内对 gather 出的 token 做 | 统计可分块合并（max 结合律），变换必须全局一次；避免整池除法的 59s 开销 |
| 量化对象 | 每序列 gather 出的干净 K/V（全局 scale 传入） | 与"先整池量化再 gather"逐位等价（gather 与逐元素运算可交换），且天然绕开无主内存 |
| 除法精度 | fp32 | 4.2 的 A/B 证据 |
| 冻结已验证算子 | `yang_paged_attn_int8_accelerate` 尽量不动，脏活留给适配层 | 640 用例的"已验证"标签最贵；环境问题归适配器管，数学层假设输入干净 |
| per-seq scale（更细粒度） | 讨论后未采用 | 精度会更好但偏离单测与文档定义的"全局 scale"算法，实验可比性优先；作为取舍话题保留 |
| cascade attention | 不处理 | 本版本默认关闭（`disable_cascade_attn=True` 需 opt-in），gate 不会被绕过——从总开关向下查证,而非从触发条件向上猜 |

---

## 六、与工业方案（vLLM fp8）的对照

| 环节 | 本项目（动态 int8） | vLLM fp8 生产路线 |
| --- | --- | --- |
| scale 何时定 | 每次调用现场 findmax | 部署前离线校准，随权重发货（`layer._k_scale`） |
| 数据何时量化 | 每次调用重新量化引用到的 token | 每 token 写入 cache 那一刻量化一次，终身不改 |
| cache 存储格式 | 仍 bf16（临时 int8 副本，不省显存） | 物理 fp8，容量翻倍是真的 |
| 注意力算子 | 纯消费者：吃 int8 + scale | 纯消费者：吃 fp8 + descale（同一接口思想） |
| 长上下文成本 | 平方级（4.3） | 常数级 |

三级台阶总结：**正确性从来不需要 kernel（纯 PyTorch 全程可达）；真省显存需要 int8 存储格式（仍可纯 PyTorch，慢）；kernel 买的只是速度**。标准开发顺序：纯 PyTorch 版当标准答案 → kernel 版对拍。本项目的 PyTorch 实现即未来 kernel 的对拍基准。

---

## 七、遗留与可选项

1. **成果条目提炼**（按 Doc 的收尾流程）——下一个动作。
2. 静态量化：原理已可讲；代码实现（校准 + 写入钩子 + 读取新模式，纯 PyTorch）可选。
3. aime 三组对照补齐：带 chat template 重跑，约一晚机器费，可选。
4. 逐题错误分析：需带 `--log_samples` 重跑（当时未加，逐题输出不存在），需要深挖时再做。
5. CUDA kernel（④a 写入内核起步）：当练手场，Doc 未作要求。

## 八、档案索引

- 代码：`feature/int8-kvcache` 分支，commits `491e48c21`、`a728143de`；核心文件 `vllm/v1/attention/backends/yang_attn.py` + `flash_attn.py` 岔路口；单测 `tests/kernels/attention/test_flash_attn.py`。
- 评测原始日志/JSON：gpuhub 机器数据盘 `/root/autodl-tmp/evals/{A_baseline, B_*, C_*}`（关机保数据）。
- 技术笔记：`KV_CACHE_QUANTIZATION.md`（附录 E 为本阶段实验详录；D.5 为量化粒度判读方法）。
- 工作方式备注：代码由本人手写（两轮重写：接线适配器、scale 重构），AI 担任评审（严重度排序错误清单）、运维（远程机器与评测编排）与教学讲解；诊断结论均经探针数据或源码原文验证。
