# qfn / MUSA S4000 Handoff

更新时间：2026-10-22
仓库：`/root/lvllm/repos/vllm-xiaotu-moe`（分支 `feature/musa-correctness`）
MUSA 运行机：`root@192.168.100.248`
MUSA vLLM：`/root/vllm-musa`（分支 `v0.32.0-dev`）
目标：在 MTT S4000（MP22，2×48GB，Hygon x86，512GB）上验证 Qwen3.8-Flash-Next（qfn）官方 FP8 权重的 GPU/CPU 混合推理；S4000 无原生 FP8，采用 FP8 权重 + BF16 计算/反量化回退；后续 NVFP4 同样 BF16 计算；TP=2 必要时自己实现 all-reduce；最终尝试 NVFP4 全量驻留 96GB。

## 铁律

- **不要 push**。MUSA 完成前只本地 commit。
- 默认 CUDA 路径必须保留；MUSA 通过 `GPU_DEVICE=MUSA` 选择。
- 不要改 README 性能数据；之前 DS4.1F 数据被改过，用户说“就这样吧”，恢复需用户明确指定。
- 重要方向变化先写 `dev-docs/USER_QA_LEDGER.md`，再动手。
- `/root/vllm-musa/third_party/vllm` 被 `.gitignore` 忽略，改动不会自动进 git；必须同步快照到 `/root/lvllm/patches/musa_qfn_wip/` 再 commit。

## 当前结论（最重要）

qfn **还不能作为正常服务**：

```text
5 token：语义正确
8 token：只有把非法 GDN state index clamp 到 0 才能跑，输出退化
16 token：仍异步失败
8K / TP=2 / CPU golden / NVFP4：未开始
```

5 token 证据：

```text
"What is 1+1?" -> "\n\nThe sum 2"
token_ids = [271, 760, 2542, 220, 17]
```

8 token（clamp 后）：

```text
"\n\nThe sum 2 1 "
ids = [271, 760, 2542, 220, 17, 220, 16, 220]
```

16 token 输出退化，例如：

```text
"\n\nThe sum 2 1 1 1 1 1 1 1 "
```

## 已修复

1. **PLE FP8→BF16**
   - 文件：`third_party/vllm/vllm/models/qwen4_exp/common/ngram_embedding.py`
   - 在 CPU 上解码 PLE 的 FP8 行，乘 `weight_scale_inv`，再回 MUSA。
   - 修复前 hidden 会爆到 ~1e24 后清零，所有 token 都是 0。

2. **QSA decode 非法 main KV slot**
   - 文件：`third_party/vllm/vllm/models/qwen4_exp/amd/qsa.py`
   - 新增 `_xtu_main_slot_mapping()`：从 `metadata.block_table` + `positions` 重新推导 main KV slots。
   - 修复前 decode slot 是 `1812362199`；修复后 `max_tokens=5` 通过。
   - 相关：`amd/ops/qsa.py` 里还有 torch fallback / dense select。

## 当前阻塞

### A. vLLM block table 在 seq=9 已经是垃圾

`XTU_GDN_META_TRACE=1` + `MUSA_LAUNCH_BLOCKING=1` 抓到：

```text
seq=[7] raw0=[[1]]  bt0=[[1]]
seq=[8] raw0=[[2]]  bt0=[[2]]
seq=[9] raw0=[[293]]           bt0=[[293]]
seq=[9] raw0=[[1046632475]]    bt0=[[1046632475]]   # 垃圾
seq=[9] raw0=[[398]]           bt0=[[398]]
```

说明是 `common_attn_metadata.block_table_tensor` 本身坏，不是 GDN gather 之后才坏。

已试过但无效：

- `causal_conv1d_update` 换 torch fallback；
- state index `.clone()`；
- 禁用 metadata reuse；
- prefill 保存有效 state index；
- 每层固定 state slot；
- `mamba_cache_mode="none"`；
- clamp mamba gather 起点；
- `StagedWriteTensor` 的 host apply（还遇到 1D/2D 索引问题，未修好）。

重点文件：

```text
/root/vllm-musa/third_party/vllm/vllm/v1/worker/gpu/buffer_utils.py
/root/vllm-musa/third_party/vllm/vllm/v1/worker/gpu/block_table.py
/root/vllm-musa/third_party/vllm/vllm/v1/attention/backends/gdn_attn.py
/root/vllm-musa/third_party/vllm/vllm/v1/attention/backends/utils.py
```

`gdn_attn.py` 里已有 `XTU_GDN_META_TRACE` 打印，可直接复用。

### B. MATE MP22

用户明确：不是“排除 MATE”，而是**我们给 MATE 补 MP22**。

事实：

- 本机 MATE 0.2.7 已安装；`mate check` 显示 JIT 自动识别 `mp22`，不是硬编码 MP31。
- 但 GDN decode 使用显式 TMA：
  ```text
  T.tma_copy() requires TMA-capable target ...
  TMA is not available for src=state, dst=state_load_stage
  ```
- 只加 `TL_DISABLE_TMA_LOWER=True` 无效，因为是显式 `T.tma_copy` + mbarrier。
- 源码已 clone：`/root/mate_src`（upstream `3c5b592`）。
- 已机械尝试把 `gdn_decode.py` 的 `T.tma_copy` 改 `T.copy`、去掉 barrier，能编译，但运行仍 illegal；这个改动在 **site-packages**，不是正式方案。

**重要：安装目录的 `mate/gdn_kernels/tilelang/gdn_decode.py` 已被改坏，备份在：**

```text
/root/venvs/vllm310/lib/python3.10/site-packages/mate/gdn_kernels/tilelang/gdn_decode.py.tma_bak
```

下次继续 MATE 前，要么恢复备份，要么从 `/root/mate_src` 重新构建安装。不要在改坏的 site-packages 上做结论。

## 环境

```bash
export LD_LIBRARY_PATH=/usr/local/musa/lib:/usr/local/musa/mudnn/lib:/usr/local/mtshmem/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/root/lvllm
export GPU_DEVICE=MUSA
export XTU_MUSA_CORRECTNESS=1
export VLLM_EXPERTS_LOAD_DEVICE=cpu
export XIAOTU_MOE_THREADS=60
export MUSA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
```

Python：

```text
/root/venvs/vllm310/bin/python
torch 2.11.0.post1
torch_musa 2.11.0.post1
vllm 0.32.0
vllm_musa 0.1.32
triton 3.2.0
MATE 0.2.7
```

模型：

```text
/root/models/Qwen3.8-Flash-Next-FP8
```

## 测试脚本

```text
/root/qfn_q.py        max_tokens=8
/root/qfn_q3.py       3
/root/qfn_q4.py       4
/root/qfn_q5.py       5
/root/qfn_q16.py      16
/root/qfn_one.py      单 token
/root/qfn_verify.py   验证
```

关键日志：

```text
/root/isolate_q8.log
/root/causal_torch8.log
/root/causal_clamp8.log
/root/q16.log
/root/gdn_meta_trace.log        # block table 垃圾证据
/root/mate_gdn16.log            # MATE TMA 失败
/root/mate_sync16.log
/root/mate_sync16_block.log
```

常驻测试（未成功）：

```text
/root/qfn_resident.py
/tmp/qfn_cmd
/root/qfn_resident.log
```

失败原因：启动时没带 `XTU_MUSA_CORRECTNESS=1 / VLLM_EXPERTS_LOAD_DEVICE=cpu`，被 FP8 capability 31 校验拒绝；另外 `vllm serve` 缺 `uvicorn`。当前进程已停。

## Git 状态

```text
/root/vllm-musa  09b5200  fix: derive QSA main KV slots from block table on MUSA
/root/lvllm      （以 git log 为准；最近 handoff 前有 MATE 文档提交）
```

`/root/lvllm/patches/musa_qfn_wip/` 保存了 third_party 快照；但最新 GDN/MATE 实验未全部快照，继续前先检查 `git status` 和目录内容。

未 push。

## 下一步建议顺序

1. **先修 block table staging**，不要继续在 GDN state index 上打补丁。
   - 用 `XTU_GDN_META_TRACE=1` 在 seq=9 抓 `raw0` 是哪一个 `StagedWriteTensor`/group 写坏；
   - 检查 `buffer_utils.py` 的 `_apply_write_kernel` 和 `FusedStagedWriter`；
   - 临时可靠方案：所有 block table staged writes 走 host apply，但必须处理 1D/2D 和 fused multi-group。
2. 修好后跑 `q8`、`q16`，确认不是“只坏 seq=9”。
3. 再决定 MATE：
   - 从 `/root/mate_src` 正式加 `feature/mp22`；
   - 只改 GDN decode 的 state load/store staging，`T.copy` + `T.sync_threads()`；
   - 先单 kernel 与 torch 对拍，再接 vLLM。
4. 之后才是 8K、TP=2、CPU golden、NVFP4、96GB 驻留。

## 不要做的事

- 不要声称 qfn 已可用；目前只有 5 token 正确。
- 不要把 `clamp state index` 当成修复。
- 不要在改坏的 site-packages MATE 上继续。
- 不要 push，不要改 README 性能数据。
