# 生产树增量（收回真源用）

## 这是什么

`0001-production-tree-delta-20261106.patch` = 生产树
（远端 `process_data/ref/repos/vllm-mainline`，benchmark 于 2026-11-06）与
「上游 `f03026a548` + `patches/xtu-series/`」之间的**差额**，覆盖 29 个文件。

## 为什么存在

按纪律「本地是唯一真源」，生产树上不应存在本地没有的改动。但实测发现生产树有 29 个
文件的内容与补丁系列重建树不同（`attention.py`、`dspark.py`、`speculator.py`、
`model_runner.py`、`indexer_k_store.py`、`cache_utils.py`、`sparse_mla_kernels.py` 等），
其中包含**别的 agent 未提交的修复与调试代码**。本补丁把它们收回本地，使本地可复现生产。

## 用法

```bash
git clone --depth 1 <vllm> && git fetch --depth 1 origin f03026a548b18ba0a99ca0a632b9343f25a66bc9
bash scripts/apply_xtu_patches.sh <dir>                      # 先打主线系列
git apply <dir>/../prod-recovered/0001-production-tree-delta-20261106.patch   # 再打本增量
```

## ⚠️ 状态：**未逐条审查**

* 已知三类内容：① 越界断言（审计修复，`cache_utils.py` / `indexer_k_store.py` /
  `sparse_mla_kernels.py`）② 行为改动（`attention.py` / `fused_compress_quant_cache.py` /
  `indexer.py` / `dspark.py`）③ 调试残留（注释与采样断言）；
* **本补丁不是验收过的代码**，只是生产现状的忠实记录；
* 其中 `*.bak*` / `*.xtu*bak` 等备份文件**未包含**（它们是未跟踪文件，不属于源码）。

## 与退化排查的关系

本增量同时是**退化嫌疑范围**：退化只可能出现在「生产独有」或「两边不同」的代码里，
纯补丁系列独有的改动可以排除。

---

## 0002-strict-bounds-env-gated.patch

把四处**采样式**越界守卫（`_bound_check_skip` / `_bt_bound_skip`，原本 1/256 抽样、
图捕获期跳过）改成**环境变量门控的严格模式**：

```bash
XIAOTU_STRICT_BOUNDS=1   # 除图捕获期外，每次都校验
```

目的：把"静默越界"变成显式 `AssertionError`，用于定位长期漂移的写入点。
**默认不设该变量 ⇒ 行为与生产完全一致**，可安全留在树上。

⚠️ 适用范围：只覆盖 **eager 路径（prefill）**。decode 跑在 CUDA 图里，
`.max()` 在捕获期非法 ⇒ 图内 path 仍然无法校验，这是本补丁的已知盲区。
