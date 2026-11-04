# patches/maintenance —— 维护类补丁（**不**进 `xtu-series/series`）

与 `patches/xtu-series/` 的区别：

| | `xtu-series/` | `maintenance/` |
|---|---|---|
| 内容 | 我们对上游的**功能/修复**补丁，按序应用到干净上游 | 处置**别人在树上留下的在途改动**（撤销/回退/临时规避） |
| 是否进 `series` | 是（有序） | **否**（按需手工应用） |
| 可否重放到新上游 | 是 | 通常否——它依赖"污染版本"的具体内容 |

## 清单

| 文件 | 用途 | 日期 |
|---|---|---|
| `cleanup-dspark-remote-scaffold-20261104.patch` | 移除 `spec_decode/dspark/speculator.py` 里的 DSpark-remote 脚手架（`_IdentityDraftModel` / `_xtu_remote` / `load_draft_model` 与 `set_attn` 的 remote 分支）。用户 2026-11-04「GPU0独立mtp已经废除」；这些符号在树内已零引用。**不动** `XTU_VARLEN_K` 变长草稿特性。 | 2026-11-04 |
| `revert-dspark-remote-mtp-20261104.patch` | 撤销 `spec_decode/dflash/speculator.py` 里已废除的 DSpark-remote（GPU0 独立 MTP）代码，回到上游 HEAD。来由：该段代码直接访问 `self.model.main_norm/main_proj`，而 remote 回落的 `_IdentityDraftModel` 只有 `combine_hidden_states()` ⇒ 启动 AttributeError；即便绕过，identity 不投影导致 15360 vs 5120 尺寸不匹配。上游 HEAD 本来用多态调用，无此 bug。 | 2026-11-04 |

## 使用

```bash
# 在含污染代码的 vllm 树里反向应用（撤销）
git apply -R patches/maintenance/<name>.patch
```

应用前先确认该文件确实是"污染版本"（本地 HEAD 不含被撤销的行）：
```bash
git diff --stat -- vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
git show HEAD:vllm/v1/worker/gpu/spec_decode/dflash/speculator.py | grep -n main_norm
```

## 撤回记录（**不要重新加回来**）

| 位置 | 撤回内容 | 原因 |
|---|---|---|
| `vllm/models/deepseek_v41/common/ops/cache_utils.py` `compute_global_topk_indices_and_lens` | `assert token_to_req_indices.max() < block_table.shape[0]` | 调用点传的是 `attn_metadata.block_table[:num_decodes]`（按 decode 数切片），而 `token_to_req_indices` 是**全局请求号** ⇒ 混合 prefill+decode 时 `num_decodes=1` 但 token 覆盖 2 个请求 ⇒ 假报。2026-11-04 14:17 把生产 assert 死（单请求正常、加第二路并发即崩）。本函数只能断言**结构性契约**（`stride(1)==1`、`stride(0)==shape[1]*stride(1)`），取值上界取决于调用方切片语义。 |
