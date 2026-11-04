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
