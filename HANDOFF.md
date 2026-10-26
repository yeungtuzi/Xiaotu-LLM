# HANDOFF —— 当前状态（2026-10-26 阶段收尾）

> 本文件只保留**当前入口**。过程稿已归档：`dev-docs/archive/2026-10-26-stage-close/`。
> 上一版（2026-10-22，MUSA qfn 排障）原文见该目录 `HANDOFF_2026-10-22_MUSA.md`。

## 合并状态（2026-10-26 已执行）

| 项 | 值 |
|---|---|
| 唯一主线 | **`main`** @ **`5cb0d63`**（tag `cuda和musa线初次合并`） |
| 原 MUSA 分支 | 已归档为 **`archive/musa-merged-20261026`**（同一提交，不再作为开发线） |
| `dev/mainline` | **已删除**（本地与远端）；历史已全部包含在 `main` |
| 回基 | `a76f6ce`（merge `origin/main`，补齐 115 个提交） |
| 吸收主线 | `7183162`（merge 阶段收尾 4 个提交） |
| 单代码守卫 | `da53c16`（`scripts/check_single_code.sh`） |
| 详见 | `dev-docs/MERGE_EXECUTION_2026-10-26.md` |

- ⭐ **终态（用户 2026-10-26 定）**：**一份代码，同时支持 CUDA 与 MUSA，用 `GPU_DEVICE` 门控**。
  MUSA 差异只在 `musa_correctness.py` 单点判定 + 同一函数内的分支；⛔ 不保留平台专用实现副本。
- **MUSA 结论**：瓶颈是每步固定开销（约 613 ms/token、12.9 ms/层，首疑每层 TP 归约与 62 µs 发射），
  属平台侧；**不再投入内核级优化**，5 个厂商问题作为会议议程，后续平台（S5000）沿用 roofline 方法验收。
- **主线结论**：2×512K 达成（2.21× 余量）；`BENCH_ds41f` §3 正确性尚未做，留档仍不完整。
- **编译验证（2026-10-26）**：合并后代码在 A100 临时副本完整编译 **8/8 变体、0 error**
  （仅 `-Wformat` 警告；`import vllm_xiaotu_moe` 通过）。**运行时正确性 / 性能 / 功能**
  由两条线各自在分支会话完成（用户 2026-10-26 定）。

## 运行状态（2026-10-26 21:33 核实）

- **生产 8700：未运行**。最后一版 `dsv41_prod_r53` 于 10-26 12:45 正常 shutdown；
  A100 三卡 0 MiB、无计算进程、8700 未监听。
- 重启生产属“生产操作”，**须用户当次许可**（`AGENTS.md` 第 0 条）。
- 远端部署树 HEAD 停在 `156ce25`，实际运行的是逐文件 rsync 的**执行副本** ⇒ 远端 git 状态不是版本判据。

## 分支 / 工作树

- 主开发树：`/root/lvllm/repos/vllm-xiaotu-moe`（`main`）
- 基准 / 编译树：`/root/lvllm/repos/vllm-mainline`（**detached @ `main`**，只读/基准，不提交）
- ⭐ **纪律：主线 = `main`，没有 `mainline`**。开发从 `main` 开短命 `topic/*`，验收后合并回 `main` 并删除；
  不要再恢复 `dev/mainline` / `feature/musa-correctness`。

## 未决 / 下一步

1. 主线 `BENCH_ds41f` §3 正确性（greedy / 工具调用 / needle / 语义；需起服务）
2. `dev-docs` 单一真源（当前两棵 worktree 各一份，本次已做**内容级合并**，架构见合并路径文档）
3. XID31 根因未钉到内核（两个候选机制，判别实验待做）
4. MUSA 报告 C=1 口径待核对（`EXPERIMENTS.md` B1075 的 31 tok/s 快路径 vs 报告的 1.6 tok/s）
5. tp=2 每 worker 内存 120 GiB（不含 PLE ≈96 GiB）仍高于目标 80 GiB（`dev-docs/TODO_MAINLINE_2026-10-22.md`）

## 纪律入口

- `AGENTS.md`（每轮注入）
- `dev-docs/report/tuning/IRON_RULES.md`（约束）
- `dev-docs/USER_QA_LEDGER.md`（错题本）
- `dev-docs/ENV_TWO_MACHINES.md`（两台机器与路径映射）
