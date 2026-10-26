# HANDOFF —— 当前状态（2026-10-26 阶段收尾）

> 本文件只保留**当前入口**。过程稿已归档：`dev-docs/archive/2026-10-26-stage-close/`。
> 上一版（2026-10-22，MUSA qfn 排障）原文见该目录 `HANDOFF_2026-10-22_MUSA.md`。

## 两条线的阶段结论

| 线 | 分支 | 阶段成果 | 权威文档 |
|---|---|---|---|
| **主线**（A100 / CUDA） | `dev/mainline` | draft 常驻 GPU + 2×512K KV；C=1/C=2 步时 81.5 / 113.7 ms，KV 池 4.42× | `docs/BENCH_ds41f_2026-10-26.md` |
| **MUSA**（MTT S4000 MP22×2） | `feature/musa-correctness` | 形态 A（FP8 混合）与形态 B（NVFP4 全驻留）均跑通；MP22 线收尾 | `dev-docs/MUSA_MP22_STAGE_REPORT_2026-10-26.md`（v2.0，唯一权威版） |

- **MUSA 结论**：瓶颈是每步固定开销（约 613 ms/token、12.9 ms/层，首疑每层 TP 归约与 62 µs 发射），
  属平台侧；**不再投入内核级优化**，5 个厂商问题作为会议议程，后续平台（S5000）沿用 roofline 方法验收。
- **主线结论**：2×512K 达成（2.21× 余量）；`BENCH_ds41f` §3 正确性尚未做，留档仍不完整。

## 运行状态（2026-10-26 21:xx 核实）

- **生产 8700：未运行**。最后一版 `dsv41_prod_r53` 于 10-26 12:45 正常 shutdown；
  A100 三卡 0 MiB、无计算进程、8700 未监听。
- 重启生产属“生产操作”，**须用户当次许可**（`AGENTS.md` 第 0 条）。
- 远端部署树 HEAD 停在 `156ce25`，实际运行的是逐文件 rsync 的**执行副本** ⇒ 远端 git 状态不是版本判据。

## 分支 / 工作树

- 主开发树：`/root/lvllm/repos/vllm-xiaotu-moe`（`feature/musa-correctness`）
- 主线基准 worktree：`/root/lvllm/repos/vllm-mainline`（`dev/mainline`）
- ⚠️ MUSA 分支已落后 `origin/main` **115 个提交**（非合并 96 个），merge-base 为 `b6a30b9`，**不是直系后代**。
  合并与回基路径见 `dev-docs/MERGE_MUSA_TO_MAINLINE_PLAN.md`。

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
