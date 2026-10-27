# 长稳压测与退化检测

> 本文描述 `scripts/stress/` 这套工具的用途、判据与使用方式。
> 作者：`vllm-xiaotu-moe` 引擎作者。文中不预设结论；所有判据都以可复现的命令给出。

## 1. 目的与范围

面向 **OpenAI 兼容的 vLLM 端点**，回答三类问题：

1. **长稳**：连续跑数十小时，显存/宿主内存是否单调增长，服务会不会崩、崩在什么位置；
2. **能力**：在长上下文与长时运行中，模型是否还能正常给出答案（而不是只输出思考、重复、乱码）；
3. **归因**：把"崩溃"与"退化"分别量化，避免把**截断**误判成模型退化。

范围之外：不做基准排名、不替代性能基准（吞吐/延迟的正式口径见 `docs/BENCH_PROTOCOL.md`）。

## 2. 组件与数据流

| 组件 | 角色 | 产出 |
|---|---|---|
| `replay_driver.py` | 把真实会话按轮次重新发给服务（dry-run，不执行工具） | `replay.jsonl`、`driver.out` |
| `monitor.py` | 5 s 采样：`/health`、`/metrics`、GPU、进程树 RSS、内核 oom、PSI | `monitor.csv`、`alerts.log`、`incident-*/` |
| `intel_probe.py` | 周期智力体检（单请求、题目轮换、机械判分） | `intel_probe.jsonl`、`alerts.log`（`INTEL_FAIL`） |
| `health4.sh` | 一次性体检（7 项），供人工或定时任务调用 | 标准输出 |
| `degen_report.py` | 从 `replay.jsonl` 抽取退化点 | `degeneration.md` |
| `degen_ab.py` | 长度 × `max_tokens` 对照 | `degen_ab.jsonl` |
| `degen_suite.py` | 四类深度退化测试 | `degen_suite.jsonl`、`degen_suite.md` |

驱动、监控、体检三者互相独立：驱动崩了不影响监控，监控崩了不影响驱动。

## 3. 快速开始

```bash
# 1) 启动（预检不过则拒绝启动）
OUT=$HOME/stress-run1 bash scripts/stress/run_all.sh \
    --port <port> --hours 72 --mode transcript \
    --max-ctx-tokens 200000 --max-outage-min 360

# 2) 查看
bash scripts/stress/status.sh <out-dir>

# 3) 体检（7 项）
bash scripts/stress/health4.sh

# 4) 停止（只按 PID 文件停自己那两个进程）
bash scripts/stress/stop.sh <out-dir>
```

周期智力体检单独起：

```bash
setsid nohup python3 -u scripts/stress/intel_probe.py --port <port> --interval 600 \
    > <out-dir>/intel_probe.log 2>&1 < /dev/null &
```

## 4. 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--port` | `8000` | 被测端点端口；实际端口以 `dev-docs/ENV_TWO_MACHINES.md` 为准 |
| `--hours` | `72` | 墙钟时长；崩溃等待时间计入，不自动延长 |
| `--mode` | `transcript` | `transcript` 会把记录里的工具输出放回上下文（撑大上下文）；`chat` 只发用户消息 |
| `--max-ctx-tokens` | `200000` | 上下文估算上限，超过则重开会话（估算为 `字符数/4`，偏保守） |
| `--max-tokens` | `1024` | 单请求生成上限；**至少 4096 才适合判断退化**，见 §7 |
| `--max-outage-min` | `30` | 服务中断容忍；期间驱动等待，恢复即续跑 |

## 5. 判据与告警

| 告警 | 触发 | 含义 |
|---|---|---|
| `SERVICE_DOWN` | `/health` 连续 3 次失败 | 服务不可用；同时抓现场 |
| `SERVICE_RESTARTED` | 服务 PID 变化 | 实例换了，指标需按实例分段看 |
| `OOM_KILL` | 内核 `oom_kill` 计数增加 | **宿主**内存被内核回收（不是显存 OOM） |
| `LEAK_SUSPECT` | 30 分钟窗口斜率超阈值 | 显存或 RSS 单调上涨 |
| `GPU_FREE_LOW` | 最满卡剩余显存低于阈值（默认 2048 MiB） | 贴顶先兆；斜率归零后仍能看见风险 |
| `KV_NOT_RELEASED` | 无请求时 KV 使用率仍高 | KV 块未释放 |
| `INTEL_FAIL` | 智力体检判错 | 能力异常，附实际输出 |
| `MONITOR_ERROR` | 单次采样异常 | 已自愈，仅记录 |

显存判读要点：**瞬时值会骗人**。真正的泄漏信号是"按 10 分钟窗口取谷值，谷值逐窗抬升"；
谷值持平或下降说明内存在归还，属于分配器扩缩而非泄漏。

## 6. 崩溃现场

`SERVICE_DOWN` 时自动创建 `incident-<UTC 时间戳>/`：

| 文件 | 内容 |
|---|---|
| `engine-tail.log` | 引擎日志尾部（约 3 MB），**路径按当前监听端口的进程 stdout 解析** |
| `kernel-20min.txt` | `journalctl -k` 近 20 分钟（查 `Xid`、`trap int3`） |
| `nvidia-smi.txt` / `ps.txt` / `listen.txt` | 当时快照 |

另有 `engine-tail-latest.log`：每 10 分钟落一次引擎日志尾，进程被强杀也有最近记录。

## 7. 退化检测：先排除截断，再谈退化

思考型模型在 agentic 任务里会先消耗大量思考 token。若 `max_tokens` 偏小，模型会在思考中途被截断，
表现为"只有 reasoning、`finish_reason=length`、正文为空"。**这是预算不足，不是退化。**

判定流程：

1. `degen_report.py`：统计形态分布与按上下文档位的比例；
2. `degen_ab.py`：同一上下文、不同 `max_tokens` 各发若干次，看"答出率"是否随预算变化；
3. 只有当 **`max_tokens` 充足**时仍出现高重复、乱码、或答非所问，才计为退化。

实测口径（2026-10-27，受控总结类任务）：

| 上下文 | `max_tokens` | 答出率 | 仅思考撞限 |
|---|---|---|---|
| 92k | 1024 / 4096 | 4/4 / 4/4 | 0 |
| 156k | 1024 / 4096 | 4/4 / 4/4 | 0 |
| 219k | 1024 / 4096 | 4/4 / 4/4 | 0 |

⇒ 上表说明：**长上下文本身不导致答不出**；回放中出现的"只有思考"主要是任务复杂度与预算不匹配。

`degen_suite.py` 提供四类更敏感的自建测试（均为本地生成、无需下载数据集）：

| 子测试 | 测什么 | 机械判据 |
|---|---|---|
| `ladder` | 同一问题在不同上下文长度下的稳定性 | 答出率 / 重复率 |
| `instruction` | 长上下文末尾的硬约束是否仍被遵守 | JSON 可解析 + 键齐全 + 前缀 + 字数 + 中文 |
| `multihop` | 埋入事实 + 结构相似的干扰事实 | 期望实体是否出现在正文 |
| `consistency` | 温度 0 重复同一 prompt | 与首次输出的相似度 |

## 7.1 并发（C=2）验证

`conc_c2.py` 补的是长稳压测覆盖不到的一层：驱动是**单请求串行**，只走到 `max-num-seqs=1` 的路径；
C=2 才会真正触发批处理、KV 块分配/回收与调度。

判分重点不是吞吐，而是**串话检测**：每轮同时发两条 prompt，各自埋一个不同的
「代号 → 负责人」事实，然后检查

| 结果 | 含义 |
|---|---|
| 两个都对 | 并发路径正确 |
| 互相串了（A 的答案出现在 B） | KV/批处理正确性缺陷，比吞吐下降严重得多 |
| 都答不出 | 能力或调度问题 |

```bash
python3 scripts/stress/conc_c2.py --port <port> --rounds 20 --ctx-tokens 8000
```

建议节奏：**24 小时长稳达成后**先跑 20 轮 8k，再用 `--ctx-tokens 32000` 跑 5 轮做大 KV 压力。

## 8. 智力体检题库

`intel_probe.py` 内置 10 题循环，覆盖：多步算术、速率、百分比、序关系传递、三段论、
Python 表达式求值、右结合幂、指令 JSON、中文计数、8k 上下文埋针。判分为正则匹配期望值，
失败写 `INTEL_FAIL`。题号持久化在 `intel_probe.idx`，重启后续走。

约束：**单请求**。与服务并发上限（`--max-num-seqs`）配合时不会额外压垮服务。

## 9. 已知坑

1. **日志会搬家**：部署目录变化后，按固定路径取日志会一直"成功"地读旧文件。本工具改为按
   `监听端口 → /proc/<pid>/fd/1` 解析，并在解析失败时退回"近 10 分钟被写过的 .log"。
2. **端点会换端口**：监控在连续 DOWN 时会扫描 `vllm.entrypoints / api_server` 的监听端口并自动跟随；
   驱动器把端口写在启动参数里，换端口需要按 PID 文件重启驱动器。
3. **监控自身可能被单点异常打死**：采样主体已包异常处理，单次失败只记 `MONITOR_ERROR`。
4. **外部会反复启停服务**：驱动器用 `OUTAGE_START/END` 标注中断，恢复后自动续跑；墙钟不延长。
5. **报错掩盖退出码**：构建/关键命令应重定向到日志后判退出码，不要 `| tail`。
6. **数据集下载**：本机直连可访问大多数站点；受网络限制的站点需代理（地址见内部说明）。
   建议在可控环境下载后同步到执行机，避免把外网依赖引入长稳测试。

## 10. 数据格式

`replay.jsonl` 每行一条请求：

```
ts, epoch, session, turn, mode, latency_s, finish_reason,
prompt_tokens, completion_tokens, ctx_est_tokens, out_chars, reasoning_chars,
reasoning_only, repeat_ratio, garbage_ratio
(+ 可疑应答额外带 content_head / reasoning_head / reasoning_tail)
```

`monitor.csv` 列：`ts, health_ok, health_ms, running, waiting, kv_usage, gpu_mem_sum, gpu_mem_max,
gpu_util_max, gpu_temp_max, gpu_power_sum, mem_total_gib, mem_avail_gib, psi_some10, load1,
oom_kill, server_pid, procs, rss_sum_mib, rss_max_mib, threads_sum, fds_sum,
prompt_tokens_total, gen_tokens_total, requests_ok_total, preemptions_total`

`intel_probe.jsonl` 每行一题：`ts, item, expect_note, prompt_tokens, completion_tokens,
finish_reason, latency_s, ok, answer_head, reasoning_tail, repeat_ratio`。
