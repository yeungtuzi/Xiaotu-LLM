# scripts/stress —— 长稳压测与退化检测工具

这里是从运行环境回收整理的一整套工具，用于对任意 OpenAI 兼容的 vLLM 端点做
**长稳压力测试** 与 **能力/退化探测**。完整说明（设计、判据、常见坑、证据格式）见：

* [`docs/STRESS_AND_DEGEN_TESTING.md`](../../docs/STRESS_AND_DEGEN_TESTING.md)

## 文件清单

| 文件 | 作用 |
|---|---|
| `replay_driver.py` | 会话回放驱动器：按轮次把真实会话重新发给服务，记录延迟/token/重复率 |
| `monitor.py` | 5 s 采样监控：健康、KV、显存、进程树 RSS、内核 oom、告警、崩溃现场快照 |
| `run_all.sh` / `stop.sh` / `status.sh` | 启动（含预检）/ 停止 / 查看 |
| `health4.sh` | 单次体检：持久缓冲、GPU、RSS、致命信号、谷值漂移、显存守卫、智力体检 |
| `intel_probe.py` | 周期性智力体检（单请求、题目轮换、机械判分） |
| `degen_report.py` | 从回放日志里抽取退化故障点，生成 markdown |
| `degen_ab.py` | 上下文长度 × max_tokens 的对照实验（区分"截断"与"真退化"） |
| `degen_suite.py` | 四类深度退化测试：长度阶梯 / 指令遗忘 / 多跳干扰 / 一致性 |

## 端口与路径

* 端点端口一律走 `--port`，默认 `8000`；实际端口以 `dev-docs/ENV_TWO_MACHINES.md` 为准。
* 语料根目录可用 `STRESS_CORPUS_ROOTS`（冒号分隔）覆盖，默认取仓库内 `docs/`、`vllm_xiaotu_moe/`、`dev-docs/`。
