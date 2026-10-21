# MUSA / S4000 正确性验证

目标：在 **MTT S4000 + 海光 x86 + 256GB** 上先验证 **官方 Qwen3.8-Flash-Next FP8 混合推理**的正确性；FP8 通过后立即转 NVFP4。**永不 BF16**。不追求性能。

- [总计划](MUSA_S4000_CORRECTNESS_PLAN.md)
- [端口映射](MUSA_PORT_MAP.md)
- [版本/仓库锁定](MUSA_VERSION_MATRIX.md)
- [qwen4_exp/MUSA 审计](MUSA_QWEN4EXP_AUDIT.md)
- [候选补丁](../../patches/musa/README.md)
- [上线 Runbook](MUSA_BRINGUP_RUNBOOK.md)
- [正确性对拍协议](MUSA_CORRECTNESS_PROTOCOL.md)
- [preflight 脚本](../../scripts/musa_preflight.sh)

分支：`feature/musa-correctness`（验证通过后再合并到 `main`）。

版本：**vllm-musa `v0.32.0-dev` + 官方 Qwen3.8-Flash-Next FP8**；qwen4_exp 在 vLLM 0.32 才有。

当前状态：设备未上线；先做离线准备。设备上线后从 Phase 1 开始。
