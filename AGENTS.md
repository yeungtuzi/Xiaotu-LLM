# AGENTS.md —— vLLM 主线仓(本仓库只做“跟上游 + 打 patch”)

> **2026-10-25 精简**:本文件原为与 `vllm-xiaotu-moe/AGENTS.md` 近乎重复的 783 行副本(两份仅差 59 行)。
> 现在只保留**本仓特有**的纪律;**其余全部规则以 `../vllm-xiaotu-moe/AGENTS.md` 为唯一权威源**
> (第一纪律“不许阻塞前台”、进程管理、重启许可、文档独占锁、R38 审计、敏感信息等)。

## 1. 本仓定位

* 本仓 = **vLLM 上游的只跟版**(`dev/mainline`),**不放自有长命特性分支**;
* 我们的功能一律以 patch 形式留在 `vllm-xiaotu-moe/patches/xtu-series/`(唯一真源) ✓

## 2. patch 纪律

* 生成:`git -C <本仓> format-patch origin/main..<分支> -o ../vllm-xiaotu-moe/patches/xtu-series --no-signature`
  ⇒ **每次 rebase 后重新生成**,不手改 patch ✗
* 应用:`vllm-xiaotu-moe/scripts/apply_xtu_patches.sh <含 vllm/ 的目录>`;失败**即报错**,
  ⛔ 禁止 `patch -f`(会静默跳过 hunk)✗
* rebase 流程:导出**纯上游快照**(`git archive` 到 /tmp,不注册 worktree)→ 全量应用 → 逐文件比对 →
  通过后再更新**在服务的树**(该步需用户许可)✓

## 3. 工作树

* 同时最多 **唯一主线 + 至多一条在用开发树**;临时预演只在 `/tmp`,用完 `git worktree prune`;
  机械自检:`git worktree list` 应 ≤ 2 条(排除 prunable)✓

## 4. 文档与对外

* `README.md` / `docs/` = **发布面**,只放外部用户现在就该看的内容;
  规划/设计/报告进 `vllm-xiaotu-moe/dev-docs/`(gitignored)✓
* README 的版本段 = **一句话 + 链接**,不展开细节/口径表/性能数字 ✓
* 面向**别人项目**的 GitHub 写操作(PR/issue/comment/push 到 fork 的 PR 分支)必须**逐条**取得用户批准;
  读取类不受限;本仓自己的提交/推送不受限(发布类 tag/release 仍需点头)✓

## 5. 提交前自检

```bash
vllm-xiaotu-moe/scripts/check_no_secrets.sh          # 扫已暂存;退出码 1 = 命中 ⇒ 禁止提交
```

⛔ 禁止把内网 IP/端口、代理地址、凭据、私钥、内网主机名写进任何会被提交的文件 ✓

## 6. 其余

* 环境是两台机器(harness VM + gpu 执行器):`ssh -F /root/lvllm/tmp/sshcfg gpu`,
  本机没有 GPU/vLLM/权重;`/home/user/...` 默认是**远端**路径 ⇒ 详见
  `vllm-xiaotu-moe/dev-docs/ENV_TWO_MACHINES.md` ✓
* 一切“重启服务 / 杀进程 / 起第二个实例 / 改默认值”的动作,遵循
  `vllm-xiaotu-moe/AGENTS.md` 与 `dev-docs/report/tuning/IRON_RULES.md` ✓
