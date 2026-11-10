#!/usr/bin/env bash
# check_pc_expandable.sh —— 纯函数判据：磁盘前缀缓存（KV connector）与 expandable_segments **硬互斥**
#
# 背景：vLLM `config/vllm.py::_verify_kv_transfer_compat()` 明确规定——
#   配置了任何 KV connector 时，`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 会被拒
#   （除非启用 cumem allocator，而 cumem 自己又会把 expandable 关掉）。
# 我们生产口径是 `PC=0`（不配 connector）+ `EXPANDABLE=1` ⇒ 相容；
# 但若有人写了 `PC=1`，服务会**在启动约 5 分钟后**因配置校验失败退出，
# 而这一趟重启代价 20–40 分钟。故把它做成可离线验证的判据。
#
# 用法： bash scripts/check_pc_expandable.sh "<启动命令行>" "<PYTORCH_CUDA_ALLOC_CONF 值>"
# 退出码：0 = 相容；2 = 冲突（不许启动）
set -uo pipefail
cmd=${1:-}
alloc=${2:-}

has_connector=0
case "$cmd" in
    *--kv-transfer-config*|*--kv-offloading-size*) has_connector=1 ;;
esac
has_expandable=0
case "$alloc" in
    *expandable_segments:True*) has_expandable=1 ;;
esac

if [ "$has_connector" = 1 ] && [ "$has_expandable" = 1 ]; then
    echo "⛔ 冲突：命令行含 KV connector（磁盘前缀缓存），但 PYTORCH_CUDA_ALLOC_CONF=$alloc"
    echo "   vLLM 会在 _verify_kv_transfer_compat() 处拒绝启动（5 分钟后才报）"
    echo "   二选一：PC=0（保留 expandable＝控制显存增长）或 EXPANDABLE=0（保留磁盘前缀缓存）"
    exit 2
fi
if [ "$has_connector" = 1 ]; then
    echo "✅ 有 KV connector，但 expandable 未开 ⇒ 相容（注意：显存增长由 connector 侧负责）"
    exit 0
fi
if [ "$has_expandable" = 1 ]; then
    echo "✅ 无 KV connector（PC=0）+ expandable ⇒ 相容（当前生产口径）"
    exit 0
fi
echo "✅ 两者都没有 ⇒ 相容"
exit 0
