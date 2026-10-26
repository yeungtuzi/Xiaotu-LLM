#!/usr/bin/env bash
# MCCL 最小复现（提交厂商用）—— 单 rank 也会失败，与多卡无关。
#
# 用法：bash mccl_repro.sh [world_size]
# 环境：MUSA 5.2.0 驱动 / mccl-s4000 2.4.0 / torch 2.11.0.post1 + torch_musa 2.11.0.post1+b431668
set -uo pipefail

WORLD=${1:-1}
PORT=${PORT:-29700}
WORK=$(mktemp -d)
cat > "$WORK/t.py" <<'PY'
import os
import torch
import torch_musa  # noqa: F401
import torch.distributed as dist

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
dist.init_process_group(backend="mccl", rank=rank, world_size=world)
torch.musa.set_device(rank)
t = torch.ones(4, device="musa") * (rank + 1)
dist.all_reduce(t)
torch.musa.synchronize()
if rank == 0:
    print("ALL_REDUCE_OK", t.tolist(), flush=True)
dist.destroy_process_group()
PY

echo "=== MCCL repro: world_size=$WORLD ==="
MCCL_DEBUG=${MCCL_DEBUG:-INFO} torchrun --nproc_per_node="$WORLD" \
  --master_port="$PORT" "$WORK/t.py" 2>&1 | tail -40

echo
echo "=== 期望 ==="
echo "  ALL_REDUCE_OK [<world*(world+1)/2> ...]"
echo "=== 实测（本机）==="
echo "  MCCL WARN misc/musawrap.cc:126  Musa failure 101 'invalid device ordinal'"
echo "  RuntimeError: MCCL error ... unhandled system error, MCCL version 2.30.4"
echo "  注：world_size=1 亦复现；MCCL_P2P_DISABLE/MCCL_IB_DISABLE/MCCL_SOCKET_IFNAME 均无效"
rm -rf "$WORK"
