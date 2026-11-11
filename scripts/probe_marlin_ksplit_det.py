#!/usr/bin/env python3
"""验证 `moe_marlin_fused.cu` 的**确定性归约**（`run_det`）：结果是否与 `KSPLIT=1` 逐位相同。

按「局部验证纪律」：只测内核，不加载权重、不起服务。

做法：
  1. 构造随机 MXFP4 权重与激活（scale 固定为 1.0 的 bf16 位型，避免 inf/NaN）；
  2. 参考值 = `run(ksplit=1)`（普通写，天然确定）；
  3. 对照 1 = `run(ksplit=4)`（原子累加，顺序不定）；
  4. 对照 2 = `run_det(ksplit=4, P)`（两段式固定顺序归约）；
  5. 判据：`run_det` **连跑 10 次逐位相同**（这是验收 A 需要的性质），且与参考值相对误差 ≤1e-3；
     原子版允许同样量级的数值差异，但**不保证**每次相同（这正是缺陷）。

用法（远端，需要 nvcc）：
    NVCC=/home/user/lvllm/tmp/cuda13/bin/nvcc \
    XIAOTU_MOE4_MARLIN=1 XIAOTU_MOE4_KSPLIT=4 \
    python3 scripts/probe_marlin_ksplit_det.py
退出码：0 = 确定性归约正确；1 = 不一致；2 = 环境问题（编译失败/无扩展）。
"""
from __future__ import annotations

import ctypes
import os
import sys

import torch


def main() -> int:
    os.environ.setdefault("XIAOTU_MOE4_MARLIN", "1")
    from vllm_xiaotu_moe import moe4_marlin as M

    lib = M._build()
    if lib is None:
        print("⛔ 内核不可用（编译失败或缺 nvcc）")
        return 2

    torch.manual_seed(0)
    dev = torch.device("cuda:0")
    E, SLOT = 4, 16
    K, N = 256, 256                      # K 需为 64 的倍数；N 需为 128 的倍数
    NA = SLOT

    A = torch.randn(E * SLOT, K, dtype=torch.bfloat16, device=dev)
    W = torch.randint(0, 256, (E, K // 2, N), dtype=torch.uint8, device=dev)
    SC = torch.full((E, K // 32, N), 127, dtype=torch.uint8, device=dev)   # 127<<7 = bf16 1.0
    P = torch.empty((4 - 1) * E * SLOT * N, dtype=torch.float32, device=dev)
    stream = torch.cuda.current_stream().cuda_stream

    def p(t):  # noqa: E741
        return ctypes.c_void_p(t.data_ptr())

    def call(ksplit: int, det: bool) -> torch.Tensor:
        C = torch.zeros(E * SLOT, N, dtype=torch.float32, device=dev)
        if det:
            rc = lib.run_det(p(A), p(W), p(SC), p(C), p(P), NA, K, N, E,
                             N // 128, ksplit, stream)
        else:
            rc = lib.run(p(A), p(W), p(SC), p(C), NA, K, N, E, N // 128, ksplit, stream)
        torch.cuda.synchronize()
        assert rc == 0, f"内核返回 {rc}"
        return C.clone()

    ref = call(1, det=False)

    det_runs = [call(4, det=True) for _ in range(10)]
    atom_runs = [call(4, det=False) for _ in range(10)]

    det_stable = all(torch.equal(r, det_runs[0]) for r in det_runs)
    atom_stable = all(torch.equal(r, atom_runs[0]) for r in atom_runs)
    # 判据只看「同输入多次运行是否逐位相同」，不要求与 ksplit=1 逐位相同
    # （K 的求和顺序不同 ⇒ 允许浮点级差异；相对误差 1e-3 足够严）。
    det_close = torch.allclose(det_runs[0], ref, rtol=1e-3, atol=0)
    atom_close = torch.allclose(atom_runs[0], ref, rtol=1e-3, atol=0)
    det_max = (det_runs[0] - ref).abs().max().item()
    atom_max = (atom_runs[0] - ref).abs().max().item()

    print(f"参考(ksplit=1) 与 run_det 最大绝对差 = {det_max:.3e}")
    print(f"参考(ksplit=1) 与 原子版   最大绝对差 = {atom_max:.3e}")
    print(f"run_det 与参考 相对误差 1e-3 内 : {'✅' if det_close else '❌'}")
    print(f"run_det 连跑 10 次完全一致      : {'✅' if det_stable else '❌'}")
    print(f"原子版 连跑 10 次完全一致       : {'✅' if atom_stable else '❌（预期：顺序不定）'}")
    print(f"原子版 与参考 相对误差 1e-3 内  : {'✅' if atom_close else '❌'}")

    ok = det_stable and det_close
    print("== 判定：" + ("确定性归约正确 ✅" if ok else "存在问题 ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
