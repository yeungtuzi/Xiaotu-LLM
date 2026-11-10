"""用真实 `_host_km_fill` 对拍：假 engine 提供 shard_geometry/hostbuf_ptr。

不启动 vLLM。直接 import 被测函数，feed 合成 shard，与「原三块切片路径」逐字节比对。
判据：torch.equal == True。
"""
import ctypes
import sys
import torch

# ⛔ 不能 `import vllm_xiaotu_moe.xxx`：包的 __init__ 会拉 vllm 依赖（本 probe 不需要）。
#   直接按文件路径加载被测模块，避免引入整套 vLLM。
import importlib.util
import types
_ROOT = "/home/mccuseradmin/s5000/xiaotu-llm"
# 模块内有 `from vllm_xiaotu_moe.byte_transpose import ...` ⇒ 需要一个**空壳包**
# 让该子模块能被解析，同时不执行真正的 __init__（那会拉 vllm）。
if "vllm_xiaotu_moe" not in sys.modules:
    _pkg = types.ModuleType("vllm_xiaotu_moe")
    _pkg.__path__ = [_ROOT + "/vllm_xiaotu_moe"]
    sys.modules["vllm_xiaotu_moe"] = _pkg
_spec = importlib.util.spec_from_file_location(
    "gpf_probe", _ROOT + "/vllm_xiaotu_moe/gpu_prefill_fp8.py")
_gpf = importlib.util.module_from_spec(_spec)
sys.modules["gpf_probe"] = _gpf
_spec.loader.exec_module(_gpf)
_host_km_fill = _gpf._host_km_fill
_HOST_KM_CACHE = _gpf._HOST_KM_CACHE

dev = torch.device("musa")
H, I, NS = 2560, 640, 2
cr13, cr2 = I // NS, H // NS          # 320, 1280
Et = 48
c13, c2 = cr13 * H, cr2 * I

torch.manual_seed(7)
# 合成 host shard（每 node 一段连续内存）
bufs = {}
for which, size in ((0, Et * 2 * c13), (1, Et * c2)):
    for n in range(NS):
        arr = torch.randint(0, 256, (size,), dtype=torch.uint8)
        bufs[(which, n)] = arr


class FakeEngine:
    def shard_geometry(self):
        return {"ns": NS, "w13_crows": cr13, "w13_cbytes": c13,
                "w2_crows": cr2, "w2_cbytes": c2}

    def hostbuf_ptr(self, which, n):
        return bufs[(which, n)].data_ptr()


eng = FakeEngine()

# ── 参照：原来的三块切片写法（复制自改动前源码）──
w13_ref = torch.zeros((Et, H, 2 * I), dtype=torch.uint8, device=dev)
w2_ref = torch.zeros((Et, I, H), dtype=torch.uint8, device=dev)
import numpy as _np
for n in range(NS):
    ptr = bufs[(0, n)].data_ptr()
    arr = _np.ctypeslib.as_array((ctypes.c_uint8 * (Et * 2 * c13)).from_address(ptr))
    t = torch.from_numpy(arr).view(Et, 2, cr13, H).permute(0, 3, 1, 2).contiguous().pin_memory()
    w13_ref[:, :, n * cr13:(n + 1) * cr13].copy_(t[:, :, 0, :])
    w13_ref[:, :, I + n * cr13:I + (n + 1) * cr13].copy_(t[:, :, 1, :])
for n in range(NS):
    ptr = bufs[(1, n)].data_ptr()
    arr = _np.ctypeslib.as_array((ctypes.c_uint8 * (Et * c2)).from_address(ptr))
    t = torch.from_numpy(arr).view(Et, cr2, I).permute(0, 2, 1).contiguous().pin_memory()
    w2_ref[:, :, n * cr2:(n + 1) * cr2].copy_(t)
torch.musa.synchronize()

# ── 被测：真实 `_host_km_fill`（整块连续写）──
_Host_Km_Cache = _HOST_KM_CACHE
_Host_Km_Cache.clear()
w13_new = torch.zeros((Et, H, 2 * I), dtype=torch.uint8, device=dev)
w2_new = torch.zeros((Et, I, H), dtype=torch.uint8, device=dev)
ok_call = _host_km_fill(eng, dev, H, I, 0, Et, w13_new, w2_new)
torch.musa.synchronize()

print(f"_host_km_fill returned: {ok_call}")
ok13 = torch.equal(w13_ref, w13_new)
ok2 = torch.equal(w2_ref, w2_new)
print(f"w13 逐字节相等: {ok13}")
print(f"w2  逐字节相等: {ok2}")
if not ok13:
    print("  w13 mismatch:", int((w13_ref != w13_new).sum()))
if not ok2:
    print("  w2  mismatch:", int((w2_ref != w2_new).sum()))
print("RESULT:", "PASS" if (ok_call and ok13 and ok2) else "FAIL")
