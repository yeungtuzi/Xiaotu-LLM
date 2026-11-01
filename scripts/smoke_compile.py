"""离线编译冒烟：不占显存、不需要启动服务，直接验证内核源码能否被 Triton 编译。

用途（2026-11-01 第三方审计 §6 第 1 条）：`_f32_to_e4m3_uint8` 曾在 @triton.jit 体内
定义嵌套 def ⇒ Triton 拒绝编译，而该函数有 13 个调用点（含 KV 写入主路径）。
本脚本在 sm80 目标上 AOT 编译一个调用它的最小内核，能把这类"编译期约束"在
服务启动前抓出来。
"""
import sys, traceback

def main():
    try:
        import triton
        from triton.compiler import ASTSource, compile as tcompile
        try:
            from triton.backends.compiler import GPUTarget
        except Exception:
            GPUTarget = None
    except Exception as e:
        print("SKIP: triton 不可用:", e); return 2

    try:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _f32_to_e4m3_uint8, _e4m3_uint8_to_f32,
        )
    except Exception as e:
        print("SKIP: 导入 fp8_utils 失败:", type(e).__name__, e); return 2

    from vllm.triton_utils import tl, triton as tt

    @tt.jit
    def _smoke_kernel(src, dst, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(src + offs, mask=mask, other=0.0)
        b = _f32_to_e4m3_uint8(x)
        tl.store(dst + offs, b, mask=mask)
        y = _e4m3_uint8_to_f32(b)
        tl.store(src + offs, y, mask=mask)

    print("triton 版本:", triton.__version__)
    ok = True
    for cap in (80, 90):
        try:
            if GPUTarget is not None:
                target = GPUTarget("cuda", cap, 32)
            else:
                target = ("cuda", cap)
            src = ASTSource(_smoke_kernel,
                            {"src": "*fp32", "dst": "*u8", "n": "i32", "BLOCK": "constexpr"},
                            constexprs={"BLOCK": 128})
            tcompile(src, target=target)
            print(f"  ✅ sm{cap} 编译通过")
        except Exception as e:
            ok = False
            print(f"  ❌ sm{cap} 编译失败: {type(e).__name__}: {e}")
            tb = traceback.format_exc().splitlines()
            for line in tb[-6:]:
                print("     ", line)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
