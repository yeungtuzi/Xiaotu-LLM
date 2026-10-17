import torch, time
dev="cuda:0"
print("device:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))
print("torch:", torch.__version__)
# 1) fp8 存储/转换是否支持
try:
    a=torch.randn(64,256,device=dev).to(torch.float8_e4m3fn)
    b=a.to(torch.bfloat16)
    print("[1] fp8 e4m3 storage+cast: OK  dtype=%s backward=%s" % (a.dtype, b.dtype))
except Exception as e:
    print("[1] fp8 storage: FAIL", str(e)[:90])
# 2) fp8 x fp8 matmul 是否支持
for tag,fn in (("torch.matmul(fp8,fp8)", None), ):
    try:
        A=torch.randn(32,4096,device=dev).to(torch.float8_e4m3fn)
        B=torch.randn(4096,4096,device=dev).to(torch.float8_e4m3fn)
        C=torch.matmul(A,B)
        print("[2] fp8 matmul: OK -> dtype", C.dtype)
    except Exception as e:
        print("[2] fp8 matmul: FAIL", str(e)[:120])
# 3) torch._scaled_mm (fp8 GEMM path)
try:
    A=torch.randn(32,4096,device=dev).to(torch.float8_e4m3fn)
    B=torch.randn(4096,4096,device=dev).to(torch.float8_e4m3fn)
    sa=torch.tensor(1.0,device=dev); sb=torch.tensor(1.0,device=dev)
    C=torch._scaled_mm(A,B,scale_a=sa,scale_b=sb,out_dtype=torch.bfloat16)
    print("[3] _scaled_mm fp8: OK ->", C.dtype)
except Exception as e:
    print("[3] _scaled_mm fp8: FAIL", str(e)[:120])
