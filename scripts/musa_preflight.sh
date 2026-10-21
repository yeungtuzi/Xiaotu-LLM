#!/usr/bin/env bash
# MUSA/S4000 preflight: safe on machines without MUSA hardware.
set -u
echo "===== MUSA preflight $(date -Is) ====="
echo "host: $(hostname)  kernel: $(uname -r)  arch: $(uname -m)"
echo "cpu: $(lscpu 2>/dev/null | awk -F: '/Model name/{gsub(/^ +/,"",$2);print $2;exit}')"
echo "nproc: $(nproc)  mem: $(free -h 2>/dev/null | awk '/Mem:/{print $2}')"
echo "--- CPU flags ---"
lscpu 2>/dev/null | awk -F: '/Flags/{print $2}' | tr ' ' '\n' | grep -E '^(avx|avx2|avx512f|avx512_bf16|amx)' | sort -u | tr '\n' ' '; echo
echo "--- PCIe MUSA devices ---"
lspci 2>/dev/null | grep -iE 'moore|mthreads|musa' || echo "(none)"
echo "--- tools ---"
for c in musaInfo mthreads-gmi mthreads-smi nvidia-smi; do
  if command -v "$c" >/dev/null 2>&1; then echo "$c: $(command -v $c)"; else echo "$c: MISSING"; fi
done
echo "--- xiaotu CPU engine .so deps ---"
for so in xiaotu_moe/build/*.so; do
  [ -e "$so" ] || continue
  echo "$so"
  ldd "$so" 2>/dev/null | grep -E 'libcudart|libcuda|not found' || echo "  (no cuda deps)"
done
echo "--- python packages ---"
python3 - <<'PY' 2>/dev/null || true
mods=['torch','torch_musa','vllm','vllm_musa','triton','torchada']
for m in mods:
    try:
        x=__import__(m)
        print(m, getattr(x,'__version__','?'))
    except Exception as e:
        print(m, 'MISSING', type(e).__name__)
PY
echo "--- env ---"
env | grep -E '^(MUSA|TORCH_MUSA|VLLM_MUSA|CUDA_VISIBLE_DEVICES|MTHREADS)' | sort || true
echo "===== done ====="
