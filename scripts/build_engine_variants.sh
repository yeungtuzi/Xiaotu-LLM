#!/usr/bin/env bash
# Build the bundled xiaotu engine in all ISA variants (one .so per ISA level).
#
# 本脚本构建的是 **本仓库内置的引擎**(`xiaotu_moe/csrc`),不依赖外部仓库。
# 同一份 `python_binding/binding.cpp` 编译多次,每次用不同的
# ISA 宏和一个独立的 pybind 模块名 `_xiaotu_moe_C_<suffix>`;`xiaotu_moe/loader.py`
# 在导入时按 /proc/cpuinfo 选最高可用变体。
#
# 变体阶梯(越高越新,优先选):
#   scalar          无 SIMD(兜底)
#   avx2            -mavx2 -mfma
#   avx512_base     -mavx512f -mavx512bw -mavx512vl -mavx512dq
#   avx512_vnni     base + -mavx512vnni
#   avx512_bf16     base + -mavx512bf16
#   avx512_amx      AMX(未接线,不构建)
#
# 用法:
#   PYTHON=/path/to/venv/bin/python scripts/build_engine_variants.sh
#   PYBIND11_INC=... CUDART_ROOT=... 可覆盖自动探测
#
# License: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../xiaotu_moe" && pwd)"
SRC="$ROOT/csrc/python_binding/binding.cpp"
OUT_DIR="$ROOT/build"
mkdir -p "$OUT_DIR"

PYTHON="${PYTHON:-python3}"
PY_INC="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
PY_EXT="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

# pybind11(header-only):优先环境变量,其次常见位置
if [[ -z "${PYBIND11_INC:-}" ]]; then
  # 【2026-09-17 修】本 env 里**没有独立的 pybind11 包**,但 **torch 自带一份**头文件
  # (`site-packages/torch/include/pybind11/`)。原来只试 `import pybind11` 和 /usr/include,
  # 于是构建立在 `exit 2` 上 —— 而 `BUILD_EXIT` 若被 `| tail` 吞掉,门禁就会**去验旧的 .so**。
  # 注意 `-I"$PYBIND11_INC"` 要求这里是**包含 `pybind11/` 的那个目录**(不是 pybind11 本身)。
  for cand in \
      "$("$PYTHON" -c 'import pybind11, os; print(os.path.join(pybind11.get_include()))' 2>/dev/null)" \
      "$("$PYTHON" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__),"include"))' 2>/dev/null)" \
      /usr/include /usr/local/include ; do
    [[ -n "$cand" && -d "$cand" ]] && PYBIND11_INC="$cand" && break
  done
fi
if [[ -z "${PYBIND11_INC:-}" || ! -d "$PYBIND11_INC" ]]; then
  echo "!! pybind11 include dir not found; set PYBIND11_INC" >&2
  exit 2
fi

# CUDA runtime:binding.cpp 用 cudaStream_t/cudaMemcpyAsync 等。
# 注意 pip 的 nvidia/cu13 包只有 libcudart.so.13(没有 .so 符号链接),
# 所以这里同时探测目录与真实库名,用 -l:<soname> 兜底。
_cudart_candidates() {
  [[ -n "${CUDART_ROOT:-}" ]] && echo "$CUDART_ROOT"
  "$PYTHON" -c 'import nvidia.cu13, os; print(os.path.dirname(nvidia.cu13.__file__))' 2>/dev/null || true
  echo /usr/local/cuda
}
CUDART_INC=""; CUDART_LIBDIR=""; CUDART_SONAME=""
while IFS= read -r cand; do
  [[ -n "$cand" && -d "$cand/include/crt" ]] || continue
  for libdir in "$cand/lib" "$cand/lib64"; do
    if [[ -e "$libdir/libcudart.so" ]]; then
      CUDART_INC="$cand/include"; CUDART_LIBDIR="$libdir"; CUDART_SONAME="libcudart.so"; break
    fi
    for f in "$libdir"/libcudart.so.*; do
      if [[ -e "$f" ]]; then
        CUDART_INC="$cand/include"; CUDART_LIBDIR="$libdir"
        CUDART_SONAME="$(basename "$f")"; break
      fi
    done
    [[ -n "$CUDART_SONAME" ]] && break
  done
  [[ -n "$CUDART_SONAME" ]] && break
done < <(_cudart_candidates | sort -u)

CUDA_FLAGS=()
NO_CUDA="${NO_CUDA:-0}"
if [[ "$NO_CUDA" == "1" ]]; then
  CUDA_FLAGS=(-DXIAOTU_MOE_NO_CUDA)
  echo ">> NO_CUDA=1: building CPU-only engine (no cudart/libcuda)"
else
if [[ -n "$CUDART_SONAME" ]]; then
  CUDA_FLAGS=(-I"$CUDART_INC" -L"$CUDART_LIBDIR"
              -Wl,--no-as-needed -l:"$CUDART_SONAME" -Wl,--as-needed
              -Wl,-rpath,"$CUDART_LIBDIR")
  echo ">> CUDA runtime: $CUDART_INC / $CUDART_LIBDIR ($CUDART_SONAME)"
else
  echo "!! CUDA runtime not found; set CUDART_ROOT (need include/crt + libcudart)" >&2
  exit 2
fi

# CUDA **驱动** API:item 1/3 的 `cuStreamWriteValue32`/`cuStreamWaitValue32` 在 libcuda。
# 不链接会在 `import` 时报 `undefined symbol: cuStreamWriteValue32_v2`(已踩)。
DRV_LIBDIR=""; DRV_SONAME=""
for d in /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib /usr/local/cuda/lib64/stubs; do
  if [[ -e "$d/libcuda.so.1" ]]; then DRV_LIBDIR="$d"; DRV_SONAME="libcuda.so.1"; break; fi
  if [[ -e "$d/libcuda.so" ]];   then DRV_LIBDIR="$d"; DRV_SONAME="libcuda.so";   break; fi
done
if [[ -n "$DRV_SONAME" ]]; then
  CUDA_FLAGS+=(-L"$DRV_LIBDIR" -Wl,--no-as-needed -l:"$DRV_SONAME" -Wl,--as-needed)
  echo ">> CUDA driver: $DRV_LIBDIR ($DRV_SONAME)"
else
  echo "!! CUDA driver lib not found (libcuda.so.1 is required for cuStreamWriteValue32)" >&2
  exit 2
fi
fi  # NO_CUDA

COMMON="-std=c++17 -shared -fPIC -O3 -ffast-math -fno-finite-math-only"
# 【§629c】编译器可切(`CXX=g++-14` 或 conda 的
# `x86_64-conda-linux-gnu-g++`)。用来对照 lk_moe 的 GCC 13.3.1 世代与 Clang。
# 【2026-10-29 实测】GCC 16 在真实 decode batch（BS=6）上比 GCC 11 快 7-8%，
# 且生成物与原 .so **逐字节相同**（sha 6578c321…）⇒ 零数值风险的免费收益。
# 未显式指定 CXX 时，自动选可用的最新 g++；都没有则回落到 g++。
if [[ -n "${CXX:-}" ]]; then
  CXX_BIN="$CXX"
else
  CXX_BIN="g++"
  for _c in g++-16 g++-15 g++-14; do
    if command -v "$_c" >/dev/null 2>&1; then CXX_BIN="$_c"; break; fi
  done
  unset _c
fi
echo ">> CXX=$CXX_BIN"
# 【§630b】`MTUNE=<cpu>`:`-mtune` 只影响**调度**,不改 ISA ⇒ 不会破坏 ISA 变体阶梯
# (绝不能用 `-march=znver4`,那会把 AVX-512 塞进 scalar/avx2 兜底变体)。
MTUNE="${MTUNE:-}"
# 【§631】`EXTRA_FLAGS` 透传额外宏(如 `-DXIAOTU_MOE_FOLD_SCALE=1`),用于 A/B。
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
[[ -n "$MTUNE" ]] && COMMON="-mtune=$MTUNE $COMMON"
[[ -n "$EXTRA_FLAGS" ]] && COMMON="$COMMON $EXTRA_FLAGS"
echo ">> compiler: $($CXX_BIN --version | head -1)"
echo ">> COMMON=$COMMON"

VARIANTS=(
  "scalar|"
  "avx2|-mavx2 -mfma"
  "avx512_base|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma"
  "avx512_vnni|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mfma"
  "avx512_bf16|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mfma"
  "avx512_bf16_vbmi|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vbmi -mfma"
  # 【W4A8 / int8 激活】bf16 + vbmi + vnni：承载 int8 激活路径（`_mm512_dpbusd_epi32` 需要 `-mavx512vnni`）。
  # ⚠️ 这一档**故意不进 `loader._LADDER`**：用 `XIAOTU_MOE_VARIANT=avx512_bf16_vbmi_vnni` 显式选，
  #    以免"合并即改变生产默认路径"（见 /home/user/lvllm/archive/vllm-xiaotu-moe/2026-10-07-dev-docs-mywork/mywork/DECISIONS.md(原 dev-docs/mywork/ 已于 2026-10-07 取消) **D11**）✓
  "avx512_bf16_vbmi_vnni|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vbmi -mavx512vnni -mfma"
  # 【H1 2026-10-28·受控实验】与 `avx512_bf16_vbmi` 同旗标，只多一条内层软件预取
  #   （`XIAOTU_MOE_PF_DIST=8` ⇒ 预取当前行前方 2 条 cache line）。
  #   用途：单流 decode 的"未命中并行度"假设 A/B；数值零影响（只读提示）。
  #   ⚠️ 不进 `loader._LADDER`：用 `XIAOTU_MOE_VARIANT=avx512_bf16_vbmi_pf` 显式选。
  "avx512_bf16_vbmi_pf|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vbmi -mfma -DXIAOTU_MOE_PF_DIST=8"
  # H1 距离扫描：4（1 条 line）/ 16（4 条 line），与 8 一起构成单窗口可跑完的三点
  "avx512_bf16_vbmi_pf4|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vbmi -mfma -DXIAOTU_MOE_PF_DIST=4"
  "avx512_bf16_vbmi_pf16|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vbmi -mfma -DXIAOTU_MOE_PF_DIST=16"
  # 【2026-10-30 追加】更远的预取距离：行宽 2560 B（160 组），dist 只受 `g+dist<group_count` 限制。
  #   动机：8 组=128 B 仅 2 条 cache line；若瓶颈是未命中并行度，32/64 组（512 B/1 KB）应更快。
  #   只读提示、数值零影响；不进 `_LADDER`，用 `XIAOTU_MOE_VARIANT=` 显式选。
  "avx512_bf16_vbmi_pf32|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vbmi -mfma -DXIAOTU_MOE_PF_DIST=32"
  "avx512_bf16_vbmi_pf64|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vbmi -mfma -DXIAOTU_MOE_PF_DIST=64"
  # 【2026-10-30 追加】tile 形状：MR=4 + NR=4（acc 16 + av 8 + 5 = 29 zmm，不溢出）。
  #   动机：每 16 B 权重要载 12 条激活 zmm（MR=6 时），激活流量是权重的 ~24–48 倍；
  #   提高 NR 让一次激活载入喂更多输出列。代价：M=6 被切成 4+2，权重读 1.5 次。
  "avx512_bf16_vbmi_mr4nr4|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vbmi -mfma -DXIAOTU_MOE_GEMM_MR=4 -DXIAOTU_MOE_GEMM_NR_CT=4"
  # 【受控实验用·不用于生产】同样的旗标(含 -mavx512vnni)但**强制 base 实现** ⇒
  #   把"旗标对共享 bf16 路径 codegen 的影响"与"W4A8 源码的影响"分离 ✓（线③② 归因）
  "avx512_bf16_vbmi_vnni_base|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vbmi -mavx512vnni -mfma -DXIAOTU_MOE_FORCE_PACKED4_BASE=1"
)

# 【dev 便利】`ONLY=<空格分隔的变体名>` ⇒ 只构建列出的变体（默认全建）。
# 用途：快速迭代单个变体，避免每次都把 6 档全编一遍（每次构建都是几分钟）。
ONLY="${ONLY:-}"

for entry in "${VARIANTS[@]}"; do
  name="${entry%%|*}"
  flags="${entry#*|}"
  if [[ -n "$ONLY" && " $ONLY " != *" $name "* ]]; then
    echo ">> skip variant [$name] (ONLY=$ONLY)"
    continue
  fi
  mod="_xiaotu_moe_C_${name}"
  # 【2026-10-30】小 M 4-bit 分组 MoE 内核（moe4_smallm.cu）：
  # .cu 必须用 nvcc 单独编译成 .o 再参与链接（$CXX_BIN 编不了 CUDA）。
  # 默认**不编译**（BUILD_MOE4=1 才编）⇒ 缺 nvcc 的机器行为不变。
  # 全变体共用同一个 .o（源码与 arch 与变体名无关），只编一次。
  if [[ "$NO_CUDA" == "1" || "${BUILD_MOE4:-0}" != "1" ]]; then
    OBJ4=""
  else
    if [[ -z "${OBJ4:-}" ]]; then
      NVCC_BIN="${NVCC:-}"
      if [[ -z "$NVCC_BIN" ]]; then
        NVCC_BIN="$(command -v nvcc || true)"
        [[ -z "$NVCC_BIN" && -x /usr/local/cuda/bin/nvcc ]] && NVCC_BIN=/usr/local/cuda/bin/nvcc
        [[ -z "$NVCC_BIN" && -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]] && NVCC_BIN="$CUDA_HOME/bin/nvcc"
      fi
      if [[ -z "$NVCC_BIN" ]]; then
        echo "!! BUILD_MOE4=1 但找不到 nvcc（设 NVCC= 指定路径）" >&2
        exit 4
      fi
      OBJ4="${TMPDIR:-/tmp}/moe4_smallm.o"
      "$NVCC_BIN" -c -O3 -Xcompiler -fPIC \
        -gencode "arch=compute_${NVCC_SM:-80},code=sm_${NVCC_SM:-80}" \
        -gencode "arch=compute_${NVCC_SM:-80},code=compute_${NVCC_SM:-80}" \
        "$ROOT/csrc/moe4_smallm.cu" -o "$OBJ4" \
        || { echo "!! nvcc failed: moe4_smallm.cu" >&2; exit 4; }
      echo ">> moe4_smallm.o built -> $OBJ4"
    fi
  fi
  OUT="$OUT_DIR/${mod}${PY_EXT}"
  echo ">> Building variant [$name] -> $OUT"
  # shellcheck disable=SC2086
  "$CXX_BIN" $COMMON $flags \
      -DXIAOTU_MOE_MODULE_NAME="$mod" \
      -I"$PY_INC" -I"$PYBIND11_INC" -I"$ROOT/csrc" \
      "${CUDA_FLAGS[@]}" \
      ${OBJ4:+-DXIAOTU_MOE_WITH_MOE4=1} \
      "$SRC" ${OBJ4:+"$OBJ4"} \
      -o "$OUT"
  # 【默认构建门】没编内核时，产物里**不能**出现 moe4 的未定义符号
  #   （否则 `import xiaotu_moe` 会在 RTLD_NOW 下直接失败；R38 审计 P0-1）
  if [[ -z "${OBJ4:-}" ]]; then
    _nm_def="$(nm -D "$OUT" 2>/dev/null | grep -c "U moe4_smallm_" || true)"
    if [[ "${_nm_def:-0}" != "0" ]]; then
      echo "!! 默认构建产物含未定义 moe4 符号（BUILD_MOE4 未开却引用了内核）: $OUT" >&2
      exit 6
    fi
    unset _nm_def
  fi
  if [[ -n "${OBJ4:-}" ]]; then
    # 机械门：产物里真的带符号才算成功（教训：产物存在 ≠ 构建成功）
    # 用 nm（不是 nm -D）：符号只被同 .so 内的 binding 引用时不会进动态符号表。
    # ⚠️ 不能写成 `nm ... | grep -q`：`set -o pipefail` 下 grep -q 提前退出会让
    #    nm 收到 SIGPIPE(141) ⇒ 管道整体判失败 ⇒ **符号明明存在也报缺失**（本轮实测踩到）。
    _nm_out="$(nm "$OUT" 2>/dev/null || true)"
    case "$_nm_out" in
      *moe4_smallm_gu*) ;;
      *) echo "!! 产物缺少 moe4_smallm 符号: $OUT" >&2; exit 5 ;;
    esac
    unset _nm_out
  fi
done

echo ">> Done. Variants in $OUT_DIR:"
ls -1 "$OUT_DIR"/_xiaotu_moe_C*"${PY_EXT}"
