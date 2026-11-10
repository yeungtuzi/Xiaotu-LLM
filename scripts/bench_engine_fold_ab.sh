#!/usr/bin/env bash
# Engine A/B for the audit-recommended arms (CPU only; no service control).
#
# Arms:
#   fp32_old   XIAOTU_MOE_DPBF16=0 XIAOTU_MOE_FOLD_SCALE=0
#   fp32_fold  XIAOTU_MOE_DPBF16=0 XIAOTU_MOE_FOLD_SCALE=1
#   bf16_old   XIAOTU_MOE_DPBF16=1, VBMI=0, FOLD=0
#   bf16_vbmi  XIAOTU_MOE_DPBF16=1, VBMI=1, FOLD=0
#   bf16_fold  XIAOTU_MOE_DPBF16=1, VBMI=0, FOLD=1
#   bf16_fold_vbmi XIAOTU_MOE_DPBF16=1, VBMI=1, FOLD=1
#
# Run on the 192-core EPYC host, pinned to one core:
#   bash scripts/bench_engine_fold_ab.sh
#   SESSIONS=3 BS=1,2 DEDUP=6 REP=60 WARMUP=200 bash ...
#
# The script only runs the existing bench_engine_ab.py; it does not start,
# stop, or kill any service.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
MODEL="${MODEL:-/home/user/lvllm/models/ds41f-l8}"
LAYER="${LAYER:-3}"
BS="${BS:-1,2}"
DEDUP="${DEDUP:-6}"
REP="${REP:-60}"
WARMUP="${WARMUP:-200}"
SESSIONS="${SESSIONS:-3}"
CORE="${CORE:-0}"
OUT="${OUT:-$ROOT/dev-docs/report/tuning/logs/fold_ab_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT"
printf 'out=%s model=%s layer=%s BS=%s DEDUP=%s rep=%s warmup=%s sessions=%s\n' \
  "$OUT" "$MODEL" "$LAYER" "$BS" "$DEDUP" "$REP" "$WARMUP" "$SESSIONS" | tee "$OUT/manifest.txt"

run_arm() {
  local session="$1" arm="$2" dotp16="$3" fold_scale="$4" bf16_fold="$5" vbmi16="$6"
  local log="$OUT/s${session}_${arm}.log"
  echo "== s${session} ${arm} dotp16=${dotp16} fold_scale=${fold_scale} bf16_fold=${bf16_fold} vbmi16=${vbmi16} =="
  env \
    XIAOTU_LAYER1_NPZ="$MODEL" LAYER="$LAYER" \
    BS="$BS" DEDUP="$DEDUP" REP="$REP" WARMUP="$WARMUP" \
    XIAOTU_MOE_VARIANT=avx512_bf16_vbmi \
    XIAOTU_MOE_THREADS=1 THREADS=1 \
    XIAOTU_MOE_DPBF16="$dotp16" \
    XIAOTU_MOE_FOLD_SCALE="$fold_scale" \
    XIAOTU_MOE_DPBF16_FOLD="$bf16_fold" \
    XIAOTU_MOE_DPBF16_VBMI="$vbmi16" \
    taskset -c "$CORE" "$PYTHON" "$ROOT/scripts/bench_engine_ab.py" \
    > "$log" 2>&1
  tail -n 4 "$log"
}

for s in $(seq 1 "$SESSIONS"); do
  # Rotate order to reduce thermal/order bias.
  case $((s % 5)) in
    1) order=(fp32_old bf16_old bf16_vbmi bf16_fold bf16_fold_vbmi) ;;
    2) order=(bf16_fold_vbmi bf16_vbmi bf16_fold bf16_old fp32_old) ;;
    3) order=(bf16_vbmi bf16_fold bf16_old bf16_fold_vbmi fp32_old) ;;
    4) order=(bf16_old bf16_fold_vbmi bf16_vbmi fp32_old bf16_fold) ;;
    *) order=(bf16_fold bf16_old fp32_old bf16_fold_vbmi bf16_vbmi) ;;
  esac
  for arm in "${order[@]}"; do
    case "$arm" in
      fp32_old)  run_arm "$s" "$arm" 0 0 0 0 ;;
      bf16_old)  run_arm "$s" "$arm" 1 0 0 0 ;;
      bf16_vbmi) run_arm "$s" "$arm" 1 0 0 1 ;;
      bf16_fold) run_arm "$s" "$arm" 1 0 1 0 ;;
      bf16_fold_vbmi) run_arm "$s" "$arm" 1 0 1 1 ;;
    esac
  done
done

echo "done: $OUT"
