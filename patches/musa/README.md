# MUSA bring-up patches (candidate)

These patches target **vLLM v0.32.0** and are intentionally kept outside the
normal build until validated on S4000.

Apply order:

```bash
git apply patches/musa/0001-qwen4exp-musa-select-triton.patch
git apply patches/musa/0002-qwen4exp-musa-disable-pdl.patch
```

- `0001`: route MUSA to the Triton/AMD qwen4_exp path instead of `nvidia`.
- `0002`: force `launch_pdl=False` on MUSA; AMD kernels otherwise call
  `tl.extra.cuda.gdc_*`, which MUSA Triton does not provide.

Status: **candidate, not hardware-validated**. `0002` only covers `amd/ops/hc.py`;
`common/ops/ple.py` and `common/qsa_cache.py` still contain `tl.extra.cuda.gdc_*`
calls that need separate guards if PLE/QSA are exercised on MUSA.
