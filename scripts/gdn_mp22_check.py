"""Standalone MP22 check for MATE GDN decode (no pytest needed).

Compares mate.gated_delta_rule_decode against a float32 torch reference taken
from tests/test_gdn_decode.py, on the target device, and prints which staging
path (TMA / plain copy) the kernel picked.
"""

import os
import sys

import torch
import torch.nn.functional as F

import mate
from mate.gdn_kernels.tilelang import gdn_decode as gdn_decode_mod


def reference_decode(q, k, v, state_kv, A_log, a, dt_bias, b, scale_factor):
    """Reference with state in [B, H, K, V] (KV) layout, mirrored from tests."""
    B = q.size(0)
    Hq, K = q.size(1), q.size(2)
    Hv, V = v.size(1), v.size(2)

    A_log = A_log.float()
    a = a.float()
    dt_bias = dt_bias.float()
    b = b.float()

    x = a + dt_bias
    softplus_x = torch.where(
        x <= 20.0, torch.log(1.0 + torch.exp(x)), x
    )
    g = -torch.exp(A_log) * softplus_x
    beta = 1.0 / (1.0 + torch.exp(-b))

    if Hq < Hv:
        q = q.repeat_interleave(Hv // Hq, dim=1)
        k = k.repeat_interleave(Hv // Hq, dim=1)
    elif Hq > Hv:
        q = q.reshape(B, Hv, Hq // Hv, K).mean(dim=2)
        k = k.reshape(B, Hv, Hq // Hv, K).mean(dim=2)

    q = F.normalize(q.float(), p=2.0, dim=-1)
    k = F.normalize(k.float(), p=2.0, dim=-1)
    q = q * scale_factor

    out = torch.zeros(B, Hv, V, device=q.device, dtype=torch.float32)
    new_state = torch.zeros(B, Hv, K, V, device=q.device, dtype=torch.float32)
    for bi in range(B):
        for hi in range(Hv):
            h = state_kv[bi, hi].float().clone()
            h = h * torch.exp(g[bi, hi])
            v_new = v[bi, hi].float() - torch.sum(h * k[bi, hi][:, None], dim=0)
            v_new = v_new * beta[bi, hi]
            h = h + k[bi, hi][:, None] * v_new[None, :]
            out[bi, hi] = torch.sum(h * q[bi, hi][:, None], dim=0)
            new_state[bi, hi] = h
    return out, new_state


def run_case(B, Hq=16, Hk=16, Hv=32, head=128, use_pool=False, seed=0):
    torch.manual_seed(seed)
    dev = "musa"
    dt = torch.bfloat16

    q = torch.randn(B, 1, Hq, head, dtype=dt, device=dev)
    k = torch.randn(B, 1, Hk, head, dtype=dt, device=dev)
    v = torch.randn(B, 1, Hv, head, dtype=dt, device=dev)
    k = F.normalize(k, p=2.0, dim=-1)

    state_kv = torch.randn(B, Hv, head, head, dtype=torch.float32, device=dev)
    state_vk = state_kv.transpose(-2, -1).contiguous()

    A_log = torch.randn(Hv, dtype=torch.float32, device=dev) * 0.1
    dt_bias = torch.randn(Hv, dtype=torch.float32, device=dev) * 0.1
    a = torch.randn(B, 1, Hv, dtype=dt, device=dev) * 0.1
    b = torch.randn(B, 1, Hv, dtype=dt, device=dev)

    kwargs = dict(
        q=q,
        k=k,
        v=v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        state_layout="VK",
        scale=1.0,
        use_qk_l2norm=True,
    )
    if use_pool:
        state_pool = state_vk.clone()
        kwargs["state"] = state_pool
        kwargs["state_indices"] = torch.arange(B, dtype=torch.int32, device=dev)
    else:
        kwargs["state"] = state_vk.clone()

    out, new_state = mate.gated_delta_rule_decode(**kwargs)
    torch.musa.synchronize()

    ref_out, ref_state_kv = reference_decode(
        q.squeeze(1),
        k.squeeze(1),
        v.squeeze(1),
        state_kv,
        A_log,
        a.squeeze(1),
        dt_bias,
        b.squeeze(1),
        scale_factor=1.0,
    )

    got_out = out.squeeze(1).float()
    got_state_kv = new_state.transpose(-2, -1).float()

    out_err = (got_out - ref_out).abs().max().item()
    state_err = (got_state_kv - ref_state_kv).abs().max().item()
    out_scale = ref_out.abs().max().item()
    state_scale = ref_state_kv.abs().max().item()

    print(
        f"[case B={B:3d} pool={int(use_pool)}] "
        f"out max|err|={out_err:.3e} (scale {out_scale:.2f}) "
        f"state max|err|={state_err:.3e} (scale {state_scale:.2f})"
    )
    return out_err, out_scale, state_err, state_scale


def main():
    print(f"torch={torch.__version__} mate={mate.__version__}")
    props = torch.musa.get_device_properties(0)
    print(
        f"device={props.name} arch=mp{props.major}{props.minor} "
        f"tma_available={gdn_decode_mod.target_supports_tma()} "
        f"MATE_GDN_DECODE_TMA={os.environ.get('MATE_GDN_DECODE_TMA')}"
    )
    fails = []
    tol = 3e-2  # bf16 activations; fp32 accumulation. Relative tolerance.
    for B in (1, 2, 4, 16):
        for pool in (False, True):
            try:
                out_err, out_scale, state_err, state_scale = run_case(B, use_pool=pool)
                rel_out = out_err / max(out_scale, 1e-6)
                rel_state = state_err / max(state_scale, 1e-6)
                if rel_out > tol or rel_state > tol:
                    raise AssertionError(
                        f"out rel {rel_out:.3e} / state rel {rel_state:.3e} > {tol:.0e}"
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[FAIL] B={B} pool={int(pool)}: {type(exc).__name__}: {exc}")
                fails.append((B, pool, exc))
    if fails:
        print(f"\nRESULT: FAIL ({len(fails)} case(s))")
        sys.exit(1)
    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()
