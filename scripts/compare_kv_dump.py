#!/usr/bin/env python3
"""比对两次进程（健康 / 退化）的 KV 转储，定位第一个不同的层与行。

配合 `patches/diag/apply_kv_dump.py` 使用：转储文件形如
`<TAG>-<prefix>-<kind>-<NN>.pt`，内容见 `patches/diag/files/xtu_kv_dump.py`。

判读：
  * `rows` 相同、`block_table` 不同 ⇒ 只是分配不同，KV 内容一致；
  * `rows` 在某个行号开始不同 ⇒ **该行（= 该 token 位置）写进去的字节不同**，
    行号 × compress_ratio 即 token 位置；层名由文件名给出 ⇒ 第一个不同的层/位置即缺陷点。

用法：
    python3 scripts/compare_kv_dump.py /tmp/kv_dump/ --tags r1 r3
    python3 scripts/compare_kv_dump.py DIR --tags r1 r3 --kind compressed
"""
import argparse
import pathlib
import sys

import torch


def load(d: pathlib.Path, tag: str) -> dict:
    """按 (prefix, kind, seq_len) 建索引；seq_len 相同即同一 prompt 的同一 layer。"""
    out = {}
    for f in sorted(d.glob("%s-*.pt" % tag)):
        if f.name.startswith("_"):
            continue
        rec = torch.load(f, map_location="cpu", weights_only=False)
        key = (rec["prefix"], rec["kind"],
               int(rec.get("seq_raw") or rec.get("seq_len") or rec.get("n_rows") or 0))
        out[key] = rec
    return out


def _ranges(idx: torch.Tensor, max_ranges: int = 6) -> str:
    """把差异行号压成连续区间字符串，如 '0-12,3000-3231'。"""
    if idx.numel() == 0:
        return "-"
    out, start, prev = [], int(idx[0]), int(idx[0])
    for v in idx[1:].tolist():
        if v == prev + 1:
            prev = v
            continue
        out.append((start, prev))
        if len(out) >= max_ranges:
            out.append((prev, prev))
            return ",".join("%d-%d" % r if r[0] != r[1] else "%d" % r[0] for r in out) + ",..."
        start = prev = v
    out.append((start, prev))
    return ",".join("%d-%d" % r if r[0] != r[1] else "%d" % r[0] for r in out)


def slot_of_req(rec, rows: torch.Tensor):
    """把请求内的行号映射成压缩 slot：slot = block_table[r // bs] * bs + r % bs。"""
    bt = rec.get("block_table")
    bs = int(rec.get("block_size") or 0)
    if bt is None or bs <= 0:
        return None
    n = rows.numel()
    blk = bt[rows // bs]
    return blk * bs + (rows % bs)


def cmp_rows(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape:
        return {"shape_mismatch": (tuple(a.shape), tuple(b.shape))}
    differ = (a != b).any(dim=1)
    idx = torch.nonzero(differ).flatten()
    n = int(idx.numel())
    return {
        "n_rows": int(a.shape[0]),
        "n_diff_rows": n,
        "first_diff_row": int(idx[0]) if n else -1,
        "last_diff_row": int(idx[-1]) if n else -1,
        "diff_ranges": _ranges(idx),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--tags", nargs=2, required=True, metavar=("A", "B"))
    ap.add_argument("--kind", default="")
    ap.add_argument("--limit", type=int, default=400)
    a = ap.parse_args()
    d = pathlib.Path(a.dump_dir)
    A, B = load(d, a.tags[0]), load(d, a.tags[1])
    keys = sorted(set(k for k in A if (not a.kind or k[1] == a.kind)) & set(B))
    rows = []
    for prefix, kind, idx in keys:
        ra = A.get((prefix, kind, idx))
        rb = B.get((prefix, kind, idx))
        st = cmp_rows(ra["rows"], rb["rows"])
        same_bt = torch.equal(ra["block_table"], rb["block_table"])
        # 差异行是否落在"本请求本 chunk 写过的 slot"里
        st["in_slots"] = None
        sm = ra.get("slot_mapping")
        if st.get("n_diff_rows", 0) > 0 and sm is not None:
            idx = torch.arange(ra["rows"].shape[0])
            slots = slot_of_req(ra, idx)
            if slots is not None:
                written = set(int(x) for x in sm.reshape(-1).tolist() if int(x) >= 0)
                diff_idx = torch.nonzero((ra["rows"] != rb["rows"]).any(dim=1)).flatten()
                diff_slots = slots[diff_idx].tolist()
                n_in = sum(1 for x in diff_slots if int(x) in written)
                st["in_slots"] = (n_in, len(diff_slots))
        rows.append((st.get("n_diff_rows", -1), st.get("first_diff_row", -1), prefix, kind,
                     st, same_bt, ra.get("seq_len")))
    rows.sort(key=lambda r: (r[0] == 0, -r[0] if r[0] >= 0 else 0, r[2]))
    print("%-52s %-10s %8s %8s %6s %s" % ("prefix", "kind", "diff行", "首差行", "块表同", "seq_raw"))
    shown = 0
    for n, first, prefix, kind, st, same_bt, seq_len in rows:
        if shown >= a.limit:
            print("... 省略 %d 条" % (len(rows) - shown))
            break
        shown += 1
        if "shape_mismatch" in st:
            print("%-52s %-10s 形状不同 %s" % (prefix, kind, st["shape_mismatch"]))
            continue
        print("%-52s %-10s %8d %8d %6s %s  %s"
              % (prefix[-52:], kind, n, first, "Y" if same_bt else "N", seq_len,
                 st.get("diff_ranges", "")) +
              ("" if st.get("in_slots") is None else
               "  [写过的slot内 %d/%d]" % st["in_slots"]))
    tot = sum(r[0] for r in rows if r[0] > 0)
    print("\n层数=%d  有差异的层=%d  差异行合计=%d"
          % (len(rows), sum(1 for r in rows if r[0] > 0), tot))
    return 0 if tot == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
