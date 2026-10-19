#!/usr/bin/env python3
"""检查一个服务日志是否**真的**通过了 CUDA Graph 捕获(R65 的机械验收)。

用法:
    python scripts/check_capture_log.py <log> [--expect-capture]

判据(R65):
  ① 若 `--expect-capture`(图臂必带):日志必须出现 `Capturing CUDA graphs`;
  ② `device-side assert` / `StreamCaptureInvalidated` / `CUDA error` 必须为 0;
  ③ `Capturing CUDA graphs` 行**紧邻**不得出现本插件的打印/进度
     (`[xtu-pin` / `[xtu-pf-` / `[vllm-xtu-moe]`)—— 出现即说明捕获窗口里有 host 工作(B1049)。

退出码:0 通过;1 不通过;2 用法错误。
"""
import sys

ERR_PAT = ("device-side assert", "StreamCaptureInvalidated", "CUDA error")
MARK_PAT = ("[xtu-pin", "[xtu-pf-", "[vllm-xtu-moe]")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    expect = "--expect-capture" in argv[2:]
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError as exc:
        print(f"⛔ 读不到日志: {exc}")
        return 2

    errs = [(i + 1, l) for i, l in enumerate(lines) if any(p in l for p in ERR_PAT)]
    caps = [i for i, l in enumerate(lines) if "Capturing CUDA graphs" in l]
    bad_marks = []
    # ⭐ 只看**同一物理行**:进度条用 \r 刷新,守卫失效时打印会直接拼在进度条后面
    #   (`Capturing ... 14%[xtu-pin][hit] ...` = B1049 的原始证据)。
    #   ⛔ 不要向后多看几行 —— 捕获**完成后**的 `worker startup finished` 是正常日志,
    #      会被误判(2026-10-19 实测)。
    for i in caps:
        tail = lines[i].split("Capturing", 1)[-1]
        for p in MARK_PAT:
            if p in tail:
                bad_marks.append((i + 1, lines[i][:160], p))

    ok = True
    print(f"日志: {path}  行数: {len(lines)}")
    if expect and not caps:
        print("⛔ 期望图捕获,但日志里没有 `Capturing CUDA graphs`")
        ok = False
    if caps:
        print(f"✓ 捕获行: {len(caps)} 条(首次 行{caps[0]+1})")
    else:
        print("· 无捕获行(非图臂可为正常)")
    if errs:
        ok = False
        print(f"⛔ 致命签名 {len(errs)} 条:")
        for ln, s in errs[:5]:
            print(f"   行{ln}: {s[:150]}")
    else:
        print("✓ 无 device-side assert / StreamCaptureInvalidated / CUDA error")
    complete = (any(("Capturing CUDA graphs" in l and "100%" in l) for l in lines)
                or any("graphs captured" in l for l in lines)
                or any("Application startup complete" in l for l in lines))
    if bad_marks and not complete:
        ok = False
        print(f"⛔ 捕获**未完成**且窗口内有插件打印 {len(bad_marks)} 处:")
        for ln, s, p in bad_marks[:5]:
            print(f"   行{ln} [{p}]: {s}")
    elif bad_marks:
        # ⚠️ 只算 **WARN**:2026-10-19 实测(B1049)捕获**最终成功**(READY),
        #    但打印仍拼在进度条上 ⇒ 说明 `_in_capture_window()` 在 vLLM 捕获期
        #    仍可能为 False;它是否致命取决于时序(这次没坏,下次可能坏)。
        #    ⇒ 不判 FAIL,但必须写进日志并在复盘里回炉守卫。
        print(f"⚠️ 捕获窗口内出现插件 host 打印 {len(bad_marks)} 处(守卫可能失效,需回炉):")
        for ln, s, p in bad_marks[:5]:
            print(f"   行{ln} [{p}]: {s}")
    else:
        print("✓ 捕获行附近无插件打印")

    print("== PASS ==" if ok else "== FAIL ==")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
