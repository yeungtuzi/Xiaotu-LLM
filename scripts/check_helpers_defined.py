#!/usr/bin/env python3
"""检查 scope 内每个文件"用了但没定义"的 gate helper。

来由：`_bound_check_skip` 之类因为"用了但没定义"已三次导致启动 NameError
（第 35、46、57 轮）。`ast.parse` 查不出这种错（语法合法），必须做名字级检查。
"""
import ast
import os
import sys

HELPERS = ("_bound_check_skip", "_bt_bound_skip", "_fcc_bound_skip")


def _fx(*c):
    for x in c:
        try:
            if x and os.path.exists(x):
                return x
        except OSError:
            continue
    return c[0]


BASE = _fx(os.environ.get("XTU_VLLM_ROOT", ""), "/root/lvllm/data/repos/vllm-mainline",
           "/home/user/lvllm/process_data/ref/repos/vllm-mainline")
PLUG = _fx(os.environ.get("XTU_PLUGIN_ROOT", ""), "/root/lvllm/repos/xiaotu-llm/vllm_xiaotu_moe",
           "/home/user/lvllm/vllm-xiaotu-moe/vllm_xiaotu_moe")
SCOPE = _fx(os.environ.get("XTU_SCOPE", ""),
            "/root/lvllm/repos/xiaotu-llm/scripts/kernel_audit_scope.txt",
            "/home/user/lvllm/kernel_audit_scope.txt")


def main() -> int:
    files = []
    for line in open(SCOPE, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = (os.path.join(PLUG, line[len("plugin:"):].split("/", 1)[-1])
             if line.startswith("plugin:") else os.path.join(BASE, line))
        if os.path.exists(p):
            files.append(p)
    bad = 0
    for p in files:
        try:
            t = ast.parse(open(p, encoding="utf-8", errors="replace").read())
        except SyntaxError:
            continue
        used, defined = set(), set()
        for n in ast.walk(t):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in HELPERS:
                used.add(n.id)
            if isinstance(n, ast.FunctionDef) and n.name in HELPERS:
                defined.add(n.name)
            if isinstance(n, ast.Assign):
                for tg in n.targets:
                    if isinstance(tg, ast.Name):
                        defined.add(tg.id)
        missing = used - defined
        if missing:
            print(f"❌ {os.path.basename(p)}: 使用了未定义的 {sorted(missing)} ⇒ 启动会 NameError")
            bad += 1
    print(f"\n扫描 {len(files)} 个文件；未定义 helper：{bad} 个")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
