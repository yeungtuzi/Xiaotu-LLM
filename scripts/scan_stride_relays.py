"""一次性扫描：局部变量承载 `.stride(0)`/`.shape[0]` → 喂给 constexpr 形参。"""
import ast, os, re, sys

SCOPE_FILE = os.environ.get('XTU_SCOPE', '/home/user/lvllm/kernel_audit_scope.txt')
scope = [l.strip() for l in open(SCOPE_FILE) if l.strip() and not l.startswith('#')]
def resolve(rel):
    rel = rel.replace('plugin:', '')
    for base in ('/home/user/lvllm/process_data/ref/repos/vllm-mainline', '/home/user/lvllm/vllm-xiaotu-moe/vllm_xiaotu_moe'):
        p = os.path.join(base, rel)
        if os.path.exists(p): return p
        p2 = os.path.join(base, os.path.basename(rel))
        if os.path.exists(p2): return p2
    return None

hits = []
for rel in scope:
    p = resolve(rel)
    if not p: continue
    src = open(p, encoding='utf-8', errors='replace').read()
    try: tree = ast.parse(src)
    except SyntaxError: continue
    # 该文件所有 constexpr 形参名
    cx = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            for a in n.args.args + n.args.kwonlyargs:
                if a.annotation and 'constexpr' in ast.unparse(a.annotation):
                    cx.add(a.arg)
    if not cx: continue
    # 局部变量 = .stride(0)/.shape[0]
    relays = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            v = ast.unparse(n.value)
            if re.search(r"\.stride\(\s*0\s*\)", v) or re.search(r"\.shape\[\s*0\s*\]", v) or re.search(r"\.stride\(\s*0\s*\)", v):
                relays[n.targets[0].id] = (n.lineno, v)
    if not relays: continue
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call): continue
        for kw in n.keywords:
            if kw.arg in cx and isinstance(kw.value, ast.Name) and kw.value.id in relays:
                ln, v = relays[kw.value.id]
                hits.append((p, kw.lineno, kw.arg, kw.value.id, ln, v))

print(f"局部变量承载 → constexpr：{len(hits)} 处")
for p, ln, param, var, rln, v in hits:
    print(f"  {p}:{ln}  {param}={var}  ← :{rln} {v[:50]}")

if __name__ == "__main__":
    import sys as _s
    _s.exit(1 if hits else 0)
