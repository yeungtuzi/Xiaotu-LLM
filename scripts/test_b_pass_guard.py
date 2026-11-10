#!/usr/bin/env python3
"""test_b_pass_guard.py —— 证明 B 臂不会在"根本没走缓存"时假通过。

背景（台账 #223）：B 的判据是"unique 命中率 100%"。但如果服务压根没命中前缀缓存，
四个臂各自冷跑也能 100% ⇒ 结论与"交错 shared/unique 是否串味"毫无关系。
所以 test_b 现在额外要求 usage.prompt_tokens_details.cached_tokens > 0 至少发生一次。

本测试用 monkeypatch 替换 ask()，跑三种情形：
  1) 全命中 + 有缓存命中   ⇒ 必须通过
  2) 全命中 + 缓存 0 命中  ⇒ 必须**不**通过（历史漏洞）
  3) 有一条没命中 MARK      ⇒ 必须不通过
"""
import importlib.util
import io
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("accept_abc", os.path.join(HERE, "accept_abc.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def run(fake_ask):
    """替换 ask 后调用 test_b，返回 (是否通过, 输出文本)。"""
    mod.ask = fake_ask
    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = mod.test_b(port=0, rounds=1, n_words=200, max_tokens=32)
    return ok, buf.getvalue()


def make_fake(cached_tokens, drop_one=False):
    state = {"n": 0}

    def fake(port, prompt, max_tokens):
        state["n"] += 1
        # 从 prompt 里抽出 MARK-... 作为"模型正确回答"
        mark = ""
        for tok in prompt.split():
            if tok.startswith("MARK-"):
                mark = tok
                break
        text = mark
        if drop_one and state["n"] == 2:
            text = "SOMETHING-ELSE"
        usage = {"prompt_tokens": 1234,
                 "prompt_tokens_details": {"cached_tokens": cached_tokens}}
        return text, usage

    return fake


cases = []

ok, out = run(make_fake(cached_tokens=1024))
cases.append(("全命中+有缓存命中 ⇒ 通过", ok is True, out))

ok, out = run(make_fake(cached_tokens=0))
cases.append(("全命中但缓存 0 命中 ⇒ 必须拒绝（防假通过）", ok is False and "没有发生缓存命中" in out, out))

ok, out = run(make_fake(cached_tokens=1024, drop_one=True))
cases.append(("有一条没命中 MARK ⇒ 不通过", ok is False and "未达 100%" in out, out))

fails = 0
for name, good, out in cases:
    print(("✅ " if good else "❌ ") + name)
    if not good:
        fails += 1
        print("--- 实际输出 ---")
        print(out)
print()
if fails:
    print(f"❌ {fails} 个用例不符合预期")
    sys.exit(1)
print("✅ 3/3 用例符合预期")
