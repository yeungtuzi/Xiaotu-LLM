#!/usr/bin/env python3
"""前缀缓存一致性探针（小模型、无真权重）。

判据：**同一段 token 序列、同一次引擎生命周期内，反复发送必须给出逐 token 相同的输出。**
这是生产判据 (A)/(B) 的微缩版；与权重是否"正确"无关，只检验 `prefix cache + decoder-side
SWA bounded replay` 这条路径有没有把 KV 搞坏。

用法（远端，单卡，eager 以绕开小尺寸下的图捕获限制）：
    python scripts/probe_cache_consistency.py --model /home/user/lvllm/models/tiny41-probe \
        --len 4096 --rounds 8 [--no-pc]

退出码：0 = 全部一致；1 = 出现不一致（打印首次分歧位置与 token）。
"""

from __future__ import annotations

import argparse
import random
import sys

import torch


def build_prompt(length: int, vocab: int, seed: int = 1234) -> list[int]:
    rnd = random.Random(seed)
    # 避开特殊 token 区（前 100 个）
    return [rnd.randrange(100, vocab - 100) for _ in range(length)]



def replay_mode(llm, tok, args, SamplingParams) -> int:
    """生产结构的微缩版：一条共享长文档 + 每轮唯一后缀。

    round 0: 只发 doc（把 doc 的 KV 写进缓存）
    round i: doc + suffix_i（命中 doc ⇒ 触发 decoder-side SWA bounded replay）
    round i': 再发一次同样的 doc + suffix_i（全命中）
    判据：O_i 必须 == O_i'（同一 token 序列两次必须逐 token 相同）。
    """
    vocab = tok.vocab_size
    doc = build_prompt(args.len, vocab, seed=999)
    doc_text = tok.decode(doc)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, seed=0,
                        ignore_eos=True, detokenize=False)
    llm.generate([doc_text], sp, use_tqdm=False)  # 播种前缀
    print("seeded doc", flush=True)

    bad = []
    for i in range(1, args.rounds + 1):
        suf = build_prompt(64, vocab, seed=10000 + i)
        text = tok.decode(doc + suf)
        o1 = list(llm.generate([text], sp, use_tqdm=False)[0].outputs[0].token_ids)
        o2 = list(llm.generate([text], sp, use_tqdm=False)[0].outputs[0].token_ids)
        same = o1 == o2
        print(f"round {i}: hit_replay={o1[:6]} repeat={o2[:6]} same={same}", flush=True)
        if not same:
            first = next(k for k in range(min(len(o1), len(o2))) if o1[k] != o2[k])
            bad.append((i, first, o1[first], o2[first]))
    if bad:
        for i, first, a, b in bad[:5]:
            print(f"❌ round {i}: 首次分歧 {first}：replay 臂={a} 全命中臂={b}")
        return 1
    print("✅ replay 臂与全命中臂逐 token 一致")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/user/lvllm/models/tiny41-probe")
    ap.add_argument("--len", type=int, default=4096)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--no-pc", action="store_true")
    ap.add_argument("--replay-mode", action="store_true")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    vocab = tok.vocab_size
    ids = build_prompt(a.len, vocab)
    text = tok.decode(ids)

    llm = LLM(
        model=a.model,
        load_format="dummy",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.35,
        max_model_len=max(a.len + 512, 8192),
        enable_prefix_caching=not a.no_pc,
        max_num_seqs=1,
        dtype="bfloat16",
        enforce_eager=True,
        seed=0,
    )
    sp = SamplingParams(max_tokens=a.max_tokens, temperature=0.0, seed=0,
                        ignore_eos=True, detokenize=False)

    if a.replay_mode:
        return replay_mode(llm, tok, a, SamplingParams)

    outs: list[list[int]] = []
    for i in range(a.rounds):
        res = llm.generate([text], sp, use_tqdm=False)
        toks = list(res[0].outputs[0].token_ids)
        outs.append(toks)
        print(f"round {i + 1}: {toks[:8]}", flush=True)

    ref = outs[0]
    bad = [(i, o) for i, o in enumerate(outs[1:], start=2) if o != ref]
    print(f"\nPC={'off' if a.no_pc else 'on'} rounds={a.rounds} prompt_len={a.len}")
    if bad:
        for i, o in bad[:5]:
            first = next(k for k in range(min(len(o), len(ref))) if o[k] != ref[k])
            print(f"❌ round {i} 与 round 1 不一致：首个分歧位置 {first} "
                  f"ref={ref[first]} got={o[first]}")
        return 1
    print("✅ 全部一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
