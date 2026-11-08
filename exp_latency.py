"""Phase 4, step 1: where does the time go for one question?

Measures each stage separately:
  embed      question -> vector (bge-m3)
  search     dense matmul + BM25 + RRF fusion
  llm load   loading the LLM into memory (only when it is not already loaded)
  prefill    LLM reads the prompt (rules + 5 chunks + question)
  first tok  time until the first answer token arrives (what the user feels as "waiting")
  decode     LLM writes the answer, one token at a time
Ollama reports its own timings in the last streamed message (durations in nanoseconds).
"""

import json
import sys
import time

import numpy as np
import requests

from embedding import embed
from generation import LLM_MODEL, OLLAMA_URL, SYSTEM_PROMPT, build_prompt
from hybrid import HybridRetriever, ranks_of
from retrieval import Retriever

QUESTIONS = [
    "하이퍼탄젠트를 쓰면 레이블을 어떻게 바꿔야 해?",
    "미니배치 크기는 보통 얼마로 해?",
    "오류 역전파는 전방 계산보다 시간이 얼마나 더 걸려?",
    "학습률을 정하는 방법에는 어떤 것들이 있어?",
    "드롭아웃 비율은 보통 얼마로 해?",
]
K = 5
args = sys.argv[1:]
opt = lambda flag, default: args[args.index(flag) + 1] if flag in args else default
K = int(opt("--k", K))
MODEL = opt("--model", LLM_MODEL)


def timed_llm(prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "stream": True,
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }
    t0 = time.perf_counter()
    first_token, stats = None, None
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            msg = json.loads(line)
            if first_token is None and msg.get("message", {}).get("content"):
                first_token = time.perf_counter() - t0
            if msg.get("done"):
                stats = msg
                break
    ns = 1e9
    return {
        "load": stats.get("load_duration", 0) / ns,
        "prefill": stats["prompt_eval_duration"] / ns,
        "prompt_tokens": stats["prompt_eval_count"],
        "first_token": first_token,
        "decode": stats["eval_duration"] / ns,
        "answer_tokens": stats["eval_count"],
        "total": time.perf_counter() - t0,
    }


def unload_all_models():
    """Start every run from a cold LLM. Otherwise Ollama reuses cached work from earlier prompts
    that share the same beginning (prompt caching), and prefill looks unrealistically fast."""
    for m in requests.get(OLLAMA_URL.replace("/api/chat", "/api/ps")).json().get("models", []):
        requests.post(OLLAMA_URL.replace("/api/chat", "/api/generate"), json={"model": m["name"], "keep_alive": 0})


def main():
    unload_all_models()
    retriever = HybridRetriever(Retriever())
    rows = []
    for q in QUESTIONS:
        t0 = time.perf_counter()
        qv = embed(retriever.dense.model, [q])[0]
        t_embed = time.perf_counter() - t0

        t0 = time.perf_counter()
        dense_scores = retriever.dense.vectors @ qv
        bm25_scores = retriever.bm25.scores(q)
        fused = 1 / (60 + ranks_of(dense_scores)) + 1 / (60 + ranks_of(bm25_scores))
        top = np.argsort(-fused)[:K]
        results = [(float(fused[i]), retriever.chunks[i]) for i in top]
        t_search = time.perf_counter() - t0

        llm = timed_llm(build_prompt(q, results))
        rows.append({"embed": t_embed, "search": t_search, **llm})
        print(f"{q[:28]:<30} embed {t_embed*1000:5.0f}ms  search {t_search*1000:4.1f}ms  "
              f"load {llm['load']:4.1f}s  prefill {llm['prefill']:5.1f}s ({llm['prompt_tokens']} tok)  "
              f"first-token {llm['first_token']:5.1f}s  decode {llm['decode']:5.1f}s ({llm['answer_tokens']} tok, "
              f"{llm['answer_tokens']/llm['decode']:.1f} tok/s)  total {llm['total']:5.1f}s")

    # Skip the first row in the median: it may include one-time model loading.
    warm = rows[1:]
    med = lambda key: float(np.median([r[key] for r in warm]))
    print(f"\nmedian (warm, excluding the first question)  model={MODEL} k={K}:")
    for key, unit in [("embed", "s"), ("search", "s"), ("prefill", "s"), ("first_token", "s"), ("decode", "s"), ("total", "s")]:
        print(f"  {key:<12} {med(key):7.3f}{unit}")
    print(f"  prompt tokens {med('prompt_tokens'):.0f}, answer tokens {med('answer_tokens'):.0f}, "
          f"prefill speed {med('prompt_tokens')/med('prefill'):.0f} tok/s")


if __name__ == "__main__":
    main()
