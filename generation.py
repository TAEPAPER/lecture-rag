"""Answer a question from retrieved lecture chunks (the "G" in RAG).

Pipeline: question -> Retriever.search (top-k chunks) -> build prompt -> local LLM (Ollama) -> answer.
"""

import json
import sys

import requests

from chunking import fmt_time
from hybrid import HybridRetriever
from retrieval import Retriever

OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "exaone3.5:7.8b"  # LG's Korean-focused model, ~4.8GB (Q4)

# v1 (2026-11-06) -> v2 (2026-11-08). v1 problem found by eval_generation.py:
# the model admitted "not in the lecture" and then kept answering from its own knowledge ("일반적으로…").
SYSTEM_PROMPT = """너는 대학 강의 내용만 전달하는 조교다. 너 자신의 지식은 쓰지 않는다.

순서:
1. 먼저 [강의 발췌] 1번부터 마지막 번호까지 전부 읽고, 질문에 답하는 문장이 있는지 찾는다. 답은 마지막 발췌에 있을 수도 있다.
2. 찾았으면: 그 문장의 내용만으로 답하고, 문장 끝에 발췌 번호를 [1]처럼 붙인다.
3. 못 찾았으면: 정확히 "강의에서 다루지 않은 내용입니다." 한 문장만 쓰고 끝낸다. 그 뒤에 아무것도 덧붙이지 않는다.

금지:
- "일반적으로", "보통은", "~일 것입니다" 같은 추측이나 일반 지식을 덧붙이지 않는다.
- 발췌에 없는 용어, 도구 이름, 예시를 새로 꺼내지 않는다.
- 답이 없다고 말한 뒤에 설명을 이어 가지 않는다.

참고:
- 발췌는 음성 인식으로 받아 적은 것이라 오타가 있다 (예: '벡타'=벡터, '네이블'=레이블, '래루'=ReLU). 문맥으로 바로잡아 읽는다.
- 한국어로 1~4문장 이내로 짧게 답한다.

예시 (발췌에 답이 없는 경우):
질문: 드롭아웃 비율은 보통 얼마로 해?
답: 강의에서 다루지 않은 내용입니다."""


def build_prompt(question: str, results: list[tuple[float, dict]]) -> str:
    # Number each chunk so the model can cite it and we can map citations back to timestamps.
    blocks = [
        f"[{i}] ({c['source']} {fmt_time(c['start'])}~{fmt_time(c['end'])})\n{c['text']}"
        for i, (_, c) in enumerate(results, 1)
    ]
    return "[강의 발췌]\n\n" + "\n\n".join(blocks) + f"\n\n[질문]\n{question}"


def generate(prompt: str, verbose: bool = True, model: str = LLM_MODEL) -> str:
    """Call Ollama with streaming: print tokens as they arrive (if verbose), return the full answer."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "options": {
            "temperature": 0.1,  # low = stick to the given text, less "creative" wording
            "num_ctx": 8192,     # context window in tokens; Ollama's default is small
        },
    }
    answer = []
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():  # one JSON object per line (NDJSON)
            msg = json.loads(line)
            token = msg.get("message", {}).get("content", "")
            if verbose:
                print(token, end="", flush=True)
            answer.append(token)
            if msg.get("done"):
                break
    if verbose:
        print()
    return "".join(answer)


def ask(retriever, question: str, k: int = 5, show_prompt: bool = False) -> str:
    # k=5: hybrid search puts the correct chunk in the top 5 for 35/36 eval questions (top 3: 32/36)
    results = retriever.search(question, k=k)
    prompt = build_prompt(question, results)
    if show_prompt:
        print("=" * 20, "PROMPT", "=" * 20, f"\n{prompt}\n", "=" * 48)
    print(f"Q: {question}\nA: ", end="")
    answer = generate(prompt)
    print("sources:")
    for i, (score, c) in enumerate(results, 1):
        print(f"  [{i}] {fmt_time(c['start'])}~{fmt_time(c['end'])}")
    print()
    return answer


if __name__ == "__main__":
    retriever = HybridRetriever(Retriever())  # meaning + keyword search, see exp_hybrid.py
    if len(sys.argv) > 1:
        ask(retriever, " ".join(sys.argv[1:]), show_prompt=True)
    else:
        for q in [
            "하이퍼탄젠트를 쓰면 레이블을 어떻게 바꿔야 해?",
            "미니배치 크기는 어떻게 정해?",
            "역전파 알고리즘을 C로 직접 구현해 보라고 한 이유는?",
            "트랜스포머의 어텐션은 어떻게 동작해?",
        ]:
            ask(retriever, q)
