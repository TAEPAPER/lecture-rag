"""Phase 5, step 1: a minimal agent, written from scratch.

RAG (generation.py) is a FIXED pipeline:  question -> search once -> LLM answers.
An agent lets the LLM DECIDE what to do:   question -> LLM -> (maybe call a tool) -> LLM -> ... -> answer.

The whole agent is this loop:
    1. send the conversation + the list of available tools to the LLM
    2. if the LLM asks to call a tool -> run it, append the result to the conversation, go to 1
    3. if the LLM writes a normal answer -> done

Run:  uv run python agent.py                     (demo questions)
      uv run python agent.py "질문"
"""

import json
import sys

import requests

from chunking import fmt_time
from hybrid import HybridRetriever
from retrieval import Retriever

OLLAMA_URL = "http://localhost:11434/api/chat"
AGENT_MODEL = "qwen3:8b"  # supports tool calling in Ollama; EXAONE 3.5 does not
MAX_STEPS = 6             # safety limit: stop even if the LLM keeps calling tools

SYSTEM_PROMPT = """너는 대학 강의(오일석 「기계학습」 3장 3.4~3.7절)로 공부하는 학생을 돕는 조교다.

- 질문에 답하기 전에 반드시 search_lecture 도구로 강의 내용을 찾아본다.
- 질문에 주제가 여러 개 있으면 주제마다 따로 검색한다.
- 검색 결과가 질문과 맞지 않으면 다른 표현으로 다시 검색해도 된다.
- 검색 결과에 있는 내용만으로 답하고, 근거가 된 강의 시간(예: 21:00~22:24)을 함께 적는다.
- 검색해도 찾을 수 없으면 "강의에서 다루지 않은 내용입니다."라고 답한다.
- 한국어로 짧게 답한다."""


# ---------------------------------------------------------------- tools

retriever = None  # loaded lazily so importing this file stays fast


def search_lecture(query: str) -> str:
    """Tool: search the lecture transcript, return the top 3 chunks with timestamps."""
    global retriever
    if retriever is None:
        retriever = HybridRetriever(Retriever())
    results = retriever.search(query, k=3)
    return "\n\n".join(f"({fmt_time(c['start'])}~{fmt_time(c['end'])}) {c['text']}" for _, c in results)


TOOLS = {"search_lecture": search_lecture}

# What the LLM sees: a name, a description, and the arguments (JSON Schema).
# The LLM never runs code itself; it only writes "please call search_lecture with query=...".
TOOL_SPECS = [{
    "type": "function",
    "function": {
        "name": "search_lecture",
        "description": "강의 전사본에서 query와 관련된 구간 3개를 찾아 시간과 함께 돌려준다.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "검색할 내용. 짧은 키워드나 문장"}},
            "required": ["query"],
        },
    },
}]


# ---------------------------------------------------------------- agent loop

def call_llm(messages: list[dict]) -> dict:
    payload = {
        "model": AGENT_MODEL,
        "messages": messages,
        "tools": TOOL_SPECS,
        "stream": False,
        "think": False,  # Qwen 3 can "think out loud" first; off to keep it fast and the trace readable
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]


def run_agent(question: str, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for step in range(1, MAX_STEPS + 1):
        msg = call_llm(messages)
        messages.append(msg)  # the LLM's turn becomes part of the conversation

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:  # a normal answer: the agent is done
            if verbose:
                print(f"  [step {step}] 답변 작성")
            return msg["content"]

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]  # already a dict in Ollama's response
            if verbose:
                print(f"  [step {step}] 도구 호출: {name}({json.dumps(args, ensure_ascii=False)})")
            if name in TOOLS:
                result = TOOLS[name](**args)
            else:
                result = f"error: unknown tool {name}"
            if verbose:
                print(f"           결과: {result[:80].replace(chr(10), ' ')}...")
            messages.append({"role": "tool", "tool_name": name, "content": result})

    return "(멈춤: 도구 호출이 너무 많습니다)"


if __name__ == "__main__":
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else [
        "미니배치 크기는 보통 얼마로 해?",
        "미니배치 방식이 뭔지, 그리고 오류 역전파가 전방 계산보다 얼마나 느린지 각각 알려줘",
        "드롭아웃 비율은 보통 얼마로 해?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        answer = run_agent(q)
        print(f"A: {answer}")
