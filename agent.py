"""Phase 5: a lecture study agent, written from scratch.

RAG (generation.py) is a FIXED pipeline:  question -> search once -> LLM answers.
An agent lets the LLM DECIDE what to do:   message -> LLM -> (maybe call tools) -> LLM -> ... -> reply.

The core is one loop (Agent.reply):
    1. send the whole conversation + the list of available tools to the LLM
    2. if the LLM asks to call tools -> run them, append the results to the conversation, go to 1
    3. if the LLM writes a normal message -> show it to the user

Step 2 adds what a quiz needs and RAG lacks:
    - memory:  the conversation (self.messages) is kept across turns, so the agent remembers
               which quiz question it asked and which lecture section it came from
    - choice:  answer a question / make a quiz / grade the student's answer, picked by the LLM
    - tools:   search_lecture (find content), pick_quiz_section (choose quiz material)

Run:  uv run python agent.py          (chat in the terminal; type q to quit)
      uv run python agent.py --demo   (scripted conversation)
"""

import json
import random
import sys

import requests

from chunking import fmt_time
from hybrid import HybridRetriever
from retrieval import Retriever

OLLAMA_URL = "http://localhost:11434/api/chat"
AGENT_MODEL = "qwen3:8b"  # supports tool calling in Ollama; EXAONE 3.5 does not
MAX_STEPS = 6             # safety limit per user message: stop even if the LLM keeps calling tools

SYSTEM_PROMPT = """너는 대학 강의(오일석 「기계학습」 3장 3.4~3.7절)로 공부하는 학생을 돕는 조교다.
학생의 메시지를 보고 아래 세 가지 중 맞는 일을 한다.

[질문에 답하기]
- 반드시 search_lecture 도구로 강의 내용을 찾아본 뒤 답한다. 주제가 여러 개면 주제마다 따로 검색한다.
- 검색 결과에 있는 내용만으로 답하고, 근거가 된 강의 시간(예: 21:00~22:24)을 적는다.
- 찾을 수 없으면 "강의에서 다루지 않은 내용입니다."라고 답한다.

[퀴즈 내기] (학생이 퀴즈, 문제, 테스트를 요청할 때)
- pick_quiz_section 도구로 강의 구간을 하나 받아, 그 구간의 내용만으로 짧은 서술형 문제를 딱 1개 낸다.
- 정답이나 힌트는 말하지 않는다. 문제와 출제 구간 시간만 보여 주고 학생의 답을 기다린다.

[채점하기] (직전에 낸 퀴즈에 학생이 답했을 때)
- 앞에서 받은 구간 내용과 학생의 답을 비교해 "정답", "부분 정답", "오답" 중 하나로 판정한다.
- 판정 이유와 강의에서 말한 내용을 짧게 설명하고, 해당 강의 시간을 적는다.

한국어로 짧게 답한다."""


# ---------------------------------------------------------------- tools

retriever = None  # loaded lazily so importing this file stays fast
quizzed = set()   # chunk ids already used for quizzes, to avoid repeats


def get_retriever() -> HybridRetriever:
    global retriever
    if retriever is None:
        retriever = HybridRetriever(Retriever())
    return retriever


def search_lecture(query: str) -> str:
    """Tool: search the lecture transcript, return the top 3 chunks with timestamps."""
    results = get_retriever().search(query, k=3)
    return "\n\n".join(f"({fmt_time(c['start'])}~{fmt_time(c['end'])}) {c['text']}" for _, c in results)


def pick_quiz_section(topic: str = "") -> str:
    """Tool: pick one lecture section (not used before) as quiz material, optionally about a topic."""
    if topic:  # the student asked for a quiz on something specific
        chunks = [c for _, c in get_retriever().search(topic, k=3)]
    else:
        chunks = get_retriever().chunks
    unused = [c for c in chunks if c["id"] not in quizzed] or chunks
    c = random.choice(unused)
    quizzed.add(c["id"])
    return f"({fmt_time(c['start'])}~{fmt_time(c['end'])}) {c['text']}"


TOOLS = {"search_lecture": search_lecture, "pick_quiz_section": pick_quiz_section}

# What the LLM sees: a name, a description, and the arguments (JSON Schema).
# The LLM never runs code itself; it only writes "please call search_lecture with query=...".
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_lecture",
            "description": "강의 전사본에서 query와 관련된 구간 3개를 찾아 시간과 함께 돌려준다. 질문에 답할 때 쓴다.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "검색할 내용. 짧은 키워드나 문장"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pick_quiz_section",
            "description": "퀴즈 출제용으로 강의 구간 하나를 골라 시간과 함께 돌려준다. 퀴즈를 낼 때만 쓴다.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string", "description": "특정 주제로 출제할 때만. 없으면 비워 둔다"}},
            },
        },
    },
]


# ---------------------------------------------------------------- agent

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


class Agent:
    def __init__(self, verbose: bool = True):
        # The conversation IS the agent's memory: every turn, tool call and tool result is kept here
        # and sent to the LLM again on the next turn.
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.verbose = verbose

    def reply(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        for step in range(1, MAX_STEPS + 1):
            msg = call_llm(self.messages)
            self.messages.append(msg)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:  # a normal message: this turn is done
                return msg["content"]

            for call in tool_calls:
                name = call["function"]["name"]
                args = call["function"].get("arguments") or {}  # already a dict in Ollama's response
                if self.verbose:
                    print(f"  [step {step}] 도구 호출: {name}({json.dumps(args, ensure_ascii=False)})")
                # The LLM may call a tool that doesn't exist or pass wrong arguments.
                # Don't crash: report the error back so the LLM can correct itself on the next step.
                try:
                    result = TOOLS[name](**args)
                except Exception as e:
                    result = f"error: {type(e).__name__}: {e}"
                if self.verbose:
                    print(f"           결과: {result[:70].replace(chr(10), ' ')}...")
                self.messages.append({"role": "tool", "tool_name": name, "content": result})

        return "(멈춤: 도구 호출이 너무 많습니다)"


def chat():
    agent = Agent()
    print("강의 조교 agent입니다. 질문하거나 '퀴즈 내줘'라고 해 보세요. (종료: q)")
    while True:
        user_text = input("\n나: ").strip()
        if user_text in ("q", "quit", "exit"):
            break
        if user_text:
            print(f"조교: {agent.reply(user_text)}")


def demo():
    agent = Agent()
    for user_text in [
        "3장 내용으로 퀴즈 하나 내줘",
        "잘 모르겠는데, 학습률이랑 관련 있는 거 아니야?",
        "미니배치 크기는 보통 얼마로 해?",
    ]:
        print(f"\n나: {user_text}")
        print(f"조교: {agent.reply(user_text)}")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else chat()
