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
import os
import random
import re
import sys
import time
from pathlib import Path

import requests

from chunking import fmt_time
from hybrid import HybridRetriever
from retrieval import Retriever


def load_env(path: str = ".env"):
    """Read KEY=VALUE lines from .env into os.environ (the API key lives there, never in code or git)."""
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


load_env()

# Which LLM runs the agent. Only call_llm() knows the difference; the agent loop is the same.
#   ollama:     local, free, slow on a laptop (~55-90 s per question)
#   openrouter: cloud gateway to many models with one key (OPENROUTER_API_KEY in .env), OpenAI-compatible
#   groq:       cloud, very fast, fewer models, free tier (GROQ_API_KEY in .env), OpenAI-compatible
#   gemini:     Google cloud, free tier (GEMINI_API_KEY in .env), OpenAI-compatible endpoint
# Pick with env vars, e.g.  AGENT_BACKEND=openrouter AGENT_MODEL=qwen/qwen3-32b uv run python agent.py
BACKEND = os.environ.get("AGENT_BACKEND", "ollama")
OLLAMA_URL = "http://localhost:11434/api/chat"
CLOUD = {  # backend -> (chat completions URL, env var holding the API key)
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY"),
}
DEFAULT_MODEL = {
    "ollama": "qwen3:8b",           # supports tool calling in Ollama; EXAONE 3.5 does not
    "openrouter": "qwen/qwen3-32b",  # same family as the local model, 4x larger (no 8B with tools there)
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-3.6-flash",    # Google AI Studio free tier (limits shown per project in AI Studio)
}
AGENT_MODEL = os.environ.get("AGENT_MODEL") or DEFAULT_MODEL[BACKEND]
MAX_STEPS = 6             # safety limit per user message: stop even if the LLM keeps calling tools
SEARCH_K = 5              # chunks per search (was 3; plain RAG uses 5 and found more gold sections)
REFUSAL = "다루지 않은"
# Sent when the LLM refuses without having searched (eval_agent.py: 7/41 questions did this).
NUDGE = "(시스템 알림) 아직 search_lecture로 강의를 찾아보지 않았습니다. 먼저 검색해 보고, 검색 결과를 본 뒤에 답하세요."
# Sent when the answer cites a lecture time that no tool result contained (a made-up source).
# eval_agent.py: gpt-oss-120b answered off-lecture questions from its own knowledge and added fake times like 27:15~27:42.
FAKE_CITE = ("(시스템 알림) 답변에 적은 강의 시간 {times}은(는) 검색 결과에 없습니다. "
             "검색 결과에 있는 내용과 시간만으로 다시 답하세요. 검색 결과에 답이 없으면 \"강의에서 다루지 않은 내용입니다.\"라고만 답하세요.")
TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")
RANGE_RE = re.compile(r"\((\d{1,2}:\d{2})~(\d{1,2}:\d{2})\)")  # how tool results label each section
CITE_SLACK = 5  # seconds: allow a cited time a little outside a section (rounding)


def to_seconds(mm_ss: str) -> int:
    m, s = mm_ss.split(":")
    return int(m) * 60 + int(s)


def fake_citations(answer: str, messages: list[dict]) -> list[str]:
    """Times written in the answer that fall inside NO section the tools actually returned.

    This is a check in code, not a request to the LLM: we don't trust the model to be honest about its sources,
    we compare its claims against what it was really given (like validating a client's input on the server).
    """
    sections = [(to_seconds(a), to_seconds(b))
                for m in messages if m["role"] == "tool"
                for a, b in RANGE_RE.findall(m["content"])]
    fakes = []
    for m, s in TIME_RE.findall(answer):
        t = int(m) * 60 + int(s)
        if not any(a - CITE_SLACK <= t <= b + CITE_SLACK for a, b in sections):
            fakes.append(f"{m}:{s}")
    return fakes

# v1 -> v2 (2026-11-10). Same model (gpt-oss-120b): RAG grounded 40/41, agent v1 only 27/41.
# v1 had one weak line ("검색 결과에 있는 내용만으로"); v2 copies the RAG prompt's rules that worked:
# "no own knowledge", an explicit ban list, a refusal example, and the transcription-typo note.
SYSTEM_PROMPT = """너는 대학 강의(오일석 「기계학습」 3장 3.4~3.7절)로 공부하는 학생을 돕는 조교다. 너 자신의 지식은 쓰지 않는다.
학생의 메시지를 보고 아래 세 가지 중 맞는 일을 한다.

[질문에 답하기]
- 반드시 search_lecture 도구로 강의 내용을 찾아본 뒤 답한다. 주제가 여러 개면 주제마다 따로 검색한다.
- 검색 결과를 전부 읽고, 질문에 답하는 문장을 찾는다.
- 찾았으면: 그 문장의 내용만으로 답하고, 근거가 된 강의 시간(예: 21:00~22:24)을 적는다.
- 못 찾았으면: 정확히 "강의에서 다루지 않은 내용입니다." 한 문장만 쓰고 끝낸다.
- 금지: "일반적으로", "보통은", "~일 것입니다" 같은 추측이나 일반 지식 덧붙이기 /
  검색 결과에 없는 용어·사람 이름·알고리즘 이름·예시를 새로 꺼내기 / 답이 없다고 한 뒤 설명을 이어 가기.
- 검색 결과는 음성 인식으로 받아 적은 것이라 오타가 있다 (예: '벡타'=벡터, '네이블'=레이블, '래루'=ReLU). 문맥으로 바로잡아 읽는다.
- 예시 (검색 결과에 답이 없는 경우) 질문: 드롭아웃 비율은 보통 얼마로 해? → 답: 강의에서 다루지 않은 내용입니다.

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
    results = get_retriever().search(query, k=SEARCH_K)
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
            "description": "강의 전사본에서 query와 관련된 구간 5개를 찾아 시간과 함께 돌려준다. 질문에 답할 때 쓴다.",
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

def call_llm(messages: list[dict], think: bool = False) -> dict:
    return call_cloud(messages, think) if BACKEND in CLOUD else call_ollama(messages, think)


def call_ollama(messages: list[dict], think: bool = False) -> dict:
    payload = {
        "model": AGENT_MODEL,
        "messages": messages,
        "tools": TOOL_SPECS,
        "stream": False,
        "think": think,  # Qwen 3 can reason privately before acting: more careful, but slower
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]


RETRYABLE = {429, 500, 502, 503, 504}  # 429 = rate limit, 5xx = server hiccup (both pass if we wait)
MAX_WAIT = 120  # seconds: never trust a long retry-after blindly (see below)


def post_with_retry(url: str, payload: dict, headers: dict, tries: int = 30) -> requests.Response:
    """POST to a cloud API, waiting and retrying on rate limits AND temporary server errors.

    Free tiers allow only a few requests per minute, and Groq sometimes answers 503 for a moment.
    One such hiccup used to crash a 40-minute grading run, so wait (retry-after, else 15 s) and try again.
    The wait is capped at MAX_WAIT: Groq once said "retry in 20 min" for its daily limit, the quota came back
    earlier, and the run still slept ~3 hours. Short waits x more tries = notice early, still survive ~1 hour.
    """
    r = None
    for _ in range(tries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=120)
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(15)
            continue
        if r.status_code not in RETRYABLE:
            break
        time.sleep(min(float(r.headers.get("retry-after", 15)), MAX_WAIT))
    if r is None:
        raise ConnectionError(f"no response from {url} after {tries} tries")
    r.raise_for_status()
    return r


def call_cloud(messages: list[dict], think: bool = False) -> dict:
    """Same job as call_ollama, but over the OpenAI-style API that most cloud LLMs share.

    We keep the conversation in Ollama's shape and translate at the edge (like a DTO mapper):
      - tool call arguments: Ollama uses a dict, OpenAI a JSON string
      - tool results: OpenAI links each result to its call by tool_call_id
    """
    def to_openai(m: dict) -> dict:
        if m["role"] == "assistant" and m.get("tool_calls"):
            return {"role": "assistant", "content": m.get("content") or "", "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["function"]["name"], "arguments": json.dumps(c["function"]["arguments"], ensure_ascii=False)},
                 **({"extra_content": c["extra_content"]} if "extra_content" in c else {})}
                for c in m["tool_calls"]]}
        if m["role"] == "tool":
            return {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]}
        return {"role": m["role"], "content": m["content"]}

    payload = {
        "model": AGENT_MODEL,
        "messages": [to_openai(m) for m in messages],
        "tools": TOOL_SPECS,  # same tool list: Ollama copied OpenAI's format
        "temperature": 0.1,
    }
    if BACKEND == "openrouter":
        payload["reasoning"] = {"enabled": think}  # OpenRouter's common switch for thinking models (Qwen3 etc.)
    else:
        payload["reasoning_effort"] = "medium" if think else "low"  # gpt-oss on Groq always reasons a little
    url, key_name = CLOUD[BACKEND]
    headers = {"Authorization": f"Bearer {os.environ[key_name]}"}
    m = post_with_retry(url, payload, headers).json()["choices"][0]["message"]
    msg = {"role": "assistant", "content": m.get("content") or ""}
    if m.get("tool_calls"):
        # Gemini attaches extra_content.google.thought_signature (its private reasoning, encrypted) to each call
        # and rejects the next request (400) unless we send it back unchanged. Other providers don't send it.
        msg["tool_calls"] = [{"id": c["id"], "function": {"name": c["function"]["name"],
                                                         "arguments": json.loads(c["function"]["arguments"] or "{}")},
                              **({"extra_content": c["extra_content"]} if "extra_content" in c else {})}
                             for c in m["tool_calls"]]
    return msg


class Agent:
    def __init__(self, verbose: bool = True, think: bool = False):
        # The conversation IS the agent's memory: every turn, tool call and tool result is kept here
        # and sent to the LLM again on the next turn.
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.verbose = verbose
        self.think = think

    def reply(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        searched, nudged, cite_checked = False, False, False
        self.last_fake_cites = []  # what the citation check caught this turn (for eval_agent.py)
        for step in range(1, MAX_STEPS + 1):
            msg = call_llm(self.messages, think=self.think)
            msg.pop("thinking", None)  # keep the private reasoning out of the stored conversation
            self.messages.append(msg)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:  # a normal message
                # Guard: refusing without looking is a known failure. Push back once, then accept.
                if REFUSAL in msg["content"] and not searched and not nudged:
                    if self.verbose:
                        print(f"  [step {step}] 검색 없이 거절 → 먼저 검색하라고 다시 시킴")
                    self.messages.append({"role": "user", "content": NUDGE})
                    nudged = True
                    continue
                # Guard 2: every lecture time in the answer must come from a tool result. Push back once.
                fakes = fake_citations(msg["content"], self.messages)
                if fakes and not cite_checked:
                    if self.verbose:
                        print(f"  [step {step}] 검색 결과에 없는 시간 {fakes} → 다시 답하라고 시킴")
                    self.last_fake_cites = fakes
                    self.messages.append({"role": "user", "content": FAKE_CITE.format(times=", ".join(fakes))})
                    cite_checked = True
                    continue
                return msg["content"]

            for call in tool_calls:
                name = call["function"]["name"]
                searched = searched or name == "search_lecture"
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
                self.messages.append({"role": "tool", "tool_name": name, "tool_call_id": call.get("id"), "content": result})

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
