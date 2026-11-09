"""LLM-as-judge: grade answers by MEANING, not by matching keywords.

Keyword grading broke in two ways (ROADMAP 2026-11-08):
  - a right answer written as '‑1' (U+2011 hyphen) didn't match the keyword '-1'      -> wrong "wrong"
  - a made-up answer citing a REAL retrieved time (21:00~22:24) passed the time check  -> missed fabrication

So we ask another LLM call two separate questions per answer:
  correct   Does the answer say what the LECTURE says?      (compared with the gold lecture text)
  grounded  Is every claim backed by what the agent SAW?    (compared with the agent's search results)
They differ: an answer can be correct but not grounded (true from general knowledge, but the agent never read it).

The judge is an LLM too, so it can be wrong. Step 5 checks the judge against human labels.

Run:  uv run python judge.py        (the three known cases below)
"""

import json
import os

import agent  # reuse .env loading and the cloud backend table

JUDGE_BACKEND = "groq"
JUDGE_MODEL = "qwen/qwen3.8-27b"  # a different family from the Groq agent (gpt-oss): a judge may go easy on its own style
# (also: gpt-oss-120b hit Groq's free daily limit of 200K tokens; each model has its own quota)
TRANSCRIPT = "data/transcripts/ch03_mlp_backprop.json"

JUDGE_PROMPT = """너는 강의 질의응답 채점자다. 아래 자료를 보고 답변을 채점해 JSON으로만 출력한다.

[질문]
{question}

[정답 근거: 강의에서 실제로 말한 부분]
{gold}

[답변자가 검색으로 본 강의 구간]
{context}

[채점할 답변]
{answer}

채점 기준:
- correct: 답변의 핵심 내용이 [정답 근거]와 같은 뜻이면 true. 표기·말투·길이 차이는 무시한다.
  "왜", "이유"를 묻는 질문은 답변이 든 이유가 [정답 근거]의 이유와 같아야 true다. 주제만 맞고 이유가 다르면 false.
  [정답 근거]가 "(강의에 없음)"이면, 답변이 "강의에서 다루지 않은 내용"이라고 거절했을 때만 true.
- grounded: 답변의 모든 주장이 [답변자가 검색으로 본 강의 구간]에 실제로 있으면 true.
  구간에 없는 내용을 하나라도 덧붙였으면(일반 상식이 맞더라도) false. 거절한 답변은 true.
  [1], [3] 같은 대괄호 번호와 (21:00~22:24) 같은 시간은 출처 표시일 뿐 주장이 아니므로 판단에서 제외한다.
  그 외 괄호 안의 사람 이름·영어 용어·설명, 그리고 "확률", "평균"처럼 구간에 없는 단어 하나로 뜻을 바꾼 표현은 주장으로 보고 검사한다.
- reason: 판단 이유를 한국어 한두 문장으로.

출력 형식: {{"correct": true/false, "grounded": true/false, "reason": "..."}}"""


def gold_text(start, end) -> str:
    """The lecture's own words in the gold time range (the answer key)."""
    if start is None:
        return "(강의에 없음)"
    segs = json.load(open(TRANSCRIPT))["segments"]
    return " ".join(s["text"].strip() for s in segs if s["start"] < end and s["end"] > start)


def judge(question: str, answer: str, gold: str, context: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, context=context or "(검색 안 함)", answer=answer)
    url, key_name = agent.CLOUD[JUDGE_BACKEND]
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,                            # a grader should be as repeatable as possible
        "response_format": {"type": "json_object"},  # force parseable JSON instead of free text
    }
    if JUDGE_MODEL.startswith("openai/gpt-oss"):
        payload["reasoning_effort"] = "medium"  # let it think a bit: grading needs care, not speed
    headers = {"Authorization": f"Bearer {os.environ[key_name]}"}
    r = agent.post_with_retry(url, payload, headers)  # same retry on 429 / 5xx as the agent
    return json.loads(r.json()["choices"][0]["message"]["content"])


if __name__ == "__main__":
    questions = {q["id"]: q for q in json.load(open("data/eval/qa_ch03.json"))}
    answers = {r["id"]: r["answer"] for r in json.load(open("data/eval/agent_results_groq.json"))}
    # Known cases from the Groq runs. Context here = the section each answer cited (full search results
    # are not saved yet; step 3 adds them).
    cases = [
        (3, gold_text(96, 135)),     # right answer, keyword grader said wrong ('‑1')
        (105, gold_text(1260, 1344)),  # made-up answer citing a real section (21:00~22:24, mini-batch)
        (101, None),                 # refusal on an off-lecture question
    ]
    for qid, context in cases:
        q = questions[qid]
        answer = answers[qid] if qid != 101 else "강의에서 다루지 않은 내용입니다."
        result = judge(q["question"], answer, gold_text(q["start"], q["end"]), context)
        print(f"#{qid} {q['question']}\n  answer: {answer[:80].replace(chr(10), ' ')}...\n  judge:  {result}\n")
