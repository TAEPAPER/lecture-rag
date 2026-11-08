"""What happens when several people ask at the same time?

Sends N questions to the running server (POST /ask) at the same moment, from N threads,
and records for each one: when its sources arrived, when its first answer token arrived, when it finished.

Run the server first:  uv run uvicorn server:app --port 8000
Then:                  uv run python exp_concurrency.py 1     (one user, baseline)
                       uv run python exp_concurrency.py 3     (three users at once)
"""

import json
import sys
import threading
import time

import requests

URL = "http://localhost:8000/ask"
QUESTIONS = [
    "미니배치 크기는 보통 얼마로 해?",
    "하이퍼탄젠트를 쓰면 레이블을 어떻게 바꿔야 해?",
    "오류 역전파는 전방 계산보다 시간이 얼마나 더 걸려?",
    "학습률을 정하는 방법에는 어떤 것들이 있어?",
]


def one_user(i: int, question: str, start: float, out: dict):
    t = {"sources": None, "queued_ahead": 0, "first_token": None, "done": None}
    with requests.post(URL, json={"question": question}, stream=True, timeout=600) as r:
        for line in r.iter_lines(decode_unicode=True):
            if line.startswith("data: ") and t["queued_ahead"] == -1:
                t["queued_ahead"] = json.loads(line[len("data: "):])["ahead"]
            if not line.startswith("event: "):
                continue
            event = line[len("event: "):]
            now = time.perf_counter() - start
            if event == "sources" and t["sources"] is None:
                t["sources"] = now
            elif event == "queued":
                t["queued_ahead"] = -1  # filled from the next data line
            elif event == "token" and t["first_token"] is None:
                t["first_token"] = now
            elif event in ("done", "error"):
                t["done"] = now
                t["error"] = event == "error"
    out[i] = t


def main(n: int):
    # Unload the LLM to clear Ollama's prompt cache (otherwise a question asked in an earlier run
    # comes back almost instantly), then load it again with a warm-up question before timing.
    requests.post("http://localhost:11434/api/generate", json={"model": "exaone3.5:7.8b", "keep_alive": 0})
    requests.post(URL, json={"question": "워밍업"}).close()
    out, start = {}, time.perf_counter()
    threads = [threading.Thread(target=one_user, args=(i, QUESTIONS[i % len(QUESTIONS)], start, out)) for i in range(n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print(f"{n} user(s) at once:")
    for i in range(n):
        t = out[i]
        print(f"  user {i + 1}: sources {t['sources']:5.1f}s   first token {t['first_token']:5.1f}s   "
              f"done {t['done']:5.1f}s   told ahead={t['queued_ahead']}" + ("   ERROR" if t.get("error") else ""))
    print(json.dumps({"n": n, "results": out}))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
