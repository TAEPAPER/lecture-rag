"""Is the agent better than plain RAG? Run the agent on the SAME 41 eval questions.

Same scoring as eval_generation.py (answer keywords, refusal), plus agent-specific numbers:
  searches     how many times the agent called search_lecture for the question
  found        whether any search result overlapped the gold time range
  timestamp    whether the answer mentions a lecture time like 21:00
Each question starts a fresh Agent (no memory carried between questions).
"""

import json
import re
import sys
import statistics
import time
from collections import Counter

import agent as agent_module
from eval_generation import REFUSAL
from eval_retrieval import EVAL_PATH

OUT_PATH = "data/eval/agent_results_latest.json"  # pass --out to keep a named run
TIME_RE = re.compile(r"\d{1,2}:\d{2}")


def gold_overlaps(text: str, start: float, end: float) -> bool:
    """Did a tool result include a section overlapping the gold range? (sections are '(mm:ss~mm:ss) ...')"""
    for a, b in re.findall(r"\((\d+:\d+)~(\d+:\d+)\)", text):
        to_s = lambda t: int(t.split(":")[0]) * 60 + int(t.split(":")[1])
        if to_s(a) < end and to_s(b) > start:
            return True
    return False


def main(think: bool = False, out_path: str = OUT_PATH):
    questions = json.load(open(EVAL_PATH))
    rows = []
    for i, q in enumerate(questions, 1):
        a = agent_module.Agent(verbose=False, think=think)
        t0 = time.time()
        answer = a.reply(q["question"])
        tool_msgs = [m for m in a.messages if m["role"] == "tool"]
        row = {
            "id": q["id"],
            "question": q["question"],
            "answerable": q["start"] is not None,
            "answer": answer,
            "refused": REFUSAL in answer,
            "searches": sum(m["tool_name"] == "search_lecture" for m in tool_msgs),
            "nudged": any(m["role"] == "user" and m["content"] == agent_module.NUDGE for m in a.messages),
            "timestamp": bool(TIME_RE.search(answer)),
            "fake_cite_caught": a.last_fake_cites,  # made-up times the guard caught (then asked to re-answer)
            "fake_cite_final": agent_module.fake_citations(answer, a.messages),  # made-up times left in the final answer
            "seconds": round(time.time() - t0, 1),
            # everything the tools returned = what the agent SAW (judge.py checks the answer against it)
            "context": "\n\n".join(m["content"] for m in tool_msgs),
        }
        if row["answerable"]:
            row["found"] = any(gold_overlaps(m["content"], q["start"], q["end"]) for m in tool_msgs)
            row["correct"] = any(kw.lower() in answer.lower() for kw in q["keywords"])
        rows.append(row)
        print(f"[{i}/{len(questions)}] #{q['id']} searches={row['searches']} {row['seconds']}s", flush=True)

    json.dump(rows, open(out_path, "w"), ensure_ascii=False, indent=2)

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    ok = sum(r["correct"] and not r["refused"] for r in ans) + sum(r["refused"] for r in unans)
    print(f"\ncorrect overall            {ok} / {len(rows)}")
    print(f"answerable: correct        {sum(r['correct'] and not r['refused'] for r in ans)} / {len(ans)}")
    print(f"answerable: wrongly refused {sum(r['refused'] for r in ans)}")
    print(f"answerable: gold found by its searches {sum(r['found'] for r in ans)} / {len(ans)}")
    print(f"unanswerable: refused      {sum(r['refused'] for r in unans)} / {len(unans)}")
    print(f"searches per question      {dict(sorted(Counter(r['searches'] for r in rows).items()))}  (count -> questions)")
    print(f"nudged to search           {sum(r['nudged'] for r in rows)}")
    print(f"answers with a timestamp   {sum(r['timestamp'] for r in rows)} / {len(rows)}")
    print(f"fake citations caught      {sum(bool(r.get('fake_cite_caught')) for r in rows)}")
    print(f"fake citations in final    {sum(bool(r.get('fake_cite_final')) for r in rows)}")
    print(f"seconds per question       median {statistics.median(r['seconds'] for r in rows)}")


if __name__ == "__main__":
    # usage: python eval_agent.py [--think] [--out PATH]
    args = sys.argv[1:]
    main(think="--think" in args, out_path=args[args.index("--out") + 1] if "--out" in args else OUT_PATH)
