"""Grade a whole results file with the LLM judge, and compare with the old keyword grading.

  uv run python eval_judge.py agent data/eval/agent_results_groq_ctx.json
  uv run python eval_judge.py rag   data/eval/gen_results.json

agent results carry their own "context" (what the tools returned).
RAG results don't, but RAG's search is deterministic, so we simply search again with the same settings.
Writes <input>_judged.json with a "judge" field per question, after EVERY question.
Run the same command again after a crash: already-graded questions are skipped.
"""

import json
import os
import sys

from chunking import fmt_time
from eval_retrieval import EVAL_PATH
from judge import gold_text, judge


def rag_context(question: str, retriever) -> str:
    results = retriever.search(question, k=5)  # same as eval_generation.py (K=5, hybrid)
    return "\n\n".join(f"({fmt_time(c['start'])}~{fmt_time(c['end'])}) {c['text']}" for _, c in results)


def keyword_ok(r: dict) -> bool:
    """The old grading: keyword hit without refusal (answerable) / refusal (off-lecture)."""
    return (r["correct"] and not r["refused"]) if r["answerable"] else r["refused"]


def main(kind: str, path: str):
    questions = {q["id"]: q for q in json.load(open(EVAL_PATH))}
    out_path = path.replace(".json", "_judged.json")
    # Resume: if a previous run stopped halfway, keep what it already graded (saved after every item).
    rows = json.load(open(out_path)) if os.path.exists(out_path) else json.load(open(path))
    retriever = None
    if kind == "rag":
        from hybrid import HybridRetriever
        from retrieval import Retriever
        retriever = HybridRetriever(Retriever())

    for i, r in enumerate(rows, 1):
        if "judge" in r:
            continue  # graded in an earlier run
        q = questions[r["id"]]
        context = r["context"] if kind == "agent" else rag_context(q["question"], retriever)
        r["judge"] = judge(q["question"], r["answer"], gold_text(q["start"], q["end"]), context)
        json.dump(rows, open(out_path, "w"), ensure_ascii=False, indent=2)  # save every item: a crash loses nothing
        print(f"[{i}/{len(rows)}] #{r['id']} {r['judge']['correct']} {r['judge']['grounded']}", flush=True)

    n = len(rows)
    print(f"\n{'':24}keyword   judge")
    print(f"{'correct':24}{sum(keyword_ok(r) for r in rows):>4} / {n}  {sum(r['judge']['correct'] for r in rows):>4} / {n}")
    print(f"{'grounded':24}{'-':>9}  {sum(r['judge']['grounded'] for r in rows):>4} / {n}")
    print(f"{'correct AND grounded':24}{'-':>9}  {sum(r['judge']['correct'] and r['judge']['grounded'] for r in rows):>4} / {n}")
    print("\nkeyword vs judge disagree:")
    for r in rows:
        if keyword_ok(r) != r["judge"]["correct"]:
            print(f"  #{r['id']} keyword={keyword_ok(r)} judge={r['judge']['correct']}: {r['judge']['reason']}")
    print("\nnot grounded:")
    for r in rows:
        if not r["judge"]["grounded"]:
            print(f"  #{r['id']} {r['question']}: {r['judge']['reason']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
