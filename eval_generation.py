"""Score the LLM's answers (the "G" in RAG), separately from retrieval.

For every eval question: hybrid search (top 5) -> LLM answer -> automatic checks.

Checks (simple and transparent on purpose, no LLM judge yet):
  answerable questions (36):
    correct      the answer contains one of the question's answer keywords
    refused      the answer says "다루지 않은" although the lecture covers it
  unanswerable questions (5):
    refused      the answer correctly says "다루지 않은"

Each answerable result is split by whether retrieval put the gold chunk in the top 5.
That separates "search failed" from "search succeeded but the LLM failed".

Known limit: keyword matching is lenient. An answer made from general knowledge can still
contain the keyword (e.g. "이해" for #14). Read the saved answers to check.
"""

import json
import sys
import time

from chunking import chunk_segments
from embedding import embed, load_model
from eval_retrieval import EVAL_PATH, first_hit_rank
from generation import LLM_MODEL, build_prompt, generate
from hybrid import HybridRetriever
from retrieval import Retriever

K = 5
REFUSAL = "다루지 않은"
OUT_PATH = "data/eval/gen_results.json"
HEDGES = ["일반적으로", "보통은", "일 것입니다"]  # signs of answering from general knowledge


def build_retriever(transcript=None):
    """Default: the saved index (turbo transcript). With a transcript path: build chunks in memory."""
    if transcript is None:
        return HybridRetriever(Retriever())
    chunks = chunk_segments(json.load(open(transcript))["segments"], max_chars=500, overlap=2)
    for c in chunks:
        c["source"] = "ch03_mlp_backprop"
    model = load_model()
    vectors = embed(model, [c["text"] for c in chunks])
    return HybridRetriever(Retriever(chunks=chunks, vectors=vectors, model=model))


def main(only_ids=None, transcript=None, out_path=OUT_PATH, k=K, model=LLM_MODEL):
    questions = json.load(open(EVAL_PATH))
    if only_ids:  # quick check on a few questions; full runs take ~20 min
        questions = [q for q in questions if q["id"] in only_ids]
    retriever = build_retriever(transcript)
    rows = []

    for i, q in enumerate(questions, 1):
        results = retriever.search(q["question"], k=k)
        t0 = time.time()
        answer = generate(build_prompt(q["question"], results), verbose=False, model=model)
        row = {
            "id": q["id"],
            "question": q["question"],
            "answerable": q["start"] is not None,
            "answer": answer,
            "refused": REFUSAL in answer,
            "hedged": any(h in answer for h in HEDGES),
            "seconds": round(time.time() - t0, 1),
        }
        if row["answerable"]:
            row["retrieved"] = first_hit_rank(results, q["start"], q["end"]) is not None
            row["correct"] = any(kw.lower() in answer.lower() for kw in q["keywords"])
        rows.append(row)
        print(f"[{i}/{len(questions)}] #{q['id']} {row['seconds']}s", flush=True)

    out = out_path if not only_ids else out_path.replace(".json", "_subset.json")
    json.dump(rows, open(out, "w"), ensure_ascii=False, indent=2)
    report(rows)


def report(rows):
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    got = [r for r in ans if r["retrieved"]]
    missed = [r for r in ans if not r["retrieved"]]

    def line(label, group):
        c = sum(r["correct"] for r in group)
        ref = sum(r["refused"] for r in group)
        print(f"  {label:<34} {len(group):>3}   correct {c:>2}   wrongly refused {ref:>2}")

    print(f"\nanswerable questions ({len(ans)})")
    line("search found the gold chunk", got)
    line("search missed the gold chunk", missed)
    line("total", ans)
    print(f"\nunanswerable questions ({len(unans)})")
    print(f"  correctly refused {sum(r['refused'] for r in unans)} / {len(unans)}")

    print("\nanswerable but NOT correct (read these):")
    for r in ans:
        if not r["correct"] or r["refused"]:
            print(f"  #{r['id']:<3} search={'hit ' if r['retrieved'] else 'miss'}  {r['question']}")
            print(f"        -> {r['answer'][:110].replace(chr(10), ' ')}")
    print("\nunanswerable but answered anyway:")
    for r in unans:
        if not r["refused"]:
            print(f"  #{r['id']:<3} {r['question']}\n        -> {r['answer'][:110].replace(chr(10), ' ')}")
    print(f"\nhedged (general-knowledge words like '일반적으로'): {[r['id'] for r in rows if r['hedged']]}")


if __name__ == "__main__":
    # usage: python eval_generation.py                     (all 41, saved index)
    #        python eval_generation.py 14 104            (only these ids)
    #        python eval_generation.py --transcript PATH --out PATH
    #        python eval_generation.py --k 3 --model exaone3.5:2.4b --out PATH
    args = sys.argv[1:]
    opt = lambda flag: args[args.index(flag) + 1] if flag in args else None
    ids = [int(a) for a in args if a.isdigit()]
    ids = [i for i in ids if str(i) != opt("--k")]  # "--k 3" is not a question id
    main(ids or None, transcript=opt("--transcript"), out_path=opt("--out") or OUT_PATH,
         k=int(opt("--k") or K), model=opt("--model") or LLM_MODEL)
