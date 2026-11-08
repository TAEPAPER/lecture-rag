"""Score retrieval against the hand-made eval set.

A retrieved chunk counts as correct if its time range OVERLAPS the gold time range.
(Gold answers are stored as time ranges, not chunk ids, so the eval set survives re-chunking.)

Metrics (answerable questions only):
  Recall@k  fraction of questions whose correct chunk appears in the top k
  MRR       mean of 1/rank of the first correct chunk (rank 1 -> 1.0, rank 2 -> 0.5, missing -> 0)
"""

import json
import sys

import numpy as np

from chunking import fmt_time
from retrieval import Retriever

EVAL_PATH = "data/eval/qa_ch03.json"
KS = [1, 3, 5]


def overlaps(chunk: dict, start: float, end: float) -> bool:
    return chunk["start"] < end and chunk["end"] > start


def first_hit_rank(results: list[tuple[float, dict]], start: float, end: float) -> int | None:
    """1-based rank of the first chunk overlapping the gold range, or None if not found."""
    for rank, (_, chunk) in enumerate(results, 1):
        if overlaps(chunk, start, end):
            return rank
    return None


def evaluate(retriever: Retriever, questions: list[dict], verbose: bool = True) -> dict:
    answerable = [q for q in questions if q["start"] is not None]
    unanswerable = [q for q in questions if q["start"] is None]
    n_all = len(retriever.chunks)

    ranks, misses = [], []
    for q in answerable:
        results = retriever.search(q["question"], k=n_all)  # full ranking
        rank = first_hit_rank(results, q["start"], q["end"])
        ranks.append(rank)
        if rank is None or rank > 3:
            misses.append((q, rank, results[0]))

    metrics = {f"recall@{k}": np.mean([r is not None and r <= k for r in ranks]) for k in KS}
    metrics["mrr"] = np.mean([1 / r if r else 0 for r in ranks])
    metrics["hits@3"] = [q["id"] for q, r in zip(answerable, ranks) if r is not None and r <= 3]

    # Top-1 score: can a score threshold tell "answerable" from "not in the lecture"?
    top1 = lambda qs: [retriever.search(q["question"], k=1)[0][0] for q in qs]
    s_ans, s_un = top1(answerable), top1(unanswerable)

    if verbose:
        print(f"{len(answerable)} answerable questions, {n_all} chunks\n")
        for name in [f"recall@{k}" for k in KS] + ["mrr"]:
            print(f"  {name:10s} {metrics[name]:.3f}")

        print(f"\nmissed top-3 ({len(misses)}):")
        for q, rank, (score, top) in misses:
            print(f"  #{q['id']:<3} rank={rank}  {q['question']}")
            print(f"        gold {fmt_time(q['start'])}~{fmt_time(q['end'])}"
                  f"  | got {fmt_time(top['start'])}~{fmt_time(top['end'])} ({score:.3f})"
                  + (f"  [{q['note']}]" if q.get("note") else ""))

        print("\ntop-1 similarity score:")
        print(f"  answerable    min {min(s_ans):.3f}  mean {np.mean(s_ans):.3f}  max {max(s_ans):.3f}")
        print(f"  unanswerable  min {min(s_un):.3f}  mean {np.mean(s_un):.3f}  max {max(s_un):.3f}")

    return metrics


if __name__ == "__main__":
    questions = json.load(open(sys.argv[1] if len(sys.argv) > 1 else EVAL_PATH))
    evaluate(Retriever(), questions)
