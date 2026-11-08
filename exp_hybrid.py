"""Experiment: meaning search vs keyword search vs both combined (same 500-char chunks)."""

import json

from eval_retrieval import EVAL_PATH, evaluate
from hybrid import BM25Retriever, HybridRetriever
from retrieval import Retriever

WATCH = [7, 12, 14, 15, 18]  # questions the baseline missed in top-3

questions = json.load(open(EVAL_PATH))
dense = Retriever()  # baseline index from data/index (500 chars, overlap 2)

print(f"{'method':>8} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6}  baseline-missed now in top-3 | newly missed")
base_hits = None
for name, retriever in [("dense", dense), ("bm25", BM25Retriever(dense.chunks)), ("hybrid", HybridRetriever(dense))]:
    m = evaluate(retriever, questions, verbose=False)
    if base_hits is None:
        base_hits = set(m["hits@3"])
    fixed = [q for q in WATCH if q in m["hits@3"]]
    broken = sorted(base_hits - set(m["hits@3"]))
    print(f"{name:>8} {m['recall@1']:>6.3f} {m['recall@3']:>6.3f} {m['recall@5']:>6.3f} {m['mrr']:>6.3f}  {fixed} | {broken}")
