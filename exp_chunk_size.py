"""Experiment: does chunk size change retrieval quality?

For each (max_chars, overlap) setting: re-chunk the transcript, embed, and run the same eval set.
Nothing is written to data/index; everything stays in memory.
"""

import json

from chunking import chunk_segments
from embedding import embed, load_model
from eval_retrieval import EVAL_PATH, evaluate
from retrieval import Retriever

TRANSCRIPT = "data/transcripts/ch03_mlp_backprop.json"
SETTINGS = [(150, 1), (300, 2), (500, 2), (800, 2)]  # (max_chars, overlap)
WATCH = [7, 12, 14, 15, 18]  # questions the baseline missed in top-3

segments = json.load(open(TRANSCRIPT))["segments"]
questions = json.load(open(EVAL_PATH))
model = load_model()

print(f"{'max_chars':>9} {'chunks':>6} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6}  baseline-missed now in top-3")
for max_chars, overlap in SETTINGS:
    chunks = chunk_segments(segments, max_chars=max_chars, overlap=overlap)
    for c in chunks:
        c["source"] = "ch03_mlp_backprop"
    vectors = embed(model, [c["text"] for c in chunks])
    retriever = Retriever(chunks=chunks, vectors=vectors, model=model)

    m = evaluate(retriever, questions, verbose=False)
    fixed = [qid for qid in WATCH if qid in m["hits@3"]]
    print(f"{max_chars:>9} {len(chunks):>6} {m['recall@1']:>6.3f} {m['recall@3']:>6.3f} "
          f"{m['recall@5']:>6.3f} {m['mrr']:>6.3f}  {fixed}")
