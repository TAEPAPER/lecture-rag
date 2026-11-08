"""Experiment: does a worse transcript make search worse?

Same audio, transcribed by Whisper models of different sizes. For each transcript:
re-chunk (500 chars, overlap 2) -> embed -> run the SAME eval set.
The eval set stores answers as time ranges, so it works for any transcript.
"""

import json

from chunking import chunk_segments, fmt_time
from embedding import embed, load_model
from eval_retrieval import EVAL_PATH, evaluate
from hybrid import HybridRetriever
from retrieval import Retriever

TRANSCRIPTS = {  # model size (parameters) -> transcript file
    "tiny (39M)": "data/transcripts_exp/tiny/ch03_mlp_backprop.json",
    "base (74M)": "data/transcripts_exp/base/ch03_mlp_backprop.json",
    "small (244M)": "data/transcripts_exp/small/ch03_mlp_backprop.json",
    "medium (769M)": "data/transcripts_exp/medium/ch03_mlp_backprop.json",
    "turbo (809M)": "data/transcripts/ch03_mlp_backprop.json",  # what we use now
}
SAMPLE_AT = 1080  # seconds: "여러분이 C로 코딩을 할 수가 있어요"

questions = json.load(open(EVAL_PATH))
model = load_model()

print(f"{'whisper':>14} {'chars':>6} {'dense R@3':>9} {'hybrid R@3':>10} {'hybrid R@5':>10}")
samples = {}
for name, path in TRANSCRIPTS.items():
    segments = json.load(open(path))["segments"]
    chunks = chunk_segments(segments, max_chars=500, overlap=2)
    for c in chunks:
        c["source"] = "ch03_mlp_backprop"
    dense = Retriever(chunks=chunks, vectors=embed(model, [c["text"] for c in chunks]), model=model)

    d = evaluate(dense, questions, verbose=False)
    h = evaluate(HybridRetriever(dense), questions, verbose=False)
    chars = sum(len(s["text"]) for s in segments)
    print(f"{name:>14} {chars:>6} {d['recall@3']:>9.3f} {h['recall@3']:>10.3f} {h['recall@5']:>10.3f}")

    near = [s for s in segments if s["start"] <= SAMPLE_AT + 15 and s["end"] >= SAMPLE_AT - 5]
    samples[name] = " ".join(s["text"].strip() for s in near)

print(f"\nsame moment ({fmt_time(SAMPLE_AT)}) in each transcript:")
for name, text in samples.items():
    print(f"  {name:>14}: {text[:90]}")
