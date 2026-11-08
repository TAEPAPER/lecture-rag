"""Find the chunks most relevant to a question (the "R" in RAG).

Steps per query:
  1. embed the question with the SAME model used for the chunks
  2. score every chunk: dot product with the question vector (= cosine, vectors are normalized)
  3. return the top-k highest scores
"""

import json
import sys
from pathlib import Path

import numpy as np

from chunking import fmt_time
from embedding import INDEX_DIR, embed, load_model


class Retriever:
    def __init__(self, index_dir: Path = INDEX_DIR, chunks=None, vectors=None, model=None):
        # Experiments can pass chunks/vectors/model built in memory instead of loading from disk.
        if chunks is not None:
            self.chunks, self.vectors = chunks, vectors
        else:
            # Load every lecture's index and stack them into one matrix.
            # Row i of self.vectors <-> self.chunks[i], across all lectures.
            all_vectors, self.chunks = [], []
            for npy in sorted(index_dir.glob("*.npy")):
                all_vectors.append(np.load(npy))
                self.chunks += json.load(open(npy.with_suffix(".chunks.json")))
            self.vectors = np.vstack(all_vectors)  # (n_chunks, 1024)
        self.model = model or load_model()  # loaded once, reused for every query

    def search(self, query: str, k: int = 3) -> list[tuple[float, dict]]:
        q = embed(self.model, [query])[0]   # (1024,)
        scores = self.vectors @ q           # (n_chunks,) one matmul scores every chunk
        top = np.argsort(-scores)[:k]       # indices of the k highest scores
        return [(float(scores[i]), self.chunks[i]) for i in top]


def show(results: list[tuple[float, dict]], preview: int = 120) -> None:
    for rank, (score, c) in enumerate(results, 1):
        print(f"  {rank}. {score:.3f}  [{c['source']} {fmt_time(c['start'])}~{fmt_time(c['end'])}]  {c['text'][:preview]}...")


if __name__ == "__main__":
    retriever = Retriever()
    print(f"index: {retriever.vectors.shape[0]} chunks\n")

    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        queries = [
            "하이퍼탄젠트를 쓰면 레이블을 어떻게 바꿔야 해?",
            "역전파 알고리즘을 C로 직접 구현해 보라고 한 이유는?",
            "미니배치 크기는 어떻게 정해?",
            "트랜스포머의 어텐션은 어떻게 동작해?",  # NOT in this lecture
        ]
    for q in queries:
        print(f"Q: {q}")
        show(retriever.search(q, k=3))
        print()
