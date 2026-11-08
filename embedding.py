"""Turn chunks into vectors and save them as a searchable index.

An embedding model maps text to a fixed-size vector so that texts with similar
meaning land close together. We embed every chunk once (offline, like building
a DB index) and save:
  data/index/<name>.npy          float32 matrix, shape (n_chunks, dim)
  data/index/<name>.chunks.json  chunk metadata; row i of the matrix <-> chunks[i]
"""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import chunk_segments

MODEL_NAME = "BAAI/bge-m3"  # multilingual (Korean OK), 1024-dim
INDEX_DIR = Path("data/index")


def load_model() -> SentenceTransformer:
    # "mps" = Apple Silicon GPU. Falls back to CPU automatically if unavailable.
    return SentenceTransformer(MODEL_NAME, device="mps")


def embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    # normalize_embeddings=True scales every vector to length 1.
    # Then cosine similarity == plain dot product, which makes search a single matmul.
    return model.encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=True)


def build_index(transcript_path: str, max_chars: int = 500, overlap: int = 2) -> None:
    name = Path(transcript_path).stem
    segments = json.load(open(transcript_path))["segments"]
    chunks = chunk_segments(segments, max_chars=max_chars, overlap=overlap)
    for c in chunks:
        c["source"] = name  # which lecture this chunk came from

    model = load_model()
    vectors = embed(model, [c["text"] for c in chunks]).astype(np.float32)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / f"{name}.npy", vectors)
    json.dump(chunks, open(INDEX_DIR / f"{name}.chunks.json", "w"), ensure_ascii=False, indent=2)

    print(f"saved {vectors.shape} -> {INDEX_DIR / name}.npy")
    print(f"vector norms (should all be ~1.0): {np.linalg.norm(vectors, axis=1)[:5].round(4)}")


def sanity_check() -> None:
    """Build intuition: similar meaning -> high score, regardless of exact wording."""
    model = load_model()
    sentences = [
        "오류 역전파 알고리즘은 그레이디언트를 계산한다",       # A
        "backpropagation computes gradients",                  # B: same meaning, English
        "미분값을 출력층에서 입력층 방향으로 전달하며 구한다",   # C: same idea, no shared keywords
        "오늘 점심은 김치찌개를 먹었다",                         # D: unrelated
    ]
    v = embed(model, sentences)
    sim = v @ v.T  # (4, 4) matrix of pairwise cosine similarities
    print("\ncosine similarity to A:")
    for s, score in zip(sentences, sim[0]):
        print(f"  {score:.3f}  {s}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        sanity_check()
    else:
        build_index(sys.argv[1] if len(sys.argv) > 1 else "data/transcripts/ch03_mlp_backprop.json")
