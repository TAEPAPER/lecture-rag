"""Keyword search (BM25), written from scratch.

Idea: a chunk scores high if it contains the question's words,
  - more often            (term frequency, with diminishing returns)
  - and those words are rare across all chunks (rare word = more informative, "IDF")
  - normalized by chunk length (a long chunk naturally contains more words)

Korean tokenization: Korean attaches particles to words ("역전파를", "역전파가"),
so whole-word matching fails. Without a morphological analyzer, a simple trick is
character bigrams: "역전파를" -> ["역전", "전파", "파를"]. "역전파가" shares "역전", "전파".
"""

import math
from collections import Counter

import numpy as np


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in text.lower().split():
        word = "".join(ch for ch in word if ch.isalnum())  # drop punctuation
        if len(word) == 1:
            tokens.append(word)  # keep single-char words like "c"
        else:
            tokens += [word[i:i + 2] for i in range(len(word) - 1)]
    return tokens


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b  # standard default values
        self.docs = [Counter(tokenize(t)) for t in texts]           # token -> count, per chunk
        self.lengths = np.array([sum(d.values()) for d in self.docs])
        self.avg_length = self.lengths.mean()

        n = len(texts)
        doc_freq = Counter(tok for d in self.docs for tok in d)      # how many chunks contain each token
        self.idf = {tok: math.log(1 + (n - df + 0.5) / (df + 0.5)) for tok, df in doc_freq.items()}

    def scores(self, query: str) -> np.ndarray:
        """One score per chunk, shape (n_chunks,)."""
        scores = np.zeros(len(self.docs))
        length_norm = 1 - self.b + self.b * self.lengths / self.avg_length
        for tok in set(tokenize(query)):
            if tok not in self.idf:
                continue  # token never appears in any chunk
            tf = np.array([d[tok] for d in self.docs])
            scores += self.idf[tok] * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
        return scores
