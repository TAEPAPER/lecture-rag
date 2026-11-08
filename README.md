# Lecture RAG

강의 음성에 질문하면, **강의 내용만 근거로** 답하고 **근거가 나온 시간**을 알려 주는 RAG.
프레임워크 없이 바닥부터 만들고, 모든 개선을 시험지(평가셋) 점수로 확인했다. 전부 로컬(Apple M3, 16GB)에서 돈다.

```
Q: 미니배치 크기는 보통 얼마로 해?
A: 미니 배치 크기는 보통 256, 128, 512 정도로 많이 사용됩니다. [1]
sources:
  [1] 21:00~22:24
```

## 파이프라인

| 단계 | 방법 | 파일 |
|---|---|---|
| 전사 | mlx-whisper `large-v3-turbo` (`--condition-on-previous-text False --fp16 False`) | — |
| 청킹 | Whisper 세그먼트를 500자 단위로 묶음, overlap 2세그먼트, 타임스탬프 유지 | `chunking.py` |
| 임베딩 | `BAAI/bge-m3`, 정규화 → 코사인 = 내적 | `embedding.py` |
| 검색 | numpy 내적(뜻) + 직접 구현한 BM25(키워드, 한국어 글자 bigram)를 RRF로 결합, top-5 | `retrieval.py` `bm25.py` `hybrid.py` |
| 생성 | Ollama `exaone3.5:7.8b`, 발췌 밖 지식 사용 금지 프롬프트(v2), 스트리밍 | `generation.py` |
| 평가 | 검색: Recall@k, MRR / 생성: 정답 키워드, 거절 여부 | `eval_retrieval.py` `eval_generation.py` |

## 평가셋

`data/eval/qa_ch03.json`: 강의 전사본을 읽고 직접 만든 41문제 (답 있음 36, 강의에 없는 내용 5).
정답을 청크 번호가 아닌 **시간 구간**으로 저장해서, 청크 크기나 전사 모델을 바꿔도 같은 시험지를 쓸 수 있다.
검색된 청크의 시간 구간이 정답 구간과 겹치면 정답으로 채점한다.

## 결과

| 실험 | 결과 | 결정 |
|---|---|---|
| 기준 (뜻 검색, top-3) | Recall@3 0.861, MRR 0.832 | — |
| 청크 크기 150~800자 (`exp_chunk_size.py`) | 차이 최대 1문제 | 500자 유지 |
| BM25 결합 (`exp_hybrid.py`) | Recall@3 0.889, Recall@5 0.917 → **0.972** | 채택 |
| 프롬프트 v2 | 강의에 없는 질문 거절 3/5 → **5/5**, 맞은 답변 37 → 36 (비슷) | 채택 (지어내기 감소) |
| Whisper 크기 (`exp_whisper_size.py`) | 검색 점수는 tiny ≈ turbo, **최종 답변은 22 vs 36** | turbo 유지 |
| 지연시간 분해 (`exp_latency.py`) | 약 10초 중 **83%가 LLM의 프롬프트 읽기(prefill)**, 검색은 0.3ms | — |
| 조각 5→3개 / 작은 LLM (Gemma 3 4B) | 3개: 26% 빨라지고 정답 −1 / 4B: 거의 안 빨라지고 정답 36→21 | 현재 설정 유지 |

현재 설정: **41문제 중 36개 정답** (키워드 채점 기준).

### 핵심 발견
- **전사 오류는 검색 점수에는 숨어 있다가 최종 답변에서 드러난다.** Whisper tiny 전사본도 정답 위치는 찾았지만(34/36 vs 35/36), 그 위치의 글이 깨져 있어 LLM이 답하지 못했다(22/41 vs 36/41). 단계별 지표만 보면 병목을 잘못 짚는다.
- 검색은 관련 없는 질문에도 항상 무언가를 돌려준다. 답 있는 질문과 없는 질문의 유사도 분포가 겹쳐서(0.457~0.757 vs 0.439~0.567) 임계값으로 거를 수 없다 → LLM 단계에서 거절.
- 주제가 섞인 청크는 검색 순위가 떨어진다(의미 희석). 질문의 흔한 주제어가 핵심 키워드를 압도하기도 한다.
- 키워드 채점은 후하기도(추측성 답변을 정답 처리) 박하기도(동의어를 오답 처리) 하다. LLM 답변은 같은 입력에도 달라질 수 있어서 1~2문제 차이는 우연일 수 있다.

실험별 상세 수치와 관찰: [ROADMAP.md](ROADMAP.md)

## 실행

```bash
uv sync
brew install ffmpeg ollama && brew services start ollama
ollama pull exaone3.5:7.8b

# 1) 전사 (오디오는 저장소에 없음, data/audio/ 에 직접 준비)
uv run mlx_whisper data/audio/ch03_mlp_backprop.m4a --model mlx-community/whisper-large-v3-turbo \
  --language ko --output-format json --output-dir data/transcripts \
  --condition-on-previous-text False --fp16 False

# 2) 인덱스 생성 → 3) 질문
uv run python embedding.py
uv run python generation.py "하이퍼탄젠트를 쓰면 레이블을 어떻게 바꿔야 해?"

# 평가
uv run python eval_retrieval.py
uv run python eval_generation.py          # 41문제, 약 20분
```

## 데이터 출처

강의: 전북대학교 오일석, 「기계학습」 (KOCW, 2019-2), 3장 3.4~3.7절.
이용조건 [CC BY-NC 2.0 KR](http://creativecommons.org/licenses/by-nc/2.0/kr/). `data/transcripts/`의 전사본은 이 강의에서 파생된 것이며 비영리 학습 목적으로만 사용한다.
원본 오디오는 저장소에 포함하지 않는다.
