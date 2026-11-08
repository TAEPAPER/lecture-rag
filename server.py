"""Phase 4: serve Lecture RAG over HTTP, streaming the answer token by token.

Run:   uv run uvicorn server:app --port 8000      then open http://localhost:8000
API:   POST /ask  {"question": "...", "k": 5}  ->  Server-Sent Events (SSE):
         event: sources  data: [{"n": 1, "start": "21:00", "end": "22:24"}, ...]   (sent right after search)
         event: token    data: "미니"                                              (one per LLM token)
         event: done     data: {"search_s": .., "first_token_s": .., "total_s": ..}
         event: error    data: "..."                                               (if something fails)
       GET /health

Why stream: ~83% of the time is the LLM reading the prompt (exp_latency.py). We cannot remove that
wait, but the user sees the sources immediately and the answer from its first token on.
"""

import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from chunking import fmt_time
from generation import build_prompt, stream_answer
from hybrid import HybridRetriever
from retrieval import Retriever

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model and index ONCE at startup, not per request (like a DB connection pool).
    state["retriever"] = HybridRetriever(Retriever())
    yield


app = FastAPI(title="Lecture RAG", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=10)


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["retriever"].chunks)}


@app.post("/ask")
def ask(req: AskRequest):
    # A plain (sync) generator: FastAPI runs it in a worker thread, so blocking calls
    # (embedding on MPS, reading Ollama's stream with requests) don't block the event loop.
    def events():
        t0 = time.perf_counter()
        try:
            results = state["retriever"].search(req.question, k=req.k)
            t_search = time.perf_counter() - t0
            yield sse("sources", [
                {"n": i, "start": fmt_time(c["start"]), "end": fmt_time(c["end"])}
                for i, (_, c) in enumerate(results, 1)
            ])
            first_token, stats = None, {}
            for item in stream_answer(build_prompt(req.question, results)):
                if isinstance(item, str):
                    if first_token is None:
                        first_token = time.perf_counter() - t0
                    yield sse("token", item)
                else:
                    stats = item
            yield sse("done", {
                "search_s": round(t_search, 3),
                "first_token_s": round(first_token or 0, 2),
                "total_s": round(time.perf_counter() - t0, 2),
                "prompt_tokens": stats.get("prompt_eval_count"),
                "answer_tokens": stats.get("eval_count"),
            })
        except Exception as e:  # e.g. Ollama not running
            yield sse("error", f"{type(e).__name__}: {e}")

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
def page():
    return PAGE


PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lecture RAG</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; line-height: 1.6; }
  form { display: flex; gap: .5rem; }
  input { flex: 1; padding: .6rem; font-size: 1rem; }
  button { padding: .6rem 1rem; font-size: 1rem; }
  #answer { white-space: pre-wrap; margin-top: 1.2rem; min-height: 3rem; }
  .muted { color: #777; font-size: .9rem; }
</style></head>
<body>
  <h2>Lecture RAG</h2>
  <p class="muted">오일석 「기계학습」 3장 (3.4~3.7절) 강의에 질문하기</p>
  <form id="f"><input id="q" placeholder="예: 미니배치 크기는 보통 얼마로 해?" autofocus><button>질문</button></form>
  <div id="status" class="muted"></div>
  <div id="answer"></div>
  <div id="sources" class="muted"></div>
  <div id="timing" class="muted"></div>
<script>
const $ = id => document.getElementById(id);
$("f").onsubmit = async (e) => {
  e.preventDefault();
  const question = $("q").value.trim();
  if (!question) return;
  $("answer").textContent = ""; $("sources").textContent = ""; $("timing").textContent = "";
  $("status").textContent = "검색 중...";
  const t0 = performance.now();
  const res = await fetch("/ask", {method: "POST", headers: {"Content-Type": "application/json"},
                                    body: JSON.stringify({question})});
  const reader = res.body.getReader(), decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += decoder.decode(value, {stream: true});
    let i;
    while ((i = buf.indexOf("\\n\\n")) >= 0) {          // one SSE event ends with a blank line
      const raw = buf.slice(0, i); buf = buf.slice(i + 2);
      const event = raw.match(/^event: (.*)$/m)[1];
      const data = JSON.parse(raw.match(/^data: (.*)$/m)[1]);
      if (event === "sources") {
        $("status").textContent = "강의 조각을 찾았어요. LLM이 읽는 중...";
        $("sources").textContent = "근거 후보: " + data.map(s => `[${s.n}] ${s.start}~${s.end}`).join("  ");
      } else if (event === "token") {
        if (!$("answer").textContent) $("status").textContent = `첫 글자까지 ${((performance.now()-t0)/1000).toFixed(1)}초`;
        $("answer").textContent += data;
      } else if (event === "done") {
        $("timing").textContent = `검색 ${data.search_s}s · 첫 글자 ${data.first_token_s}s · 전체 ${data.total_s}s`;
      } else if (event === "error") {
        $("status").textContent = "오류: " + data;
      }
    }
  }
};
</script>
</body></html>"""
