# Technical decisions

Status: implemented and measured (see README "Results"). Numbers below come from
`eval/results/*.json` on the 187-question test set.

**Structure-aware chunking (Chapter/Article/Clause).** Legal text has a fixed
hierarchy and answers must cite articles. Measured: dense search over
Chapter/Article/Clause chunks beats fixed 512-token chunks by +28.4 Recall@5
points (0.872 vs 0.588) and +0.40 MRR (0.776 vs 0.375). Clause-level chunks also
beat article-level chunks (0.893 vs 0.840 with rerank) because long articles
dilute the embedding, so `structure` is the default collection and `article`
stays as an ablation row.

**bge-m3 for dense + sparse.** One model produces both vectors and supports
Vietnamese well. Measured: adding sparse with RRF fusion improves MRR from 0.776
to 0.785 at roughly the same Recall@5 (0.872 -> 0.866), and Decree 168 Recall@5
from 0.773 to 0.864 — the exact-term/article-number cases the plan predicted.

**Qdrant hybrid search.** Native dense+sparse with fusion, so no separate keyword
engine is needed. Collections: `legal_docs` (structure, default),
`legal_docs_article`, `legal_docs_baseline` (fixed 512 tokens, frozen after
week 2).

**Rerank with bge-reranker-v2-m3.** Measured: +2.7 Recall@5 points (0.866 ->
0.893) and +0.026 MRR (0.785 -> 0.811), at ~1.8 s p50 on a 6 GB RTX 4050 (the
reranker scores 30 candidates per query). Kept as the default because answer quality matters more
than latency here; `mode="hybrid"` is the fast path.

**"Not found" via thresholds.** Dense search uses a cosine cutoff
(`DENSE_THRESHOLD`, default 0.45); reranked search uses a normalised reranker
cutoff (`RERANK_THRESHOLD`, default 0.20). Below the cutoff the retriever returns
no hits and the agent answers "not found" without calling the LLM.

**Regex cross-references in Neo4j.** 624 Article nodes and 635 REFERENCES /
4 DETAILS edges for the four documents. Extraction is deliberately conservative:
an explicit law/decree name near the "Điều N" mention wins, otherwise the
reference is assumed to stay inside the same document. Cost: some genuine
cross-document references are missed; benefit: almost no false edges. Edge counts
are logged on every ingest for spot checks.

*Phase 1:* removed from the query path; the graph returns in a later phase, measured against the phase-1 baseline.

**LangGraph router with rules first.** The router checks two unambiguous shapes
(comparison between two named documents; explicit reference/citation questions)
before asking the LLM to classify, which keeps the common path free of an extra
LLM round trip. Tools are exposed through MCP so retrieval stays decoupled from
the agent.

*Phase 1:* replaced by a plain answer function (`rag/answer.py`); MCP removed.

**Hosted LLM via OpenCode Go, MiMo-V2.6-Flash.** The app uses one
OpenAI-compatible client class pointed at the OpenCode Go gateway with
`mimo-v2.6-flash`. The gateway asks clients to identify themselves, so the OpenAI
client sends a stable session id and user agent through `LLM_API_HEADERS`
(`x-opencode-session`, `User-Agent`), and its model list lives at a
non-standard path (`LLM_MODELS_URL`). MiMo is a hybrid-thinking model whose
reasoning tokens count against the completion budget — measured: a 700-token
answer budget was entirely consumed by reasoning, returning an empty answer —
so requests disable thinking via `LLM_EXTRA_BODY`
(`chat_template_kwargs.enable_thinking=false`). Requires a Go API key from
the OpenCode Console.

**Test set from public QA datasets, not the Zalo corpus.** The plan assumed "Zalo
AI Legal Text Retrieval 2023". That track does not exist (the legal retrieval
challenge ran in 2021), and the 2021 corpus predates all four of our documents:
0 of its 3,196 gold passages map to 45/2019/QH14, 31/2024/QH15, 36/2024/QH15 or
168/2024/NĐ-CP. The test set is therefore filtered from public Vietnamese legal
QA datasets that do carry article-level gold (see `eval/README.md`), mapped to
our article ids and validated against the ingested corpus, plus LLM-generated and
cross-article "hard" questions. The Zalo mirror is kept under `eval/zalo_raw/`
(gitignored) for provenance.

**CI runs the eval offline.** The eval stage pipes the corpus through the
deterministic hashing embedder so it needs no 2 GB download on shared runners; it
proves the pipeline end to end and that metrics can be computed and published.
Meaningful numbers require bge-m3 + bge-reranker-v2-m3 and are produced locally
(`make eval`).

**FastAPI async + SSE.** Streaming answers improve perceived latency; document
processing runs in a background task with a JSON status store. SSE events:
`meta`, `sources`, `token`, `done`.

*Phase 1:* SSE removed; `/query` returns one JSON response.

**Optional Langfuse tracing.** `observability.py` wraps agent runs in a Langfuse
trace when keys are present and is a no-op otherwise, so the dependency never
blocks local development.

*Phase 1:* removed with the agent; logs are plain text for now.

**uv + ruff + pytest.** Fast, single-tool workflow for deps, lint, and tests; the
test suite is hermetic (hashing embedder, stub services) and runs in ~6 s.

**Ablation-driven development.** Every retrieval change gets an
`eval/results/<config>.json` row. Current table on 187 questions: baseline
fixed-512 (0.588 R@5, 0.375 MRR), structure dense (0.872, 0.776), +sparse/hybrid
(0.866, 0.785), +rerank (0.893, 0.811), article granularity (0.840, 0.741).

**GPU memory budget.** On a 6 GB card, bge-m3 and the reranker must fit together
without an LLM resident. Loading everything at once can make the reranker hit CUDA
OOM, so `eval.agent_eval` may need `EMBED_DEVICE=cpu` and a small `--limit`; the
retrieval ablation does not load the LLM.

**Phase 1: local-first RAG core.** Runs on one machine: Qdrant embedded at
`QDRANT_PATH` (no Docker), no Neo4j, no LangGraph, no MCP. Rationale: the goal is to
learn and measure RAG, and each extra service is one more thing to operate before
the numbers mean anything. Measured on the same 187 questions, embedded Qdrant
gives Recall@5 0.872 and MRR 0.785 for hybrid search, against 0.866 and 0.785 on
the server (phase 0); the one-question gap is noise on this test set.

**LLM as a seam, two adapters.** `LLMClient` is the only interface the answer path
depends on. `LLM_PROVIDER=openai` covers every OpenAI-compatible server, and
`LLM_PROVIDER=anthropic` covers the Messages API. A second real adapter, not a
hypothetical one, is what justifies the seam. The Anthropic adapter uses httpx, so
it adds no SDK dependency.

**Grounding checked in code, not by the model.** An answer is "not found" when no
hit clears the threshold, when the model says `NOT_FOUND`, or when no citation
points at a context that was actually given. The model is trusted for wording only.
