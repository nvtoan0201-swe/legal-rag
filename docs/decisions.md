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

**LangGraph router with rules first.** The router checks two unambiguous shapes
(comparison between two named documents; explicit reference/citation questions)
before asking the LLM to classify, which keeps the common path free of an extra
LLM round trip. Tools are exposed through MCP so retrieval stays decoupled from
the agent.

**Local Qwen3-8B in 4-bit via Ollama.** No GPU with enough memory for
unquantised weights, so the self-hosted model is Qwen3-8B Q4_K_M (~5 GB) served
by Ollama through its OpenAI-compatible endpoint. On a 6 GB card Ollama splits
the model between GPU and CPU, so answer latency is seconds. The hosted-API
backend uses the same client class; it activates when `LLM_PROVIDER=api` plus an
API key are configured.

*Operational note:* the Ollama registry was unreachable from this host because
its IPv6 route is blackholed (`ollama pull` failed on R2 timeouts), so the GGUF
was downloaded from Hugging Face over IPv4 and imported with
`ollama create qwen3-8b-gguf -f Modelfile`. Same weights, no registry dependency.

**Default LLM moved to MiMo-V2.6-Flash via OpenCode Go.** The hosted backend
(`LLM_PROVIDER=api`) now points at the OpenCode Go OpenAI-compatible gateway
with `mimo-v2.6-flash`, replacing the local Qwen3-8B as the default the app
runs with. The gateway asks clients to identify themselves, so the OpenAI
client sends a stable session id and user agent through `LLM_API_HEADERS`
(`x-opencode-session`, `User-Agent`), and its model list lives at a
non-standard path (`LLM_MODELS_URL`). MiMo is a hybrid-thinking model whose
reasoning tokens count against the completion budget — measured: a 700-token
answer budget was entirely consumed by reasoning, returning an empty answer —
so requests disable thinking via `LLM_EXTRA_BODY`
(`chat_template_kwargs.enable_thinking=false`). Requires a Go API key from
the OpenCode Console; the Qwen3-8B Ollama path remains one
`LLM_PROVIDER=local` away.

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

**Optional Langfuse tracing.** `observability.py` wraps agent runs in a Langfuse
trace when keys are present and is a no-op otherwise, so the dependency never
blocks local development.

**uv + ruff + pytest.** Fast, single-tool workflow for deps, lint, and tests; the
test suite is hermetic (hashing embedder, stub services) and runs in ~6 s.

**Ablation-driven development.** Every retrieval change gets an
`eval/results/<config>.json` row. Current table on 187 questions: baseline
fixed-512 (0.588 R@5, 0.375 MRR), structure dense (0.872, 0.776), +sparse/hybrid
(0.866, 0.785), +rerank (0.893, 0.811), article granularity (0.840, 0.741).

**GPU memory budget.** On a 6 GB card the 8B model (~5 GB in 4-bit) and the
reranker cannot both be fully resident: Ollama splits the model 53/47 between GPU
and CPU. Loading everything at once makes the reranker hit CUDA OOM, which is why
`eval.agent_eval` sets `EMBED_DEVICE=cpu` and runs with a small `--limit`;
the retrieval ablation stops the LLM first.
