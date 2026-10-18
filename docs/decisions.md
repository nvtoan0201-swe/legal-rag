# Technical decisions

**Structure-aware chunking (Chapter/Article/Clause).** Legal text has a fixed hierarchy and answers must cite articles. Fixed 512-token chunks are the baseline to beat.

**bge-m3 for dense + sparse.** One model produces both vectors, and sparse helps with exact legal terms and article numbers. Supports Vietnamese well.

**Qdrant hybrid search.** Native dense+sparse support with fusion, so no separate keyword engine is needed.

**Rerank with bge-reranker-v2-m3.** Same family as the embedder, multilingual; cheap way to raise precision in the top 5.

**Neo4j for cross-document references.** Decree 168/2024 refers to the laws, so multi-hop questions need explicit edges that vector search cannot give.

**LangGraph router with MCP tools.** The router picks vector, graph, or compare tool per question. Retrieval is exposed via MCP so it stays decoupled from the agent.

**API model vs. self-hosted Qwen3-8B.** Compare quality, p50/p95 latency, cost, and Vietnamese token count. vLLM if a GPU is available, otherwise Ollama 4-bit.

**FastAPI async + SSE.** Streaming answers improve perceived latency; background tasks handle slow document processing.

**uv + ruff + pytest.** Fast, single-tool workflow for deps, lint, and tests.

**Ablation-driven development.** Measure Recall@5, MRR, Faithfulness, and p95 after each retrieval change, starting from dense + 512-token baseline.
