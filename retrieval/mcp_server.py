"""MCP server exposing vector_search, graph_query, and compare_documents.

Run over stdio:  uv run python -m retrieval.mcp_server
Optionally over SSE: uv run python -m retrieval.mcp_server --transport sse --port 8765
"""

from __future__ import annotations

import argparse
import logging
import os

try:  # mcp >= 2 renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer
except ImportError:  # pragma: no cover - mcp 1.x fallback
    from mcp.server.fastmcp import FastMCP as MCPServer

from retrieval import tools as tool_impl

log = logging.getLogger(__name__)

mcp = MCPServer("legal-rag")
_retriever = None
_store = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from app.settings import get_settings
        from ingestion.embed import get_embedder
        from ingestion.load import get_client
        from retrieval.search import Retriever, get_reranker

        settings = get_settings()
        _retriever = Retriever(
            client=get_client(settings.qdrant_url, settings.qdrant_api_key),
            collection=settings.collection,
            embedder=get_embedder(settings.embed_backend, model_name=settings.embed_model),
            reranker=get_reranker(settings.rerank_backend, model_name=settings.rerank_model),
            dense_threshold=settings.dense_threshold,
            rerank_threshold=settings.rerank_threshold,
        )
    return _retriever


def get_store():
    global _store
    if _store is None:
        from retrieval.graph import GraphStore

        _store = GraphStore()
    return _store


def set_services(retriever=None, store=None) -> None:
    """Inject fakes in tests; also resets the cached services."""
    global _retriever, _store
    _retriever = retriever
    _store = store


@mcp.tool()
def vector_search(question: str, top_k: int = 5, mode: str = "hybrid_rerank") -> list[dict]:
    """Tìm kiếm ngữ nghĩa trong kho điều luật (dense + sparse + rerank)."""
    return tool_impl.vector_search(get_retriever(), question, top_k=top_k, mode=mode)


@mcp.tool()
def graph_query(article_id: str | None = None, doc_id: str | None = None, limit: int = 10) -> dict:
    """Tra cứu quan hệ tham chiếu giữa các điều luật trong đồ thị Neo4j."""
    return tool_impl.graph_query(get_store(), article_id=article_id, doc_id=doc_id, limit=limit)


@mcp.tool()
def compare_documents(question: str, doc_ids: list[str], top_k: int = 3) -> dict:
    """So sánh quy định giữa các văn bản (ví dụ nghị định với luật)."""
    return tool_impl.compare_documents(get_retriever(), question, doc_ids, top_k=top_k)


@mcp.tool()
def list_documents() -> list[dict]:
    """Danh sách bốn văn bản trong kho."""
    from ingestion.registry import REGISTRY

    return [
        {
            "doc_id": spec.doc_id,
            "title": spec.title,
            "number": spec.number,
            "effective_date": spec.effective_date,
            "doc_type": spec.doc_type,
        }
        for spec in REGISTRY.values()
    ]


def list_tool_names() -> list[str]:
    """Names of the tools exposed over MCP (also used by tests)."""
    import asyncio

    tools = asyncio.run(mcp.list_tools())
    return [getattr(tool, "name", str(tool)) for tool in tools]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retrieval.mcp_server", description=__doc__)
    parser.add_argument("--transport", choices=("stdio", "sse", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if os.environ.get("LEGAL_RAG_MCP_WARM", "1") != "0":
        try:
            get_store().verify()
        except Exception as exc:  # pragma: no cover
            log.warning("Neo4j not reachable: %s", exc)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return 0

    import uvicorn

    if hasattr(mcp, "settings"):  # mcp 1.x
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport=args.transport)
        return 0
    app = mcp.streamable_http_app() if args.transport == "streamable-http" else mcp.sse_app()
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
