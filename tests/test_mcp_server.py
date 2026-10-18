"""The MCP server must expose the tools the agent relies on."""

from __future__ import annotations

from retrieval.mcp_server import list_tool_names


def test_mcp_exposes_required_tools():
    names = set(list_tool_names())
    assert {"vector_search", "graph_query", "compare_documents", "list_documents"} <= names
