"""Graph queries used by the agent, the MCP server, and the eval harness."""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


class GraphStore:
    def __init__(self, driver=None, uri=None, user=None, password=None) -> None:
        self.driver = driver
        self._uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self._user = user or os.environ.get("NEO4J_USER", "neo4j")
        self._password = password or os.environ.get("NEO4J_PASSWORD", "change-me")

    @property
    def ready(self) -> bool:
        return self.driver is not None

    def connect(self):
        if self.driver is None:
            from ingestion.graph import get_driver

            self.driver = get_driver(self._uri, self._user, self._password)
        return self.driver

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def verify(self) -> bool:
        try:
            self.connect().verify_connectivity()
            return True
        except Exception as exc:  # pragma: no cover - depends on Neo4j
            log.warning("Neo4j unavailable: %s", exc)
            return False

    def _run(self, query: str, **params) -> list[dict]:
        with self.connect().session() as session:
            return [dict(record) for record in session.run(query, **params)]

    # -- queries ---------------------------------------------------------
    def article(self, article_id: str) -> dict | None:
        rows = self._run(
            """
            MATCH (a:Article {article_id: $article_id})-[:HAS_ARTICLE]-(d:Document)
            RETURN a.article_id AS article_id, a.no AS no, a.title AS title,
                   d.doc_id AS doc_id, d.title AS doc_title
            """,
            article_id=article_id,
        )
        return rows[0] if rows else None

    def neighbors(self, article_id: str, limit: int = 10) -> list[dict]:
        return self._run(
            """
            MATCH (a:Article {article_id: $article_id})-[r:REFERENCES|DETAILS]-(b:Article)
            MATCH (d:Document {doc_id: b.doc_id})
            RETURN b.article_id AS article_id, b.no AS no, b.title AS title,
                   d.doc_id AS doc_id, d.title AS doc_title,
                   type(r) AS relation, startNode(r).article_id = $article_id AS outgoing
            LIMIT $limit
            """,
            article_id=article_id,
            limit=limit,
        )

    def references(self, article_id: str, limit: int = 20) -> list[dict]:
        return self._run(
            """
            MATCH (a:Article {article_id: $article_id})-[:REFERENCES]->(b:Article)
            MATCH (d:Document {doc_id: b.doc_id})
            RETURN b.article_id AS article_id, b.no AS no, b.title AS title,
                   d.doc_id AS doc_id, d.title AS doc_title
            LIMIT $limit
            """,
            article_id=article_id,
            limit=limit,
        )

    def details_from(self, doc_id: str, limit: int = 50) -> list[dict]:
        """Decree articles that implement law articles (DETAILS edges)."""
        return self._run(
            """
            MATCH (a:Article {doc_id: $doc_id})-[:DETAILS]->(b:Article)
            MATCH (d:Document {doc_id: b.doc_id})
            RETURN a.article_id AS source_id, a.no AS source_no,
                   b.article_id AS target_id, b.no AS target_no,
                   d.doc_id AS target_doc_id, d.title AS target_doc_title
            LIMIT $limit
            """,
            doc_id=doc_id,
            limit=limit,
        )

    def cross_document_pairs(self, limit: int = 50) -> list[dict]:
        return self._run(
            """
            MATCH (a:Article)-[r:REFERENCES|DETAILS]->(b:Article)
            WHERE a.doc_id <> b.doc_id
            RETURN a.article_id AS source_id, a.doc_id AS source_doc,
                   b.article_id AS target_id, b.doc_id AS target_doc, type(r) AS relation
            LIMIT $limit
            """,
            limit=limit,
        )

    def counts(self) -> dict:
        rows = self._run(
            """
            MATCH (n)
            WITH labels(n)[0] AS label, count(n) AS n
            RETURN label, n
            """
        )
        return {row["label"]: row["n"] for row in rows}
