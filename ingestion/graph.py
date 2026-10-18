"""Neo4j loader: Document/Article nodes plus HAS_ARTICLE / REFERENCES / DETAILS.

Cross-references come from a deliberately conservative regex pass
(plans/week-4-graph-agent-mcp.md, task 2): each "Điều N" mention is resolved to
the nearest explicit law/decree name in the surrounding window, falling back to
the source document for bare mentions ("Điều 5" inside the Labor Code means the
Labor Code). A decree article pointing at a law article becomes DETAILS, every
other edge is REFERENCES. Spot-check counts are logged after loading.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from ingestion.registry import DocSpec

log = logging.getLogger(__name__)

ARTICLE_MENTION_RE = re.compile(r"Điều\s+([0-9]{1,4})")
SAME_DOC_RE = re.compile(r"luật này|bộ luật này|nghị định này|văn bản này", re.IGNORECASE)

LAW_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("blld2019", re.compile(r"bộ luật lao động|bộ luật số\s*45/2019|bllđ", re.IGNORECASE)),
    (
        "luattatgt2024",
        re.compile(
            r"luật trật tự[,\s]+an toàn giao thông đường bộ|"
            r"luật trật tự và an toàn giao thông đường bộ|luật số\s*36/2024",
            re.IGNORECASE,
        ),
    ),
    ("luadd2024", re.compile(r"luật đất đai|luật số\s*31/2024", re.IGNORECASE)),
    ("nd168_2024", re.compile(r"nghị định\s*(?:số\s*)?168/2024", re.IGNORECASE)),
)

WINDOW = 240
DETAILS_SOURCE_TYPES = {"decree"}


@dataclass(frozen=True)
class Reference:
    source_id: str
    target_id: str
    kind: str  # "REFERENCES" | "DETAILS"


def _resolve_doc(window: str, position: int, source_doc_id: str) -> str:
    """Nearest explicit law name in the window, else the source document."""
    best: tuple[int, str] | None = None
    for doc_id, pattern in LAW_PATTERNS:
        for match in pattern.finditer(window):
            distance = abs(match.start() - position)
            if best is None or distance < best[0]:
                best = (distance, doc_id)
    if best is not None:
        return best[1]
    if SAME_DOC_RE.search(window):
        return source_doc_id
    return source_doc_id


def extract_references(
    text: str,
    source_doc_id: str,
    source_article_no: int,
    registry: dict[str, DocSpec],
    window: int = WINDOW,
) -> list[Reference]:
    """All cross-references mentioned in one chunk's text."""
    source_spec = registry.get(source_doc_id)
    kind = (
        "DETAILS"
        if source_spec is not None and source_spec.doc_type in DETAILS_SOURCE_TYPES
        else "REFERENCES"
    )
    seen: set[tuple[str, str]] = set()
    references: list[Reference] = []
    for match in ARTICLE_MENTION_RE.finditer(text):
        target_article_no = int(match.group(1))
        context = text[max(0, match.start() - window) : match.end() + window]
        target_doc_id = _resolve_doc(
            context, match.start() - max(0, match.start() - window), source_doc_id
        )
        if target_doc_id == source_doc_id and target_article_no == source_article_no:
            continue
        target_spec = registry.get(target_doc_id)
        if target_spec is None:
            continue
        key = (target_doc_id, str(target_article_no))
        if key in seen:
            continue
        seen.add(key)
        edge_kind = kind
        if (
            source_spec is not None
            and source_spec.doc_type == "decree"
            and target_spec.doc_type == "law"
        ):
            edge_kind = "DETAILS"
        elif (
            source_spec is not None
            and source_spec.doc_type == "decree"
            and target_spec.doc_type == "decree"
        ):
            edge_kind = "REFERENCES"
        references.append(
            Reference(
                source_id=source_spec.article_id(source_article_no) if source_spec else "",
                target_id=target_spec.article_id(target_article_no),
                kind=edge_kind,
            )
        )
    return references


def articles_from_chunks(chunks) -> list[dict]:
    """One row per article derived from structure chunks (for Article nodes)."""
    rows: dict[str, dict] = {}
    for chunk in chunks:
        if not getattr(chunk, "article_id", ""):
            continue
        rows.setdefault(
            chunk.article_id,
            {
                "article_id": chunk.article_id,
                "doc_id": chunk.doc_id,
                "no": chunk.article_no,
                "title": chunk.article_title,
                "chapter": chunk.chapter,
            },
        )
    return sorted(rows.values(), key=lambda row: (row["doc_id"], row["no"]))


def references_from_chunks(chunks, registry: dict[str, DocSpec]) -> list[Reference]:
    pairs: dict[tuple[str, str, str], Reference] = {}
    for chunk in chunks:
        if not getattr(chunk, "article_id", ""):
            continue
        for reference in extract_references(
            chunk.text, chunk.doc_id, chunk.article_no, registry, window=WINDOW
        ):
            pairs.setdefault((reference.source_id, reference.target_id, reference.kind), reference)
    return list(pairs.values())


def get_driver(uri: str | None = None, user: str | None = None, password: str | None = None):
    from neo4j import GraphDatabase

    uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = user or os.environ.get("NEO4J_USER", "neo4j")
    password = password or os.environ.get("NEO4J_PASSWORD", "change-me")
    return GraphDatabase.driver(uri, auth=(user, password))


def ensure_constraints(driver) -> None:
    with driver.session() as session:
        session.run(
            "CREATE CONSTRAINT document_doc_id IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT article_article_id IF NOT EXISTS "
            "FOR (a:Article) REQUIRE a.article_id IS UNIQUE"
        )


def load_graph(driver, specs, chunks_by_doc: dict[str, list], recreate: bool = False) -> dict:
    """Write documents, articles, and reference edges; return counters."""
    documents = [
        {
            "doc_id": spec.doc_id,
            "title": spec.title,
            "doc_type": spec.doc_type,
            "number": spec.number,
            "effective_date": spec.effective_date,
        }
        for spec in specs
    ]
    articles: list[dict] = []
    for spec in specs:
        articles.extend(articles_from_chunks(chunks_by_doc.get(spec.doc_id, [])))
    references = references_from_chunks(
        [chunk for spec in specs for chunk in chunks_by_doc.get(spec.doc_id, [])],
        {spec.doc_id: spec for spec in specs},
    )

    with driver.session() as session:
        if recreate:
            session.run("MATCH (n) DETACH DELETE n")
        session.run(
            """
            UNWIND $documents AS row
            MERGE (d:Document {doc_id: row.doc_id})
            SET d.title = row.title, d.doc_type = row.doc_type,
                d.number = row.number, d.effective_date = row.effective_date
            """,
            documents=documents,
        )
        session.run(
            """
            UNWIND $articles AS row
            MATCH (d:Document {doc_id: row.doc_id})
            MERGE (a:Article {article_id: row.article_id})
            SET a.doc_id = row.doc_id, a.no = row.no,
                a.title = row.title, a.chapter = row.chapter
            MERGE (d)-[:HAS_ARTICLE]->(a)
            """,
            articles=articles,
        )
        kinds = sorted({reference.kind for reference in references})
        for kind in kinds:
            payload = [
                {"source_id": reference.source_id, "target_id": reference.target_id}
                for reference in references
                if reference.kind == kind
            ]
            session.run(
                f"""
                UNWIND $references AS row
                MATCH (src:Article {{article_id: row.source_id}})
                MATCH (dst:Article {{article_id: row.target_id}})
                MERGE (src)-[:{kind}]->(dst)
                """,
                references=payload,
            )

        edge_counts = dict(
            session.run(
                "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS n ORDER BY n DESC"
            ).values()
        )
        node_counts = dict(
            session.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS n").values()
        )

    by_kind = {kind: sum(1 for reference in references if reference.kind == kind) for kind in kinds}
    skipped = 0
    with driver.session() as session:
        for kind in kinds:
            payload = [
                {"source_id": reference.source_id, "target_id": reference.target_id}
                for reference in references
                if reference.kind == kind
            ]
            if not payload:
                continue
            actual = session.run(
                f"""
                UNWIND $references AS row
                MATCH (src:Article {{article_id: row.source_id}})
                OPTIONAL MATCH (src)-[:{kind}]->(dst:Article {{article_id: row.target_id}})
                RETURN count(dst) AS matched
                """,
                references=payload,
            ).single()["matched"]
            skipped += len(payload) - int(actual)

    result = {
        "documents": len(documents),
        "articles": len(articles),
        "references_extracted": len(references),
        "references_by_kind": by_kind,
        "references_unresolved_targets": skipped,
        "nodes": node_counts,
        "edges": edge_counts,
    }
    log.info("graph loaded: %s", result)
    return result
