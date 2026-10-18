"""Tests for cross-reference extraction (no Neo4j needed)."""

from __future__ import annotations

from ingestion.chunk import chunk_document
from ingestion.graph import (
    articles_from_chunks,
    extract_references,
    references_from_chunks,
)
from ingestion.registry import REGISTRY

SPECS = {spec.doc_id: spec for spec in REGISTRY.values()}


def test_explicit_law_reference_resolves_to_other_document():
    text = (
        "Hành vi vi phạm quy định tại Điều 9 Luật Trật tự, an toàn giao thông đường bộ bị xử phạt."
    )
    refs = extract_references(text, "nd168_2024", 6, SPECS)
    assert [reference.target_id for reference in refs] == ["luattatgt2024_d9"]
    assert refs[0].kind == "DETAILS"


def test_bare_reference_stays_in_source_document():
    text = "Thực hiện theo quy định tại Điều 35 và khoản 2 Điều 36."
    refs = extract_references(text, "blld2019", 1, SPECS)
    assert {reference.target_id for reference in refs} == {"blld2019_d35", "blld2019_d36"}
    assert {reference.kind for reference in refs} == {"REFERENCES"}


def test_same_document_phrase_keeps_source_document():
    text = "Chế độ quy định tại Điều 10 của Luật này được áp dụng."
    refs = extract_references(text, "luadd2024", 5, SPECS)
    assert [reference.target_id for reference in refs] == ["luadd2024_d10"]


def test_self_reference_is_skipped():
    text = "Điều 6 quy định về nội dung này."
    assert extract_references(text, "nd168_2024", 6, SPECS) == []


def test_deduplicates_repeated_mentions():
    text = "Điều 9 Luật Đất đai 2024 và Điều 9 Luật Đất đai 2024."
    refs = extract_references(text, "nd168_2024", 1, SPECS)
    assert len(refs) == 1


def test_articles_from_chunks_dedupes_per_article():
    text = (
        "Điều 1. Phạm vi\n"
        "Nội dung mở đầu đủ dài để trở thành một đoạn riêng biệt trong bài kiểm thử.\n"
        "1. Khoản một.\n"
        "2. Khoản hai.\n"
    )
    chunks = chunk_document(text, SPECS["blld2019"])
    articles = articles_from_chunks(chunks)
    assert len(chunks) == 3
    assert len(articles) == 1
    assert articles[0]["article_id"] == "blld2019_d1"
    assert articles[0]["title"] == "Phạm vi"


def test_references_from_chunks_uses_chunk_metadata():
    text = (
        "Điều 6. Xử phạt\n"
        "1. Phạt tiền đối với hành vi vi phạm quy định tại Điều 9 "
        "Luật Trật tự, an toàn giao thông đường bộ.\n"
    )
    chunks = chunk_document(text, SPECS["nd168_2024"])
    refs = references_from_chunks(chunks, SPECS)
    assert len(refs) == 1
    assert refs[0].source_id == "nd168_2024_d6"
    assert refs[0].target_id == "luattatgt2024_d9"
