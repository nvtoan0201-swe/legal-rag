"""Tests for text normalisation and the parse sanity check."""

from __future__ import annotations

from ingestion.parse import article_stats, normalize_text, parse_document
from ingestion.registry import REGISTRY

SPEC = REGISTRY["blld2019"]


def test_normalize_strips_markdown_artifacts():
    raw = "## Điều 1. **Phạm vi**\n\n\n<!-- image -->\nÐiều 2. Tiêu đề\n"
    text = normalize_text(raw)
    assert text.startswith("Điều 1. Phạm vi")
    assert "Ðiều" not in text
    assert text.count("Điều") == 2
    assert "#" not in text
    assert "**" not in text


def test_article_stats():
    text = "Điều 1. A\nĐiều 2. B\nĐiều 10. C\n"
    assert article_stats(text) == (10, 3)
    assert article_stats("không có điều nào") == (0, 0)


def test_parse_document_txt_and_warns_on_article_mismatch(tmp_path):
    path = tmp_path / "blld2019.txt"
    path.write_text("Điều 1. A\nnội dung\n", encoding="utf-8")
    parsed = parse_document(path, SPEC)
    assert parsed.parser == "text"
    assert parsed.max_article == 1
    assert any("expected max article 220" in warning for warning in parsed.warnings)
    assert parsed.meta()["chars"] > 0
