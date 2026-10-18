"""Tests for the structure-aware chunker and the fixed-size baseline."""

from __future__ import annotations

from ingestion.chunk import article_chunks, chunk_document, chunk_stats, fixed_token_chunks
from ingestion.registry import REGISTRY

SPEC = REGISTRY["blld2019"]

FIXTURE = """BỘ LUẬT
LAO ĐỘNG

Chương I
NHỮNG QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
Bộ luật này quy định tiêu chuẩn lao động, quyền và nghĩa vụ của người lao động.
1. Bộ luật này áp dụng đối với người lao động.
2. Bộ luật này áp dụng đối với người sử dụng lao động.
a) Doanh nghiệp;
b) Hợp tác xã.

Chương II
HỢP ĐỒNG LAO ĐỘNG

Mục 1
GIAO KẾT HỢP ĐỒNG LAO ĐỘNG

Điều 2. Hợp đồng lao động
Hợp đồng lao động là sự thỏa thuận giữa người lao động và người sử dụng lao động.

Điều 3. Nguyên tắc giao kết hợp đồng lao động
Nguyên tắc giao kết hợp đồng lao động được quy định như sau và đủ dài để tách đoạn.
1. Tự nguyện.
2. Bình đẳng.

Điều 4. Nghĩa vụ
Ngắn.
1. Khoản một.
"""


def by_id(chunks):
    return {chunk.chunk_id: chunk for chunk in chunks}


def test_chunk_ids_follow_design_schema():
    chunks = chunk_document(FIXTURE, SPEC, "data/raw/blld2019.pdf")
    ids = [chunk.chunk_id for chunk in chunks]
    assert ids == [
        "blld2019_d1_intro",
        "blld2019_d1_k1",
        "blld2019_d1_k2",
        "blld2019_d2",
        "blld2019_d3_intro",
        "blld2019_d3_k1",
        "blld2019_d3_k2",
        "blld2019_d4_k1",
    ]


def test_metadata_fields():
    chunks = by_id(chunk_document(FIXTURE, SPEC, "data/raw/blld2019.pdf"))
    first = chunks["blld2019_d1_k1"]
    assert first.doc_id == "blld2019"
    assert first.article_id == "blld2019_d1"
    assert first.article_no == 1
    assert first.article_title == "Phạm vi điều chỉnh"
    assert first.clause_no == 1
    assert first.chapter == "Chương I. NHỮNG QUY ĐỊNH CHUNG"
    assert first.effective_date == "2021-01-01"
    assert first.source_file == "data/raw/blld2019.pdf"

    third = chunks["blld2019_d3_k1"]
    assert third.chapter == "Chương II. HỢP ĐỒNG LAO ĐỘNG"
    assert third.section == "Mục 1. GIAO KẾT HỢP ĐỒNG LAO ĐỘNG"


def test_points_stay_with_their_clause():
    chunks = by_id(chunk_document(FIXTURE, SPEC))
    clause2 = chunks["blld2019_d1_k2"].text
    assert "người sử dụng lao động" in clause2
    assert "a) Doanh nghiệp;" in clause2
    assert "b) Hợp tác xã." in clause2


def test_short_preamble_is_merged_not_dropped():
    chunks = by_id(chunk_document(FIXTURE, SPEC))
    fourth = chunks["blld2019_d4_k1"]
    assert fourth.text.startswith("Ngắn.")
    assert "Khoản một." in fourth.text
    assert "blld2019_d4_intro" not in chunks


def test_payload_omits_embed_text():
    payload = chunk_document(FIXTURE, SPEC)[0].payload()
    assert "embed_text" not in payload
    assert payload["chunk_id"] == "blld2019_d1_intro"
    assert payload["text"].startswith("Bộ luật này quy định")


def test_article_granularity_merges_clauses():
    merged = article_chunks(chunk_document(FIXTURE, SPEC), SPEC)
    ids = [chunk.chunk_id for chunk in merged]
    assert ids == ["blld2019_d1", "blld2019_d2", "blld2019_d3", "blld2019_d4"]
    assert merged[0].granularity == "article"
    assert merged[0].clause_no is None
    assert "a) Doanh nghiệp;" in merged[0].text
    assert "Điều 1." in merged[0].embed_text


def test_fixed_baseline_windows_and_article_mapping():
    tokens = [f"t{i}" for i in range(50)]
    chunks = fixed_token_chunks("Điều 7. Tiêu đề\n" + " ".join(tokens), SPEC, size=20, overlap=5)
    assert [chunk.granularity for chunk in chunks] == ["fixed512"] * len(chunks)
    windows = [chunk.text.split() for chunk in chunks]
    # 54 tokens total (4 heading tokens + 50), step 15 -> windows overlap by 5
    assert windows[0] == ["Điều", "7.", "Tiêu", "đề", *tokens[:16]]
    assert windows[1][:5] == windows[0][-5:]
    assert all(len(window) == 20 for window in windows[:-1])
    assert all(chunk.article_no == 7 for chunk in chunks)
    assert all(chunk.article_id == "blld2019_d7" for chunk in chunks)


def test_chunk_stats_counts_articles():
    stats = chunk_stats(chunk_document(FIXTURE, SPEC))
    assert stats["articles"] == 4
    assert stats["chunks"] == 8
    assert stats["clause_chunks"] == 5
