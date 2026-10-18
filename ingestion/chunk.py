"""Structure-aware chunking for Vietnamese legal documents.

The chunker splits raw text by Chapter (Chương) / Section (Mục) / Article (Điều)
and Clause (khoản, "1.") while keeping the legal hierarchy in metadata, mirroring
docs/design.md. Point (điểm, "a)") text stays with its clause. The fixed
512-token chunker produces the week-2 baseline for the ablation table.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from ingestion.registry import DocSpec

CHAPTER_RE = re.compile(r"^[ \t]*Chương[ \t]+([IVXLCDM]+|[0-9]+)[ \t]*$", re.MULTILINE)
SECTION_RE = re.compile(r"^[ \t]*Mục[ \t]+([0-9]+)[ \t]*[.:]?[ \t]*(.*)$", re.MULTILINE)
ARTICLE_RE = re.compile(r"^[ \t]*Điều[ \t]+([0-9]{1,4})[ \t]*[.:][ \t]*(.*)$", re.MULTILINE)
ARTICLE_LOOSE_RE = re.compile(r"^[ \t]*Điều[ \t]+([0-9]{1,4})[ \t]*[.:]?[ \t]*(.*)$", re.MULTILINE)
CLAUSE_RE = re.compile(r"^[ \t]*([0-9]{1,3})\.[ \t]+(.*)$", re.MULTILINE)
POINT_RE = re.compile(r"^[ \t]*([a-zđ])\)[ \t]*(.*)$", re.MULTILINE)

MIN_PREAMBLE_CHARS = 40


@dataclass
class Chunk:
    """One retrievable unit with everything needed for a citation."""

    chunk_id: str
    doc_id: str
    doc_title: str
    doc_number: str
    doc_type: str
    article_id: str
    article_no: int
    article_title: str
    chapter: str
    section: str
    clause_no: int | None
    text: str
    embed_text: str
    effective_date: str
    source_file: str
    granularity: str = "structure"

    def payload(self) -> dict:
        data = asdict(self)
        data.pop("embed_text", None)
        return data

    def citation(self) -> str:
        if self.article_no:
            citation = f"Điều {self.article_no}"
            if self.clause_no is not None:
                citation += f" khoản {self.clause_no}"
            citation += f", {self.doc_title}"
            return citation
        return self.doc_title


@dataclass
class _ArticleBlock:
    chapter: str
    section: str
    art_no: int
    art_title: str
    lines: list


def _article_re(text: str) -> re.Pattern[str]:
    """Use the strict heading pattern when the text has any; else the loose one."""
    return ARTICLE_RE if ARTICLE_RE.search(text) else ARTICLE_LOOSE_RE


def _embed_text(
    spec: DocSpec, chapter: str, section: str, art_no: int, art_title: str, text: str
) -> str:
    header = [spec.title]
    if chapter:
        header.append(chapter)
    if section:
        header.append(section)
    if art_no:
        heading = f"Điều {art_no}."
        if art_title:
            heading += f" {art_title}"
        header.append(heading)
    return "\n".join(header) + "\n" + text


def _join_lines(lines: list[str]) -> str:
    kept = [line.strip() for line in lines if line.strip()]
    return "\n".join(kept)


def _iter_article_blocks(text: str) -> list[_ArticleBlock]:
    """Split text into article blocks, tracking the enclosing chapter/section."""
    art_re = _article_re(text)
    blocks: list[_ArticleBlock] = []
    chapter = ""
    section = ""
    current: _ArticleBlock | None = None
    pending_title = False
    pending_chapter = False
    pending_section = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            if current is not None:
                current.lines.append("")
            continue

        match_chapter = CHAPTER_RE.match(line)
        if match_chapter:
            if current is not None:
                blocks.append(current)
                current = None
            chapter = f"Chương {match_chapter.group(1)}"
            section = ""
            pending_chapter = True
            pending_section = False
            continue

        match_section = SECTION_RE.match(line)
        if match_section:
            if current is not None:
                blocks.append(current)
                current = None
            section = f"Mục {match_section.group(1)}"
            if match_section.group(2).strip():
                section += f". {match_section.group(2).strip()}"
                pending_section = False
            else:
                pending_section = True
            pending_chapter = False
            continue

        if pending_chapter:
            chapter = f"{chapter}. {stripped}"
            pending_chapter = False
            continue
        if pending_section:
            section = f"{section}. {stripped}"
            pending_section = False
            continue

        match_article = art_re.match(line)
        if match_article:
            if current is not None:
                blocks.append(current)
            title = match_article.group(2).strip()
            current = _ArticleBlock(
                chapter=chapter,
                section=section,
                art_no=int(match_article.group(1)),
                art_title=title,
                lines=[],
            )
            pending_title = not title
            continue

        if current is None:
            continue
        if pending_title:
            current.art_title = stripped
            pending_title = False
            continue
        current.lines.append(line)

    if current is not None:
        blocks.append(current)
    return blocks


def _split_clauses(lines: list[str]) -> tuple[list[tuple[int, list[str]]], list[str]]:
    """Return (clause groups, preamble lines). Points stay with their clause."""
    groups: list[tuple[int, list[str]]] = []
    preamble: list[str] = []
    current_no: int | None = None
    current: list[str] = []
    for line in lines:
        match = CLAUSE_RE.match(line)
        if match:
            if current_no is not None:
                groups.append((current_no, current))
            elif _join_lines(current):
                preamble = current
            current_no = int(match.group(1))
            current = [line.strip()]
            continue
        current.append(line)
    if current_no is not None:
        groups.append((current_no, current))
    elif _join_lines(current):
        preamble = current
    return groups, preamble


def _make_chunk(
    spec: DocSpec,
    block: _ArticleBlock,
    clause_no: int | None,
    text: str,
    source_file: str,
    granularity: str = "structure",
) -> Chunk:
    suffix = ""
    if granularity == "structure" and clause_no is None:
        suffix = "" if text == _join_lines(block.lines) else "_intro"
    elif clause_no is not None:
        suffix = f"_k{clause_no}"
    chunk_id = f"{spec.doc_id}_d{block.art_no}{suffix}"
    return Chunk(
        chunk_id=chunk_id,
        doc_id=spec.doc_id,
        doc_title=spec.title,
        doc_number=spec.number,
        doc_type=spec.doc_type,
        article_id=spec.article_id(block.art_no),
        article_no=block.art_no,
        article_title=block.art_title,
        chapter=block.chapter,
        section=block.section,
        clause_no=clause_no,
        text=text,
        embed_text=_embed_text(
            spec, block.chapter, block.section, block.art_no, block.art_title, text
        ),
        effective_date=spec.effective_date,
        source_file=source_file,
        granularity=granularity,
    )


def _chunks_for_block(block: _ArticleBlock, spec: DocSpec, source_file: str) -> list[Chunk]:
    groups, preamble = _split_clauses(block.lines)
    if not groups:
        text = _join_lines(block.lines)
        if not text and not block.art_title:
            return []
        return [_make_chunk(spec, block, None, text, source_file)]

    chunks: list[Chunk] = []
    preamble_text = _join_lines(preamble)
    if len(preamble_text) >= MIN_PREAMBLE_CHARS:
        chunks.append(_make_chunk(spec, block, None, preamble_text, source_file))
    for index, (clause_no, clause_lines) in enumerate(groups):
        text = _join_lines(clause_lines)
        if index == 0 and preamble_text and len(preamble_text) < MIN_PREAMBLE_CHARS:
            # Short lead-in belongs to the first clause; never drop source text.
            text = preamble_text + "\n" + text
        chunks.append(_make_chunk(spec, block, clause_no, text, source_file))
    return chunks


def chunk_document(text: str, spec: DocSpec, source_file: str = "") -> list[Chunk]:
    """Structure-aware chunks for one document (Chapter/Article/Clause metadata)."""
    chunks: list[Chunk] = []
    for block in _iter_article_blocks(text):
        chunks.extend(_chunks_for_block(block, spec, source_file))
    return chunks


def article_chunks(chunks: list[Chunk], spec: DocSpec) -> list[Chunk]:
    """Merge clause chunks into one chunk per article (granularity experiment)."""
    grouped: dict[int, list[Chunk]] = {}
    order: list[int] = []
    for chunk in chunks:
        if chunk.article_no not in grouped:
            grouped[chunk.article_no] = []
            order.append(chunk.article_no)
        grouped[chunk.article_no].append(chunk)

    merged: list[Chunk] = []
    for article_no in order:
        parts = grouped[article_no]
        first = parts[0]
        text = _join_lines([part.text for part in parts])
        merged.append(
            Chunk(
                chunk_id=f"{spec.doc_id}_d{article_no}",
                doc_id=first.doc_id,
                doc_title=first.doc_title,
                doc_number=first.doc_number,
                doc_type=first.doc_type,
                article_id=first.article_id,
                article_no=article_no,
                article_title=first.article_title,
                chapter=first.chapter,
                section=first.section,
                clause_no=None,
                text=text,
                embed_text=_embed_text(
                    spec, first.chapter, first.section, article_no, first.article_title, text
                ),
                effective_date=first.effective_date,
                source_file=first.source_file,
                granularity="article",
            )
        )
    return merged


def fixed_token_chunks(
    text: str,
    spec: DocSpec,
    source_file: str = "",
    size: int = 512,
    overlap: int = 64,
    granularity: str = "fixed512",
) -> list[Chunk]:
    """Week-2 baseline: fixed-size whitespace-token windows.

    Whitespace tokens are a deterministic proxy for model tokens so the baseline
    stays reproducible without loading a tokenizer. Each chunk keeps the nearest
    preceding article heading so Recall@5 can still be computed against gold
    article ids.
    """
    art_re = _article_re(text)
    tokens: list[tuple[str, int, str]] = []
    art_no = 0
    art_title = ""
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        match_article = art_re.match(line)
        if match_article:
            art_no = int(match_article.group(1))
            art_title = match_article.group(2).strip()
        for token in stripped.split():
            tokens.append((token, art_no, art_title))

    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + size]
        if not window:
            break
        # Drop only a short trailing window; a full-size window is always kept.
        if start > 0 and len(window) < min(size, 32):
            break
        _token, window_art, window_title = window[0]
        text_window = " ".join(token for token, _no, _title in window)
        chunk = Chunk(
            chunk_id=f"{spec.doc_id}_fixed{start:05d}",
            doc_id=spec.doc_id,
            doc_title=spec.title,
            doc_number=spec.number,
            doc_type=spec.doc_type,
            article_id=spec.article_id(window_art) if window_art else "",
            article_no=window_art,
            article_title=window_title,
            chapter="",
            section="",
            clause_no=None,
            text=text_window,
            embed_text=text_window,
            effective_date=spec.effective_date,
            source_file=source_file,
            granularity=granularity,
        )
        chunks.append(chunk)
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict:
    """Counts used by the parse sanity check and the ingestion manifest."""
    articles = {chunk.article_id for chunk in chunks if chunk.article_id}
    return {
        "chunks": len(chunks),
        "articles": len(articles),
        "clause_chunks": sum(1 for chunk in chunks if chunk.clause_no is not None),
        "chars": sum(len(chunk.text) for chunk in chunks),
    }
