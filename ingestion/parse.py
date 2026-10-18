"""Parse raw documents to plain text.

Docling is the primary parser (layout aware, handles PDF/DOCX/HTML); PyMuPDF is
the fallback for PDFs, and a plain read covers .txt. Every parser result goes
through normalize_text() so the chunker can rely on one text shape, and the
article count is checked against the expected number for the document
(see plans/00-overview.md, "parse sanity check").
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from ingestion.registry import DocSpec

log = logging.getLogger(__name__)

ARTICLE_HEADING_RE = re.compile(r"(?m)^[ \t]*Điều[ \t]+([0-9]{1,4})[ \t]*[.:]")

_MD_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_MD_HEADING = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]*")
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MD_ITALIC = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_RULE = re.compile(r"(?m)^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_MULTI_BLANK = re.compile(r"\n{3,}")


@dataclass
class ParsedDocument:
    doc_id: str
    source_file: str
    parser: str
    text: str
    max_article: int
    heading_count: int
    warnings: list[str] = field(default_factory=list)

    def meta(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "parser": self.parser,
            "max_article": self.max_article,
            "heading_count": self.heading_count,
            "chars": len(self.text),
            "warnings": self.warnings,
        }


def normalize_text(text: str) -> str:
    """Normalise Unicode, strip markdown artefacts, and tidy blank lines."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    # Old Vietnamese encodings use ETH (Ð/ð) for D with stroke (Đ/đ).
    text = text.replace("\u00d0", "Đ").replace("\u00f0", "đ")
    text = _MD_COMMENT.sub("", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_ITALIC.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_RULE.sub("", text)
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if lines and lines[-1].strip():
                lines.append("")
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def article_stats(text: str) -> tuple[int, int]:
    """Return (max article number, number of article headings)."""
    numbers = [int(n) for n in ARTICLE_HEADING_RE.findall(text)]
    if not numbers:
        return 0, 0
    return max(numbers), len(numbers)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_with_docling(path: Path) -> str:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(path))
    document = result.document
    try:
        return document.export_to_markdown()
    except Exception:  # pragma: no cover - depends on docling internals
        return document.export_to_text()


def _parse_pdf_pymupdf(path: Path) -> str:
    import fitz

    with fitz.open(path) as pdf:
        return "\n".join(page.get_text("text") for page in pdf)


def _parse_html(path: Path) -> str:
    raw = _read_text(path)
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return re.sub(r"<[^>]+>", "\n", raw)
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    return soup.get_text("\n")


def parse_document(path: Path, spec: DocSpec, prefer_docling: bool = True) -> ParsedDocument:
    """Parse one raw file to normalised text and sanity-check its articles."""
    warnings: list[str] = []
    suffix = path.suffix.lower()
    parser = "unknown"
    raw_text = ""

    if suffix in {".txt", ".md"}:
        raw_text, parser = _read_text(path), "text"
    elif suffix in {".html", ".htm"}:
        try:
            raw_text, parser = _parse_with_docling(path), "docling"
        except Exception as exc:
            warnings.append(f"docling failed on HTML: {exc}")
            raw_text, parser = _parse_html(path), "html"
    else:
        order = ["docling", "pymupdf"] if prefer_docling else ["pymupdf", "docling"]
        for candidate in order:
            try:
                raw_text = (
                    _parse_with_docling(path)
                    if candidate == "docling"
                    else _parse_pdf_pymupdf(path)
                )
                parser = candidate
                break
            except Exception as exc:
                warnings.append(f"{candidate} failed: {exc}")

    if not raw_text.strip():
        raise ValueError(f"no text extracted from {path} (parser={parser})")

    text = normalize_text(raw_text)
    max_article, heading_count = article_stats(text)
    if spec.expected_articles is not None and max_article != spec.expected_articles:
        warnings.append(
            f"expected max article {spec.expected_articles}, found {max_article} "
            f"({heading_count} headings)"
        )
    if heading_count == 0:
        warnings.append("no 'Điều N.' headings found - check the parser output")

    return ParsedDocument(
        doc_id=spec.doc_id,
        source_file=str(path),
        parser=parser,
        text=text,
        max_article=max_article,
        heading_count=heading_count,
        warnings=warnings,
    )
