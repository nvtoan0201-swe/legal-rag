"""Registry of the four source documents: stable ids, metadata, file matching.

The doc_id values here become the prefix of every article_id ("blld2019_d35"),
so they must stay stable: the evaluation test set and the graph loader both
depend on them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


@dataclass(frozen=True)
class DocSpec:
    """Metadata for one source document."""

    doc_id: str
    title: str
    short_title: str
    doc_type: str  # "law" | "decree"
    number: str
    issued_date: str
    effective_date: str
    aliases: tuple[str, ...]
    expected_articles: int | None = None

    def article_id(self, article_no: int) -> str:
        return f"{self.doc_id}_d{article_no}"


REGISTRY: dict[str, DocSpec] = {
    spec.doc_id: spec
    for spec in (
        DocSpec(
            doc_id="blld2019",
            title="Bộ luật Lao động 2019",
            short_title="BLLĐ 2019",
            doc_type="law",
            number="45/2019/QH14",
            issued_date="2019-11-20",
            effective_date="2021-01-01",
            aliases=("blld2019", "blld-2019", "bo-luat-lao-dong", "lao-dong-2019", "45-2019"),
            expected_articles=220,
        ),
        DocSpec(
            doc_id="luadd2024",
            title="Luật Đất đai 2024",
            short_title="Luật Đất đai 2024",
            doc_type="law",
            number="31/2024/QH15",
            issued_date="2024-01-18",
            effective_date="2025-01-01",
            aliases=("luadd2024", "luat-dat-dai", "dat-dai-2024", "31-2024", "datdai2024"),
            expected_articles=260,
        ),
        DocSpec(
            doc_id="luattatgt2024",
            title="Luật Trật tự, an toàn giao thông đường bộ 2024",
            short_title="Luật TTATGT 2024",
            doc_type="law",
            number="36/2024/QH15",
            issued_date="2024-06-27",
            effective_date="2025-01-01",
            aliases=(
                "luattatgt2024",
                "trat-tu-an-toan-giao-thong",
                "an-toan-giao-thong",
                "36-2024",
                "ttatgt",
            ),
            expected_articles=89,
        ),
        DocSpec(
            doc_id="nd168_2024",
            title="Nghị định 168/2024/NĐ-CP",
            short_title="NĐ 168/2024",
            doc_type="decree",
            number="168/2024/NĐ-CP",
            issued_date="2024-12-26",
            effective_date="2025-01-01",
            aliases=("nd168", "168-2024", "nghi-dinh-168", "nd-168"),
            expected_articles=55,
        ),
    )
}


def match_file(path: Path) -> DocSpec | None:
    """Return the DocSpec whose aliases appear in the file name, if any."""
    name = path.name.lower()
    for spec in REGISTRY.values():
        if any(alias in name for alias in spec.aliases):
            return spec
    return None


def source_files(raw_dir: Path = RAW_DIR) -> list[tuple[DocSpec, Path]]:
    """All recognised raw files, sorted by doc_id."""
    if not raw_dir.exists():
        return []
    found: list[tuple[DocSpec, Path]] = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        spec = match_file(path)
        if spec is not None:
            found.append((spec, path))
    found.sort(key=lambda pair: pair[0].doc_id)
    return found
