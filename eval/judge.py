"""LLM-as-judge faithfulness scoring (see eval/README.md).

A row is faithful when (a) every article cited in the answer was actually
retrieved, and (b) an LLM judge agrees that the context supports the answer. A
"not found" answer counts as faithful only when no evidence was retrieved.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from llm.base import system, user
from llm.factory import get_llm

log = logging.getLogger("eval.judge")

CITATION_RE = re.compile(r"Điều\s+([0-9]{1,4})")
NOT_FOUND_MARKERS = (
    "không tìm thấy",
    "không có đủ căn cứ",
    "không có thông tin",
    "không đủ căn cứ",
    "not found",
)

JUDGE_SYSTEM = (
    "Bạn là giám khảo cho hệ thống hỏi đáp pháp luật Việt Nam. "
    "Được cung cấp câu hỏi, ngữ cảnh (các điều luật đã truy xuất) và câu trả lời. "
    "Hãy đánh giá xem MỌI khẳng định trong câu trả lời có được ngữ cảnh hỗ trợ hay không. "
    "Chỉ trả lời đúng một từ: CÓ nếu mọi khẳng định đều được hỗ trợ, "
    "KHÔNG nếu có khẳng định không được hỗ trợ."
)


def is_not_found(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in NOT_FOUND_MARKERS)


def extract_cited_articles(text: str) -> list[int]:
    seen: dict[int, None] = {}
    for match in CITATION_RE.finditer(text or ""):
        seen.setdefault(int(match.group(1)), None)
    return list(seen)


def citation_valid(answer: str, hits: Sequence) -> bool:
    """Every cited article number must appear in the retrieved hits."""
    cited = extract_cited_articles(answer)
    if not cited:
        return False
    retrieved = {getattr(hit, "article_no", None) for hit in hits}
    return all(article in retrieved for article in cited)


def parse_verdict(response: str) -> bool:
    text = (response or "").strip().upper()
    if text.startswith("CÓ") or text.startswith("YES") or text.startswith("TRUE"):
        return True
    if text.startswith("KHÔNG") or text.startswith("NO") or text.startswith("FALSE"):
        return False
    log.warning("unparseable judge verdict: %r", response[:80])
    return False


class LLMJudge:
    def __init__(self, llm=None) -> None:
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def __call__(self, question: str, answer: str, hits: Sequence) -> bool:
        if is_not_found(answer):
            return len(hits) == 0
        if not citation_valid(answer, hits):
            return False
        context = "\n\n".join(f"[{hit.citation}] {hit.text}" for hit in hits)
        prompt = (
            f"Câu hỏi: {question}\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu trả lời: {answer}\n\n"
            "Mọi khẳng định trong câu trả lời có được ngữ cảnh hỗ trợ không? (CÓ/KHÔNG)"
        )
        response = self.llm.chat(
            [system(JUDGE_SYSTEM), user(prompt)], temperature=0.0, max_tokens=8
        )
        return parse_verdict(response)


def build_judge(llm=None) -> LLMJudge:
    return LLMJudge(llm=llm)
