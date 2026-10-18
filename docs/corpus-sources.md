# Corpus provenance and how to re-obtain it

`data/raw/` is not committed (see `AGENTS.md`: "raw documents are not committed"),
so this page records exactly where the four source documents came from and how
they were converted. The per-file detail, including every failed URL, is kept in
`data/raw/SOURCES.md` after a run of the acquisition step.

| doc_id | Document | Number | Issued | Effective | Articles | File (UTF-8) | Bytes |
|---|---|---|---|---|---|---|---|
| `blld2019` | Bộ luật Lao động 2019 | 45/2019/QH14 | 2019-11-20 | 2021-01-01 | 220 (1..220, no gaps) | `data/raw/blld2019.txt` | 255,320 |
| `luadd2024` | Luật Đất đai 2024 | 31/2024/QH15 | 2024-01-18 | 2025-01-01 | 260 (1..260, no gaps) | `data/raw/luadd2024.txt` | 755,606 |
| `luattatgt2024` | Luật Trật tự, an toàn giao thông đường bộ 2024 | 36/2024/QH15 | 2024-06-27 | 2025-01-01 | 89 (1..89, no gaps) | `data/raw/luattatgt2024.txt` | 216,707 |
| `nd168_2024` | Nghị định 168/2024/NĐ-CP | 168/2024/NĐ-CP | 2024-12-26 | 2025-01-01 | 55 (1..55, no gaps) | `data/raw/nd168_2024.txt` | 337,832 |

## Where the text came from

* **blld2019, luattatgt2024, nd168_2024** — the Hugging Face dataset
  `th1nhng0/vietnamese-legal-documents` (config `content`), which mirrors the raw
  vbpl.vn full-text HTML for 170,824 documents. Row ids: `139264` (BLLĐ 2019),
  `170620` (Luật TTATGT 2024), `173920` (NĐ 168/2024).
* **luadd2024** — `https://luatvietnam.vn/dat-dai/luat-dat-dai-2024-so-31-2024-qh15-296638-d1.html`
  (container `div#noidung`), because the vbpl.vn mirror copy of that law has an
  empty body.

vbpl.vn itself returns HTTP 403 to non-browser clients and its front end is a
JavaScript SPA with no reachable search API, so the mirror was used for three of
the four documents. thuvienphapluat.vn is also bot-protected (403), and no
official PDF was obtainable: see the failure table in `data/raw/SOURCES.md`.

## How the text was produced

1. Download the source HTML/full text.
2. Render block elements (`p`, `div`, `td`, `li`, headings, `br`) as line breaks,
   join inline elements, collapse whitespace, and drop known UI-noise lines.
3. Article headings are guaranteed to start a line and look like
   `Điều 35. Quyền đơn phương chấm dứt hợp đồng lao động của người lao động`.
4. Verify with `(?m)^\s*Điều\s+(\d+)`: heading count, maximum number, and missing
   numbers in `1..max` (the ingestion manifest also cross-checks the maximum
   against `DocSpec.expected_articles`).

Re-running `make ingest` is idempotent; re-obtaining the raw files means repeating
steps 1–3 for each source above.

## Caveats

* Tables (sanction amounts in NĐ 168/2024, land-price tables) are flattened to
  one value per line. Retrieval is unaffected; column relationships are lost.
* The **appendices (Phụ lục) of Luật Đất đai 2024 are not included** — the source
  page advertises an attached file that is not part of the HTML. All 260 articles
  are present.
* The document header/signature block is preserved in the text; the chunker
  ignores lines before the first article.
