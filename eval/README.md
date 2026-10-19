# Evaluation

## Test set

`eval/testset.jsonl` — one JSON object per line:

| Field | Description |
|---|---|
| `question` | Question in Vietnamese |
| `article_id` | Gold article id(s), e.g. `"blld2019_d35"` or `["nd168_2024_d6", "luattatgt2024_d9"]` |
| `answer` | Short reference answer (may be empty for dataset rows without one) |
| `source` | Which source produced the row (`external dataset name`, `grounded`, `hard`) |

### Where the questions come from

1. **Filtered from public Vietnamese legal QA datasets** that carry article-level
   gold, mapped to our ids (`source` = dataset name). Selected sources:
   `thangvip/law-reading-comprehension-qa-filtered`,
   `Anakonkai/nlp-traffic-qa`, `adamwhite625/vietnam-legal-qa`,
   `hdv2709/Vietnamese_Legal_Traffic_Judge_Prediction_QA`,
   `thangvip/vietnamese-legal-qa`, `kwnn/traffic-law-qa-vietnamese`.
   Mappings, per-dataset findings, and blocked (gated) datasets are documented in
   `eval/zalo_raw/README-zalo.md`.
2. **Grounded questions generated from the corpus** by the configured LLM: one
   question per sampled article, gold = that article (`source` = `grounded`).
3. **Cross-article "hard" questions**: the LLM receives two articles and writes a
   comparison / multi-step question; gold = both articles (`source` = `hard`).

**Deviation from the plan.** `plans/week-2-eval-baseline.md` asks for ~100
questions filtered from "Zalo AI Legal Text Retrieval". That track ran in **2021**
(not 2023), and the 2021 corpus predates all four of our documents: 0 of its
3,196 gold passages map to 45/2019/QH14, 31/2024/QH15, 36/2024/QH15 or
168/2024/NĐ-CP. The Zalo mirror is kept in `eval/zalo_raw/` (gitignored) for
provenance, and the filtered-question budget is filled from the QA datasets
above instead.

### Building the test set

```bash
make testset                      # external + generated + hard, then validate
uv run python -m eval.build_testset --no-llm              # external + cached only
uv run python -m eval.build_testset --verify-only         # check gold ids exist
```

Generated questions are checkpointed to `eval/generated_questions.jsonl` after
each successful LLM call, so a slow or interrupted model call does not lose
work. Every row is validated against the ingested chunks: the question must end
with "?", the answer must be non-empty, and every gold `article_id` must exist in
the corpus (unknown ids are dropped and reported). The result is shuffled with a
fixed seed so reruns are reproducible.

## Metrics

- **Recall@5 / Recall@10** — fraction of questions where at least one gold article
  is in the top 5 / top 10 retrieved results (after rerank).
- **MRR** — mean of `1 / rank` of the first gold article in the ranked results
  (0 if absent).
- **Faithfulness** — a row is faithful when every article cited in the answer was
  actually retrieved *and* an LLM judge agrees the context supports the answer
  (`eval/judge.py`). A "not found" answer is faithful only when no evidence was
  retrieved.
- **Token F1** — SQuAD-style token F1 between the generated answer and the
  reference answer, over answered rows only.
- **Latency p50/p95** — end-to-end time per question.

## Running

```bash
make eval                                  # every config -> eval/results/*.json
make eval-table                            # print the ablation table
uv run python -m eval --config structure_hybrid_rerank --limit 20 --judge
```

`eval/results/*.json` hold the summary, per-document breakdown, and every row
(question, gold, retrieved ids, answer, latency) so results can be audited.

## Configs (ablation rows)

| Config | Collection | Search |
|---|---|---|
| `baseline_fixed512_dense` | `legal_docs_baseline` | dense |
| `structure_dense` | `legal_docs` | dense |
| `structure_hybrid` | `legal_docs` | dense+sparse RRF |
| `structure_hybrid_rerank` | `legal_docs` | hybrid + bge-reranker-v2-m3 |
| `article_hybrid_rerank` | `legal_docs_article` | hybrid + rerank |

The baseline collection is frozen after week 2 (see `plans/00-overview.md`).

## CI

The GitLab CI `eval` stage runs the whole pipeline with the deterministic hashing
embedder (no model download on shared runners) and publishes
`eval/results/` as artifacts. It proves the pipeline and metrics work
end to end; meaningful numbers require bge-m3 + bge-reranker-v2-m3 and are
produced locally with `make eval`. Results are in the README.
