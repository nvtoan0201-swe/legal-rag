# Evaluation

## Test set

`testset.jsonl` (~200 questions), one JSON object per line:

| Field | Description |
|---|---|
| `question` | Question in Vietnamese |
| `article_id` | Gold article(s), e.g. `blld2019_d35`; string or list |
| `answer` | Short reference answer |

Sources: ~100 filtered from Zalo AI Legal Text Retrieval (mostly Labor Code), ~70 LLM-generated from 2024 laws and manually reviewed, ~30 hand-written hard questions (comparison, multi-step).

See `testset.sample.jsonl` for the format.

## Metrics

- **Recall@5** — fraction of questions where at least one gold article is in the top 5 retrieved results (after rerank).
- **MRR** — mean of `1 / rank` of the first gold article in the ranked results (0 if absent).
- **Faithfulness** — fraction of claims in the generated answer supported by the retrieved context, judged by an LLM; also check that cited articles are in the retrieved set.
- **Latency p95** — 95th percentile end-to-end time per question.

## Run

```bash
make eval   # TODO
```
