# GoodMem on BEAM-10M: GPT-6.1 Sol results

**GoodMem + GPT-6.1 Sol with medium reasoning scored 64.61% on all 200 questions
across ten complete BEAM-10M histories.** Evaluation completed on 7 November 2026.
The agent configuration stayed fixed throughout, with reviews every 25 questions.

| Metric | Result |
| --- | ---: |
| Primary score | **64.61%** |
| Event-F1-adjusted score | **62.56%** |
| Primary score, 95% conversation-bootstrap interval | 61.08–68.16% |
| Questions graded | 200/200 |

The primary metric averages nugget scores for nine abilities and normalized
Kendall tau for event ordering. The secondary metric multiplies event-ordering
scores by event F1, accounting for missing and extra events. These are mean
scores, not percentages of wholly correct answers. See [methodology](docs/methodology.md).

The interval uses 10,000 resamples of entire conversations, seed 20261107.
It measures variation across these ten histories, not judge disagreement or
repeated model sampling. The harness was developed on BEAM; transfer to a fresh
benchmark has not been measured.

## Per ability

Each score averages 20 questions.

| Ability | Score |
| --- | ---: |
| Abstention | 90.00% |
| Contradiction resolution | 51.25% |
| Event ordering, primary | 32.64% |
| Information extraction | 75.00% |
| Instruction following | 77.50% |
| Knowledge update | 78.75% |
| Multi-session reasoning | 44.50% |
| Preference following | 72.50% |
| Summarization | 68.94% |
| Temporal reasoning | 55.00% |
| Event ordering, event-F1-adjusted | 12.17% |

Event reconstruction and multi-session reasoning are the weakest areas. Batch
reviews found missing or extra events, incorrect episode/date selection, missed
conflicting claims, and answers that asked for clarification without providing
the requested information. These observations do not separate retrieval failures
from reader decisions or judge errors.

## Configuration

The full pinned release contains **208,696 messages and 108,051,936 content tokens**
under `o200k_base`. Individual conversations measure 9.66–12.93 million tokens.
All canonical chat messages, including assistant turns, were indexed into
**10,298 memory blocks**. Histories were not truncated and questions were not
subsampled. Separate profiles and generation plans were not indexed.

| Setting | Value |
| --- | --- |
| Reader | GPT-6.1 Sol, medium reasoning, Responses API |
| Judge | GPT-4.1-mini, temperature 0, upstream BEAM prompts |
| Embeddings / reranker | Voyage 4 Large / Voyage Rerank 3 |
| Memory block size | Approximately 48,000 characters |
| Chunk size / overlap | 3,200 / 400 characters |
| Dense candidates → reranked chunks per search | 128 → 20 |
| HNSW | `efSearch=256`, relaxed iterative scanning, `maxScanTuples=100000` |
| Tool-round limit | 32 |
| Active-context limit | 2,800,000 characters |
| Output limit per reader call | 16,000 tokens |
| Original-message read limit | 31 messages / 24,000 characters |
| Submission checks | Three distinct queries, cited factual checklist, fresh search/read after drafting |

GoodMem ran locally at `https://localhost:8080`; model inference used hosted
providers through its proxy. The server's development build and local changes
are recorded in the [protocol](results/gpt-6.1-sol-medium-v3/protocol.json).
Actual response model identifiers were `gpt-6.1-sol` and
`gpt-4.1-mini-2025-04-14`.

The agent can search, read source messages, and calculate date intervals.
Missing source headers are restored only by exact matching inside a parent
block already returned by GoodMem. The Responses loop retains opaque reasoning
between calls without logging or exporting it. The reader receives no ability
label, rubric, reference answer, or question-specific instructions.

## Retrieval and usage

| Recorded behavior | Result |
| --- | ---: |
| Successful searches | 1,233 |
| Mean searches per question | 6.165 |
| Maximum searches for one question | 23 |
| Questions using multiple searches | 200/200 |
| Original-message reads | 596 |
| Questions using original-message reads | 192/200 |
| Reader calls | 2,252 |
| Questions exhausting the tool-round budget | 0 |
| Context evictions | 0 |
| Reader input tokens | 253,812,210 |
| Cached input tokens | 225,085,232 (88.68%) |
| Reader output tokens | 302,750 |

All 200 answers passed the stopping checks. Four invalid tool requests were
corrected within the loop: one oversized read and three citation/checklist errors.
Output totals include 34,360 reasoning tokens. Usage covers this run's successful
answer trajectories and excludes judging and development work; it is not a
billing estimate. Internal HTTP retries were not counted.

## Artifacts and verification

All 200 answers and judgments completed without question-level errors. Four
judge explanations required normalizing invalid backslashes; their numeric scores
and raw outputs were preserved. Every event-ordering tau was defined.

After evaluation, all 10,298 memories were `COMPLETED`. Deterministic IDs and
stored ingestion metadata digests matched the prepared blocks, whose hashes
matched the index manifest. The answer-generation package matches commit
`e2935c2d94b7ef997a8d19f92583d95bb90fcd6e` and the frozen protocol hash.

| Artifact | Contents |
| --- | --- |
| [Summary](results/gpt-6.1-sol-medium-v3/summary.json) | Scores and confidence interval |
| [Protocol](results/gpt-6.1-sol-medium-v3/protocol.json) | Settings, model registrations, prompt and source hashes |
| [Outcomes](results/gpt-6.1-sol-medium-v3/outcomes.jsonl.gz) | All answers, rubrics, and raw judgments |
| [Retrieval traces](results/gpt-6.1-sol-medium-v3/retrieval-traces.jsonl.gz) | Queries, evidence identifiers, tool decisions, and usage |
| [Operations](results/gpt-6.1-sol-medium-v3/operations.json) | Model identities, tool errors, normalization, and usage scope |
| [Procedure audit](results/gpt-6.1-sol-medium-v3/procedure-audit.json) | Stopping and tool-protocol checks |
| [Index verification](results/gpt-6.1-sol-medium-v3/index-verification.json) | Readiness and stored ingestion metadata checks |
| [Sol run record](docs/gpt61-experiment.md) | All eight batch scores and reproduction steps |

Recompute aggregates and check the recorded procedure without an API key:

```bash
uv sync --locked
uv run python tools/verify_results.py results/gpt-6.1-sol-medium-v3
uv run python tools/audit_agent_trace.py results/gpt-6.1-sol-medium-v3
```

The procedure audit checks model settings, citation IDs, submission checks, final
answers, and reported usage. It does not check whether each citation supports
the associated claim. Full source excerpts and credentials are kept out of Git.

Dataset-derived artifacts are **CC BY-SA 4.0**, attributed to BEAM by Tavakoli
et al. See [third-party notices](THIRD_PARTY_NOTICES.md) and the
[comparison with other published BEAM-10M results](docs/published-beam-results.md).
