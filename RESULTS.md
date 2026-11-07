# Measured BEAM-10M results

**GoodMem + GPT-6.1 Sol with medium reasoning scored 64.61% across all 200
questions and ten complete histories.** The same v3 harness was used throughout,
with a review after every 25 questions. Evaluation completed on 7 November 2026.

| Reader and harness | Questions graded | Primary score | 95% conversation bootstrap CI | Event-F1-adjusted score |
| --- | ---: | ---: | ---: | ---: |
| GPT-4.1-mini, initial harness | 200/200 | 44.49% | 40.33–48.44% | 42.64% |
| GPT-5.4-mini, reasoning disabled, initial harness | 200/200 | 46.37% | 41.17–51.71% | 44.41% |
| **GPT-6.1 Sol, medium reasoning, v3 harness** | **200/200** | **64.61%** | **61.08–68.16%** | **62.56%** |

The gain over the previous complete GPT-5.4-mini run is **18.24 percentage
points**, with a paired conversation-bootstrap 95% interval of **13.95–22.52
points**. Both the reader and harness changed, so this does not isolate the model
effect. Earlier BEAM failures informed generic harness improvements, and the full
200-question run then used a fixed configuration; transfer to a fresh benchmark
has not been measured.

These are mean rubric scores, not percentages of wholly correct answers. The
primary metric follows upstream reporting: normalized Kendall tau for event
ordering and mean nugget scores for the other nine abilities. The event-F1-adjusted
metric uses `tau_norm × event F1` for event ordering. Both are reported because
`tau_norm` alone can score high when events are omitted, whereas F1 penalizes both
missing and extra events. See [methodology](docs/methodology.md).

Intervals use 10,000 resamples of entire conversations, seed 20261107. They
measure variation across these ten histories, not judge disagreement, repeated
model sampling, or selection effects from development. The previously discussed
52.28% was a [100-question replay with fixed evidence](FAILURE_ANALYSIS.md), not
a full end-to-end benchmark result.

## Per ability

Each cell averages 20 questions. Scores are percentages.

| Ability | GPT-4.1-mini | GPT-5.4-mini | GPT-6.1 Sol medium, v3 |
| --- | ---: | ---: | ---: |
| Abstention | 35.00 | 70.00 | 90.00 |
| Contradiction resolution | 32.50 | 41.25 | 51.25 |
| Event ordering, primary | 20.60 | 23.22 | 32.64 |
| Information extraction | 51.25 | 55.00 | 75.00 |
| Instruction following | 61.25 | 55.00 | 77.50 |
| Knowledge update | 67.50 | 75.00 | 78.75 |
| Multi-session reasoning | 35.33 | 21.83 | 44.50 |
| Preference following | 66.25 | 56.25 | 72.50 |
| Summarization | 57.69 | 58.68 | 68.94 |
| Temporal reasoning | 17.50 | 7.50 | 55.00 |
| Event ordering, event-F1-adjusted (secondary) | 2.07 | 3.55 | 12.17 |

Event reconstruction and multi-session aggregation remain the weakest areas.
The event-ordering drop after multiplying by F1 shows that recovering the
required event set without adding extra events is still difficult. Batch reviews
also found incorrect episode/date selection, missed conflicting claims, and answers that requested
clarification without supplying the requested information. These observations do
not isolate retrieval failures from reader decisions or reference/judge issues.

## Corpus and configuration

All runs use the same pinned BEAM-10M release: **208,696 messages, 108,051,936
content tokens** under `o200k_base`, and **10,298 memory blocks**. Individual
histories measure 9.66–12.93 million tokens. All canonical chat messages,
including assistant messages, are present; no history truncation or question
subsampling was used for these complete results. Separate generation plans and
profiles were not indexed. Original inline message text was retained verbatim.

The local GoodMem server at `https://localhost:8080` uses registered **Voyage 4
Large** embeddings and **Voyage Rerank 3**. Memory blocks are approximately
48,000 characters; recursive chunks are 3,200 characters with 400-character
overlap. HNSW uses `efSearch=256`, relaxed iterative scanning, and
`maxScanTuples=100000`. Model inference uses hosted providers through GoodMem's
proxy. The server is a development build with local changes, recorded in each
protocol.

| Setting | Initial harness, both mini readers | GPT-6.1 Sol medium, v3 |
| --- | --- | --- |
| API | Chat Completions | Responses; medium reasoning, all-turn reasoning context |
| Dense candidates → reranked chunks per search | 128 → 12 | 128 → 20 |
| Tool-round limit | 16 | 32 |
| Active-context limit | 180,000 characters | 2,800,000 characters |
| Output limit per reader call | 5,000 tokens | 16,000 tokens |
| Minimum distinct searches | First search required | Three distinct queries required |
| Submission checks | Model chooses when to finish | Cited factual checklist, then a fresh search/read after drafting |
| Evidence tools | Search and original-message reads | Same, plus exact source-header restoration and date calculator |
| Judge | GPT-4.1-mini, temperature 0 | Same judge and upstream prompts |

Source headers are restored only after exact matching inside the parent block
already returned by GoodMem. This does not search the rest of the corpus.
Original-message reads remain limited to 31 messages and 24,000 characters.
The Responses loop retains opaque reasoning between calls without logging or
exporting it. Actual response identifiers were `gpt-6.1-sol` for the new reader
and `gpt-4.1-mini-2025-04-14` for its judge.

The harness contains no named benchmark subjects, answer-specific facts,
hand-written queries for individual questions, or question-ID branches. The
reader receives no ability label, rubric, or reference answer. General
instructions cover preference retrieval, conflicting updates, recurring events,
summary coverage, and checking evidence before stopping. Chronological lists
use one event per line to match the released evaluator's input format.

A 50-question v2 development run and a two-question context-retention diagnostic
preceded v3. All 200 v3 answers use the same source and configuration. After the
first 25-question checkpoint, that configuration was kept unchanged through the
remaining seven reviews. No successful answer was replaced because of a low
score. The [experiment record](docs/gpt61-experiment.md) preserves the revisions,
subset results, and all eight batch scores.

## Retrieval loop and usage

| Recorded behavior | GPT-4.1-mini | GPT-5.4-mini | GPT-6.1 Sol medium, v3 |
| --- | ---: | ---: | ---: |
| Mean successful searches/question | 1.125 | 1.240 | 6.165 |
| Questions using multiple searches | 21/200 | 39/200 | 200/200 |
| Maximum searches for one question | 4 | 4 | 23 |
| Questions using original-message reads | 15/200 | 12/200 | 192/200 |
| Total successful reads | 18 | 12 | 596 |
| Questions exhausting their tool-round budget | 0 | 0 | 0 |
| Context evictions | 0 | 0 | 0 |
| Reader input tokens | 3,640,763 | 4,158,777 | 253,812,210 |
| Cached input tokens | Not summarized | Not summarized | 225,085,232 |
| Reader output tokens | 56,884 | 40,050 | 302,750 |

The new run made **1,233 searches and 2,252 reader calls**. All 200 final answers
passed the recorded stopping checks. Those checks establish a procedure, not
factual completeness or correctness. Four invalid tool requests were corrected
within the loop: one oversized read and three citation/checklist errors.

The larger context and longer trajectories substantially increase input usage;
88.68% of the new run's input tokens were reported cached. Output totals include
34,360 reasoning tokens. Usage covers successful full-run answer trajectories
only, excluding development, diagnostics, probes, and judging. It is not a
billing estimate. Shared providers and paced retrieval also prevent a controlled
latency comparison.

## Integrity and artifacts

All 200 answers and judgments completed without question-level errors. Internal
HTTP retries were not instrumented. Four judge explanations required the same
narrow invalid-backslash normalization used in the earlier runs; numeric scores
and raw outputs were preserved. No event-ordering tau was undefined.

After evaluation, all **10,298 memories** were again `COMPLETED`. Deterministic
IDs and stored ingestion metadata digests matched the prepared blocks, whose
hashes matched the index manifest. This verifies readiness and recorded source
identity; it does not re-embed or reread every stored content body.

The answer-generation package matches commit
`e2935c2d94b7ef997a8d19f92583d95bb90fcd6e` and the frozen protocol hash. Export
checks validated 2,252 private tool-payload hashes. Independent offline checks
recomputed scores, confidence intervals, tool counts, and all batch comparisons.
The procedure audit checked model/reasoning settings, citation IDs, submission
gates, final answers, and reported usage. It is not an entailment audit.

| GPT-6.1 Sol medium artifact | Contents |
| --- | --- |
| [Summary](results/gpt-6.1-sol-medium-v3/summary.json) | All aggregates and confidence interval |
| [Protocol](results/gpt-6.1-sol-medium-v3/protocol.json) | Frozen settings, model registrations, prompt and source hashes |
| [Outcomes](results/gpt-6.1-sol-medium-v3/outcomes.jsonl.gz) | All answers, rubrics, and raw judgments |
| [Retrieval traces](results/gpt-6.1-sol-medium-v3/retrieval-traces.jsonl.gz) | Queries, evidence identifiers, tool decisions, and usage |
| [Operations](results/gpt-6.1-sol-medium-v3/operations.json) | Model identities, tool errors, normalization, and usage scope |
| [Procedure audit](results/gpt-6.1-sol-medium-v3/procedure-audit.json) | Stopping and tool-protocol checks |
| [Paired comparison](results/gpt-6.1-sol-medium-v3/comparison.json) | Difference from the previous complete GPT-5.4-mini run |
| [Index verification](results/gpt-6.1-sol-medium-v3/index-verification.json) | Final readiness and stored ingestion metadata check |
| [Batch reviews](results/gpt-6.1-sol-medium-v3/batches/reviews.jsonl) | Decisions at each 25-question checkpoint |

The [initial results report](docs/initial-results.md) retains both earlier full
runs and their operational history. Public artifacts include dataset checksums;
full source excerpts and instance credentials remain in ignored local files.

Recompute published aggregates and verify the recorded procedure without an API
key, from the repository root:

```bash
uv sync --locked
uv run python tools/verify_results.py results/gpt-4.1-mini results/gpt-5.4-mini results/gpt-6.1-sol-medium-v3
uv run python tools/verify_batches.py experiments/gpt61-medium-v2-50q results/gpt-6.1-sol-medium-v3
uv run python tools/audit_agent_trace.py results/gpt-6.1-sol-medium-v3
sha256sum -c results/artifacts.sha256
sha256sum -c experiments/artifacts.sha256
```

The 20 offline tests cover data isolation, iterative retrieval, Responses state,
stopping checks, failures, budget behavior, event ranking, parser normalization,
and incomplete reports. Event metrics were also checked against the pinned
upstream implementation on eleven ranking edge cases; the nugget prompt matches
upstream byte for byte.

Dataset-derived artifacts are **CC BY-SA 4.0**, attributed to BEAM by Tavakoli
et al. See [third-party notices](THIRD_PARTY_NOTICES.md). The
[published-results comparison](docs/published-beam-results.md) lists other
BEAM-10M scores and the differences in readers, judges, and scoring rules that
affect comparisons.
