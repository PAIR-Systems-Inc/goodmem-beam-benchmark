# Initial BEAM-10M results

Historical report for the first two complete runs, before GPT-6.1 harness
development. See [current results](../RESULTS.md) for the later evaluation.

**GoodMem + the basic ReAct agent scored 44.49% with GPT-4.1-mini and 46.37%
with GPT-5.4-mini.** Each run answered and graded all **200 questions across all
ten complete histories**. Evaluation completed on 7 November 2026.

| Reader | Questions graded | Primary score | 95% conversation bootstrap CI | Recall-adjusted score |
| --- | ---: | ---: | ---: | ---: |
| GPT-4.1-mini | 200/200 | **44.49%** | 40.33–48.44% | 42.64% |
| GPT-5.4-mini, reasoning disabled | 200/200 | **46.37%** | 41.17–51.71% | 44.41% |

These are mean rubric scores, not percentages of wholly correct answers. The
primary metric follows the upstream reporting script: normalized Kendall tau for
event ordering and mean nugget scores for the other nine abilities. The secondary
metric uses `tau_norm × event F1` for event ordering. Both are reported because
omitting events can receive a high ordering score. See [methodology](methodology.md).

The paired difference is **+1.89 percentage points** for GPT-5.4-mini; its 95%
conversation-bootstrap interval is **−2.80 to +6.73 points**. This experiment does
not establish a reliable difference between the readers. Bootstrap intervals use
10,000 resamples of entire conversations, seed 20261107. They do not measure
judge disagreement or repeated-run variation.

## Per ability

Each cell averages 20 questions. Scores are percentages.

| Ability | GPT-4.1-mini | GPT-5.4-mini |
| --- | ---: | ---: |
| Abstention | 35.00 | 70.00 |
| Contradiction resolution | 32.50 | 41.25 |
| Event ordering, primary | 20.60 | 23.22 |
| Information extraction | 51.25 | 55.00 |
| Instruction following | 61.25 | 55.00 |
| Knowledge update | 67.50 | 75.00 |
| Multi-session reasoning | 35.33 | 21.83 |
| Preference following | 66.25 | 56.25 |
| Summarization | 57.69 | 58.68 |
| Temporal reasoning | 17.50 | 7.50 |
| Event ordering, recall-adjusted (secondary) | 2.07 | 3.55 |

Knowledge updates and preference/instruction questions perform better than
temporal reasoning and event reconstruction. The large event-ordering drop under
recall adjustment indicates poor recovery of the required event set. These are
observed weaknesses of this raw-history retrieval method; the experiment does
not isolate whether failures come from retrieval, reader decisions, or grading.

## Corpus and configuration

- Pinned BEAM-10M release: **208,696 messages, 108,051,936 content tokens** under
  `o200k_base`, and 10,298 ingested memory blocks. Individual histories measure
  9.66–12.93 million tokens. All canonical chat messages, including assistant
  messages, are present. No history truncation or question subsampling was used.
- Local GoodMem server at `https://localhost:8080`, using its registered
  **Voyage 4 Large** embedder and **Voyage Rerank 3** reranker. All 10,298 memories
  were verified `COMPLETED` before answering and again after evaluation.
- Approximately 48,000-character memory blocks; GoodMem recursive chunks of
  3,200 characters with 400-character overlap. Each search retrieves 128 dense
  candidates and reranks to 12 chunks. HNSW `efSearch=256` with relaxed iterative
  scanning and `maxScanTuples=100000`.
- Same agent prompt, tools, corpus, and retrieval configuration for both readers.
  Up to 16 tool rounds; the model decides whether to search again, read adjacent
  messages, or answer. The active context budget is 180,000 characters, and each
  reader call allows 5,000 output tokens.
- GPT-4.1-mini uses temperature 0. GPT-5.4-mini uses `reasoning_effort=none` because
  its Chat Completions endpoint rejects function tools with reasoning enabled.
  The common judge is GPT-4.1-mini at temperature 0 with the released BEAM prompts.
- Actual response model identifiers: `gpt-4.1-mini-2025-04-14` and
  `gpt-5.4-mini-2026-03-17`. Model inference uses hosted providers through the local
  GoodMem proxy. The server reports a development build with local changes;
  its exact reported version is in the exported protocol.

## Retrieval loop behavior

| Recorded behavior | GPT-4.1-mini | GPT-5.4-mini |
| --- | ---: | ---: |
| Mean successful searches/question | 1.125 | 1.240 |
| Questions using multiple searches | 21/200 | 39/200 |
| Maximum searches for one question | 4 | 4 |
| Questions using neighboring-message reads | 15/200 | 12/200 |
| Total successful reads | 18 | 12 |
| Questions exhausting the 16-round budget | 0 | 0 |
| Context compactions | 0 | 0 |
| Recoverable oversized read requests | 6 | 9 |
| Reader input tokens, successful final attempts | 3,640,763 | 4,158,777 |
| Reader output tokens, successful final attempts | 56,884 | 40,050 |

Repeated retrieval is enabled and occurred in both live runs. Most answers still
followed a single search because the readers chose to stop. The harness does not
guarantee they have found every relevant fact when they stop. These runs are a
basic starting point, not an optimized BEAM system or a reproduction of LIGHT.
No prompt or retrieval tuning was performed against these measured scores.

## Operational record and reproducibility

Initial embedding throttles left 1,588 memories failed. Paced repair recreated
only those owned failed memories, with the same content and deterministic IDs.
Evaluation began after the full index was searchable.

Reranking HTTP 429 errors interrupted 36 baseline and 55 GPT-5.4-mini answers.
Only failed attempts were rerun, with paced retrieval; successful answers stayed
fixed. All retries succeeded. Final reports contain zero answer or grading errors.
Timing is affected by shared provider limits and retry pacing, so this is not a
controlled latency comparison. Token totals above exclude failed attempts and
judge calls and should not be treated as a billing estimate.

The read tool rejected six baseline and nine GPT-5.4-mini requests that exceeded
its 31-message range limit. These were returned to the agent as input errors;
the agent continued, and their model calls remain included in the recorded usage.

One judge explanation per run contained invalid JSON backslashes around LaTeX.
The final parser preserved those literal characters and the numeric scores;
original raw responses and normalization flags are exported. Structural JSON
errors remain failures. Answer-generation source matches commit
`b821af2444b39b105c28b506bcbdec80f7b7e514`; the parser correction is in
`ad2803118d984aa35b66a80cde6cfc2ed0012108`. The frozen answer protocols were retained.

| Artifact | GPT-4.1-mini | GPT-5.4-mini |
| --- | --- | --- |
| Aggregate scores | [summary](../results/gpt-4.1-mini/summary.json) | [summary](../results/gpt-5.4-mini/summary.json) |
| Frozen setup and source hashes | [protocol](../results/gpt-4.1-mini/protocol.json) | [protocol](../results/gpt-5.4-mini/protocol.json) |
| All answers, rubrics, and grades | [outcomes](../results/gpt-4.1-mini/outcomes.jsonl.gz) | [outcomes](../results/gpt-5.4-mini/outcomes.jsonl.gz) |
| Queries and evidence identifiers | [traces](../results/gpt-4.1-mini/retrieval-traces.jsonl.gz) | [traces](../results/gpt-5.4-mini/retrieval-traces.jsonl.gz) |
| Retries, model snapshots, parser audit | [operations](../results/gpt-4.1-mini/operations.json) | [operations](../results/gpt-5.4-mini/operations.json) |

Also included: [full index verification](../results/index-verification.json),
[paired comparison](../results/comparison.json), dataset checksums, and retry histories.
Full source excerpts and credentials remain in ignored local files.

Recompute every published aggregate and verify tool counts without an API key:

```bash
uv sync --locked
uv run python tools/verify_results.py results/gpt-4.1-mini results/gpt-5.4-mini
sha256sum -c results/artifacts.sha256
```

The fifteen offline tests cover data isolation, iterative retrieval, failures,
budget behavior, event ranking, parser normalization, and incomplete-report
handling. Event metrics were additionally checked against the pinned upstream
implementation on eleven ranking edge cases; the nugget prompt matches upstream
byte for byte.

Dataset-derived artifacts are **CC BY-SA 4.0**, attributed to BEAM by Tavakoli et al.
See [third-party notices](../THIRD_PARTY_NOTICES.md). Results should not be presented
as a controlled comparison with paper baselines that use different readers,
judges, memory construction, or inference budgets.
