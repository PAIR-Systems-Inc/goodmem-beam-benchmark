# Failure analysis: retrieval coverage and reader errors

This historical diagnostic preceded GPT-6.1 harness development. The later full
run scored 64.61%; see [current results](RESULTS.md). The observations below concern
the original GPT-5.4-mini trajectories, not the new run.

**Both matter.** Re-answering 100 questions with a different reader, using the
exact same tool observations and no new retrieval, raised the score on those
questions from **40.12% to 52.28%**. The original end-to-end results remain
44.49% and 46.37%; this diagnostic is not another full benchmark score.

Analysis date: 7 November 2026. Source run: GPT-5.4-mini with reasoning disabled.
All original answers, grades, prompts, and index contents remain fixed.

## Controlled reader replay

For all 100 questions in five factual/reasoning categories, reconstruct the
original conversation with the agent, including its search calls, read calls, and
exact observations. Omit the original final answer. Ask GPT-5.2 with low reasoning
effort to answer using that evidence, with no further tools. The reader receives
no rubric, category, reference answer, or prior grade. Grade its answers with the
same GPT-4.1-mini judge and original BEAM rubric prompts.

| Category | Questions | Original reader | GPT-5.2 replay, same evidence |
| --- | ---: | ---: | ---: |
| Information extraction | 20 | 55.00% | 72.50% |
| Knowledge update | 20 | 75.00% | 77.50% |
| Contradiction resolution | 20 | 41.25% | 58.75% |
| Multi-session reasoning | 20 | 21.83% | 31.42% |
| Temporal reasoning | 20 | 7.50% | 21.25% |
| **Mean over these 100 questions** | **100** | **40.12%** | **52.28%** |

The paired gain is **12.17 percentage points**, with a conversation-bootstrap
95% interval of **4.08–22.50 points** (10,000 resamples, seed 20261107).
Twenty-six questions improved, eleven worsened, and sixty-three were unchanged.
All 100 replay answers and judgments completed without errors. Actual model:
`gpt-5.2-2025-12-11`.

This changes the final reader, reasoning setting, and final-answer instruction
together. It isolates a downstream evidence-use opportunity from retrieval;
it does not isolate model size from reasoning effort or instruction effects.
The earlier search trajectory remains the original reader's. A full GPT-5.2
agent would choose its own searches, so its end-to-end score is still unmeasured.
The interval also does not account for judge sampling variability.

## Concrete reader failures

These supporting passages were already in the original reader's context:

| Question ID | Available evidence | Original answer | Replay answer |
| --- | --- | --- | --- |
| `4:knowledge_update:0` | User: “increasing my correct solutions to 9 within 50 minutes,” explicitly about Section 14.3 gradients | 6 problems, 6 hours | 9 correct, 50 minutes |
| `1:temporal_reasoning:1` | User work on context-window segmentation dated November 1, 2024; starting query-rewriting pipelines dated December 16, 2024 | 26 days | 45 days, with the required endpoints |
| `6:information_extraction:1` | User: “I just got a $10 SD card from TechMart”; other retrieved messages mention $18 and $19 | $19 | Surfaces all three amounts, including $10 |
| `1:contradiction_resolution:0` | Both “I'm setting up diagnostic logs” and “I've never set up diagnostic logs” occur in the returned chunks | Yes, with implementation caveats; misses the contradiction | Acknowledges both statements, although its attempted chronology is still questionable |

The last two illustrate that an improved rubric score does not establish perfect
reasoning: the rubric rewards recovering the target fact, while the history can
contain unresolved contradictory statements. The raw responses are exported.

## Evidence-coverage audit

A separate GPT-5.2 audit sees the question, its rubric, and the original retrieved
excerpts, but is blind to the original answer and score. Each rubric is labeled
supported, partial, absent, behavioral, or uncertain. Factual support requires
literal quotes that are checked against the specific cited source text.

All **160 non-event, non-abstention questions** were audited. Of the **112 that
lost credit**:

| Evidence classification | Questions |
| --- | ---: |
| All factual targets supported by the retrieved context | 32 |
| At least one required fact absent or only partially supported | 68 |
| Ambiguous or unverifiable support | 12 |

Across those 160 questions, **46 answers missed at least one rubric target the
auditor found supported**. These can overlap with questions that also lacked
other facts. Therefore the 68 evidence-limited cases are not 68 purely retrieval
failures: some contain both missing evidence and reader mistakes.

Event coverage is especially weak. In **19 successfully audited event-ordering
questions**, out of 213 event rubrics, only 39 were classified fully supported,
32 partially supported, 112 absent, and 30 uncertain. This audit checks presence
of the event, not correctness of the order. One of the twenty event audits did
not return a valid full item list despite retries and remains an explicit error.
Abstention is excluded because a small set of retrieved excerpts cannot establish
that something is absent from the full history.

These are diagnostic model judgments, not a gold passage-recall measurement or
a proven causal allocation of lost points. There were **97 citation-validation
failures**, conservatively relabeled uncertain. Quotes were not repaired into
support. One initial pilot using GPT-5.4-mini was rejected after inspection showed
loose semantic matches and citation errors; it is excluded from these statistics.
The complete audits, reasons, and original responses are available below.

## Why the first agent leaves evidence behind

The loop permits 16 rounds, but most questions finish after one search. Among
questions losing credit, **18/18 contradiction questions** and **15/18
multi-session questions** used only one search. There is no explicit check that
both sides of a contradiction, every constituent of a sum, or the entire requested
date range have been covered before the model can answer.

Each search returns twelve chunks. The full run returned 2,976 chunks across
248 searches, averaging 8.32 distinct memory blocks per search. This favors a few
strong topical matches; it does not guarantee coverage across dates or events.
Broad event-list questions often require many separate facts, not twelve similar
passages about one part of the project.

Chunk boundaries also weaken provenance. **309/2,976 returned chunks (10.4%) had
no inline message header**, because the character-based chunker split inside a
message. Their memory-level start/end IDs remain available, but the chunk does
not directly identify its speaker and date. Long assistant explanations and code
examples compete with short user statements. Sequence metadata is present; the
agent does not systematically turn it into a timeline or retrieve before/after
a particular event.

The current traces save the final reranked chunks. They do **not** preserve the
entire initial 128-candidate pool. Where facts are missing, this analysis cannot
yet separate dense-search recall, reranker pruning, and inadequate query
decomposition. It would be premature to blame the embedding model or GoodMem's
index alone.

## Reference-data caveat

`5:information_extraction:0` asks for a tokenization-library version and expects
`4.35.0`. A scan of that complete prepared conversation finds zero matches for
`4.35.0`, `tokeniz`, or `transformers`; the history concerns functional analysis.
The original reader abstains and receives zero. This is a suspicious reference
item, not a demonstrated retrieval miss. Literal absence does not prove semantic
absence, so it is flagged for source review rather than removed or rescored.
This isolated issue does not explain the overall gap.

## Comparison and next experiments

The 60%+ comparison is real: [Hindsight reports 64.1% on BEAM-10M](https://hindsight.vectorize.io/blog/2026/04/02/beam-sota).
That system describes fact extraction and synthesized observations in addition to
retrieval. Our first method indexes raw history and uses a small reader with
reasoning disabled in the GPT-5.4-mini run. These are different system designs;
a controlled comparison would also pin the reader, judge, and scoring version.

The evidence supports testing these changes separately:

1. Use a reasoning-enabled reader and require an explicit evidence check before
   final answers: conflicting statements, current versus old facts, every sum
   component, and both temporal endpoints.
2. Make additional queries target uncovered facts and date ranges; add temporal
   range search and deterministic date arithmetic. Merely increasing the round
   limit will not help when the model stops voluntarily.
3. Repeat message identity, speaker, and date on every chunk. Compare raw-history
   retrieval with a separate index of user facts while preserving the source text.
4. Save the initial candidate pool, measure support before and after reranking,
   and compare diverse time coverage with simple top-k selection.

These are proposed follow-up experiments, not claimed score improvements.

## Artifacts and reproduction

- [Replay summary](diagnostics/2026-11-07/replay-summary.json),
  [protocol](diagnostics/2026-11-07/replay-protocol.json), and
  [all 100 original/replayed answers and grades](diagnostics/2026-11-07/replay-outcomes.jsonl.gz).
- [Coverage summary](diagnostics/2026-11-07/coverage-summary.json),
  [auditor prompt](diagnostics/2026-11-07/coverage-protocol.json), and
  [audits with supporting quotes](diagnostics/2026-11-07/coverage-audits.jsonl.gz).
- [Original trace statistics](diagnostics/2026-11-07/trace-statistics.json).

With the original local run and credentials available:

```bash
uv run python tools/audit_evidence.py --out runs/diagnostic-evidence-gpt52
uv run python tools/replay_reader.py
uv run python tools/summarize_diagnostics.py
sha256sum -c diagnostics/2026-11-07/artifacts.sha256
```

Diagnostic artifacts are BEAM-derived material under CC BY-SA 4.0. The original
published result artifacts are unchanged. This analysis uses test-set outcomes
post hoc and should not be described as independent validation of tuned changes.
