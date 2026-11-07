# Evaluation protocol

The paper was read before implementing this harness, including sections 2–4 and
the evaluation prompts. The official evaluation implementation was then inspected
to resolve details the prose does not fully specify.

## Pinned sources

- Paper: https://arxiv.org/html/2510.27246v2 (21 February 2026).
- Official code: https://github.com/mohammadtavakoli78/BEAM/tree/b2da22eac88bb0874c64665f13457eb99835774a .
- Dataset: `Mohammadta/BEAM-10M`, revision
  `9b2096193fe74e2837e4713e483351e19817773c`, both Parquet shards, all ten rows.
- Source file checksums and measured token counts are included with every export.

## Questions and histories

There are two questions per ability per conversation. Each question is asked
after the complete history, independently of every other probing question.
The canonical top-level `chat` field already combines the ten plans and assigns
global message IDs. The separate `plans` field repeats data and contains generation
scaffolding; it is never read or indexed. Sorting the strings `plan-1` to `plan-10`
lexically would corrupt chronology, so the adapter uses the release's list order
and verifies contiguous message IDs.

Only the current question text is supplied to the reader. The answer function's
interface does not accept rubric or category fields. Gold annotations are loaded
by the separate judge. The local range-reading tool opens a read-only SQLite
transcript containing just message IDs and rendered chat text.

Actual token counts use `o200k_base` on each message's content and exclude added
headers. They do not claim to reproduce the authors' tokenizer. Full histories
are preserved even for rows measuring slightly under 10M with this tokenizer.

## Agent and retrieval

This method uses raw conversation indexing and iterative inference. There is no
offline LLM fact extraction or LIGHT scratchpad. The agent can issue new semantic
queries and read nearby messages after each observation. Repeated retrieval is
part of the evaluated method. BEAM does not impose a single-retrieval restriction.

Temporal position is already retained at indexing: each source message has a
chronological ID and any explicit chat date, while each memory carries its block
number and start/end message IDs. Retrieval returns those fields plus the chunk
sequence number. The current range-reading tool can inspect neighboring messages.
Sequence records when something was mentioned, not necessarily when the described
event occurred. Explicit before/after search constraints and assembling a dated
event timeline are possible follow-up experiments, not features evaluated here.

GoodMem performs dense retrieval and registered cross-encoder reranking. HNSW uses
`efSearch=256`, relaxed iterative scanning, and `maxScanTuples=100000`. The endpoint
must return a complete rerank result set. Provider or stream failures are recorded
as infrastructure errors, never interpreted as evidence absence.

The default run uses temperature 0, 128 candidates, 12 selected chunks per search,
16 tool rounds, and 5,000 output tokens per reader call. Old tool bodies can be
removed when the active history exceeds 180,000 characters. This is a character
budget, not an exact token limit; model-reported usage is retained separately.

The GPT-5.4-mini comparison uses the same tool loop and budgets, with
`reasoning_effort=none` and no temperature parameter. Its Chat Completions endpoint
rejects function tools when reasoning effort is enabled. This model-specific
setting is explicit in `reader_inference` in the frozen protocol. Both readers
use GPT-4.1-mini for grading. The requested model can differ from the display name
of the GoodMem proxy registration; `requested_models` and the response traces
record the actual choice.

The GPT-6.1 Sol medium profile uses the Responses API and preserves every returned
input-history item between calls, including opaque reasoning and assistant phase.
Opaque reasoning is not logged or exported. The client verifies the returned
reasoning effort. Its v3 configuration reranks to 20 chunks from the same 128
candidates, allows 32 tool rounds, and retains up to 2.8 million characters of
active history with 16,000 output tokens per call. It restores missing source
headers only by exact matching within the already-retrieved parent block and adds
a calendar-day calculator. There is no additional full-corpus search outside
GoodMem; original-message reads remain bounded to 31 messages and 24,000 characters.

Stopping requires three distinct normalized queries, a factual coverage checklist
referencing returned evidence IDs, and a new search or message read after the
first eligible answer draft. These controls verify procedure, not semantic
completeness or truth. A final answer at the round limit is explicitly flagged.
The judge and scoring remain unchanged. This profile changes both reader and
harness, so comparison with the earlier runs does not isolate a model effect.

Runs are reviewed in balanced batches of 25. Earlier BEAM failures, a
50-question v2 run, and a two-question context-retention diagnostic informed the
generic v3 harness. All 200 v3 answers then used that fixed configuration without
change across all eight batch reviews. Transfer to a fresh benchmark has not been
measured. See the [experiment record](gpt61-experiment.md) for configuration
history and audits.

## Scoring

For nine abilities, the unchanged `unified_llm_judge_base_prompt` from the pinned
upstream code scores each rubric independently. We average nuggets within a
question, then questions within an ability and conversation, then conversations
and abilities. The balanced full release makes this equivalent to a mean across
all 200 question scores. The judge is GPT-4.1-mini, as in upstream `src/llm.py`.
The original prompt refers to a question but does not interpolate it; this harness
preserves that behavior instead of silently changing the judge.

Event ordering follows the released implementation:

1. Split the answer on newline characters, retaining blank lines.
2. For each predicted line, compare against unused reference events with the
   original YES/NO equivalence prompt. Choose the first positive match in reference
   order. Comparisons can run in batches of four; selection order is unchanged.
3. Build the ordered union, assign missing items a shared last rank, compute
   Kendall tau-b, and normalize it to `(tau + 1) / 2`.
4. The primary report uses `tau_norm`, matching upstream `report_results.py`.
   The event-F1-adjusted report substitutes `tau_norm × event F1` (named
   `final_score` in upstream `compute_metrics.py`), which accounts for both
   missing and extra events.

Undefined tau (for example, a wholly tied sequence) is explicitly mapped to zero
and flagged. Upstream may emit NaN here; the difference is documented rather than
letting JSON or aggregation silently discard the question. Failed or structurally
malformed judge responses stay errors. A narrow parser normalization doubles
invalid backslashes inside JSON strings (for example, LaTeX delimiters in the
reason). It preserves the literal text and numeric score, flags the normalization,
and retains the original response. It cannot repair missing fields, truncated
objects, or invalid scores. The upstream scorer uses a broader JSON repair library.

Intervals resample whole conversations with replacement (10,000 samples, seed
20261107), preserving dependence among the 20 questions about a conversation.
They measure variation across these ten histories, not variability from repeated
model sampling or judge disagreement. A single run cannot establish a ranking
against systems evaluated with different readers, judges, prompts, or budgets.

## Operational integrity

Each index has deterministic space and memory IDs, content hashes, and an exact
configuration fingerprint. Before answering, every memory is rechecked for
`COMPLETED` status. All planned questions remain in the reporting denominator;
an incomplete run has no full overall score. Answers and judgments are atomic
per-question files and can be resumed after transient errors.

The initial measured ingestion submitted all memories before provider throttling
was discovered. The harness now paces new submissions at an estimated 3M input
tokens/minute with a 40% margin for overlap/tokenizer differences. Repair waits
for active processing to finish, then deletes/recreates only failed memories
whose deterministic IDs, space, and source-content hashes match this index.
These operational retries do not change the corpus or retrieval configuration.

Source conversations, credentials, and full retrieval contexts are ignored by Git.
Public exports retain enough identifiers, queries, hashes, and individual grades
to audit and rerun the experiment without disclosing instance credentials.
