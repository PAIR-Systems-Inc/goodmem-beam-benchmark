# GPT-6.1 medium experiment

This is a development experiment after inspecting the earlier BEAM runs and
failure analysis. It changes the reader and harness together, so any improvement
cannot be attributed to the reader alone. It is not a held-out validation result.
The earlier 52.28% was a 100-question GPT-5.2 replay with fixed retrieved evidence;
the previous best complete 200-question score was 46.37% with GPT-5.4-mini.

## Configuration

The requested model is `gpt-6.1-sol`, with medium reasoning through GoodMem's
Responses proxy. [Official model documentation](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
requires Responses for tool calling. The client checks that the provider confirms
medium reasoning. All returned Responses items are replayed between calls,
including assistant phases and opaque encrypted reasoning. Opaque reasoning is
not exported or logged. Provider usage and reasoning settings are recorded.

The same completed Voyage 4 Large index and Voyage Rerank 3 registration are used.
Candidate count stays 128; returned chunks increase from 12 to 20. Source headers
are restored only after an exact substring match within the already-retrieved
parent block. This adds dates, speaker attribution and precise message IDs, not
an additional search mechanism. The `read_messages` tool still reads at most 31
messages, with a 24,000-character limit. A date tool calculates calendar intervals.

The stopping boundary requires at least three distinct normalized queries and a
factual coverage checklist citing returned evidence IDs. After a first eligible
draft, a further search or message read is required before submission can succeed.
It does not automatically prove that every requested fact was found: coverage is
still the reader's judgment. Missing facts should trigger further searches, and
unresolved uncertainty must be acknowledged. At 32 tool rounds, the harness asks
for an answer and records `budget_exhausted`. Active context is bounded to 600,000
characters by omitting old evidence bodies; reasoning and call/result structure
remain intact. Output limit is 16,000 tokens per call. Retrieval requests across
four workers are paced at least three seconds apart to reduce provider throttling.

The judge remains GPT-4.1-mini with the unchanged upstream rubric prompt and
scoring implementation. No rubric, ability label or gold answer enters the agent.

## Review every 25 questions

`tools/run_batch.py` answers and grades one batch, then exits. The order is fixed
before the first batch: round-robin conversations, with each conversation's
question list rotated by twice its conversation index. Each 25-question batch
contains two or three questions from each conversation and each ability. Every
question appears exactly once over eight batches.

```bash
BEAM_READER_MODEL=gpt-6.1-sol uv run python tools/run_batch.py \
  --config config/gpt61-medium-v2.toml --run runs/react-gpt61-medium-v2 --batch 0
```

Inspect the batch's matched comparison against the previous GPT-5.4-mini answers
and traces before incrementing `--batch`. The batch number is zero based. The
batch score is an average over those 25 questions, not a full BEAM score. Missing
answers or judgments suppress that batch score. Full-run reporting remains
incomplete until all 200 are graded. Successful answers are never replaced merely
because they score poorly.

Source/configuration changes require a new run directory. Do not combine batches
from different protocols into a claimed full score. Infrastructure-only retries
preserve the frozen source/settings and successful answers. Both old complete
evaluations and the 100-question replay remain unchanged.

## Checkpoint and revision

V2 was stopped after 50 questions for review. Batch 1 scored 58.02% versus 46.44%
on the same GPT-5.4-mini questions; batch 2 scored 60.03% versus 40.08%. Combined:
59.03% versus 43.26%. These are subset results, not a complete BEAM score. Source
for this run is commit `933e494`.

Long trajectories evicted older evidence. A diagnostic replay reconstructed the
same complete tool history in both conditions, excluding original answers/drafts
and private reasoning. Only the active-context limit differed. On two selected
difficult questions, retaining evidence scored 61.14% versus 24.91% for a timeline,
and 60% versus 60% for a summary. These are selected, single-trial diagnostics;
they do not establish an overall improvement or isolate the original agent's
reasoning state. `tools/replay_context.py` reproduces the experiment.

V3 (`config/gpt61-medium-v3.toml`) raises the active-context limit from 600,000 to
2,800,000 characters. It also adds generic instructions to search explicitly for
user preferences/restrictions, distinguish recurring event instances from updates,
and preserve major themes in summaries. No gold fact, question-specific answer,
ability label or rubric is added. The model, reasoning effort, index, reranker,
retrieval sizes, round limit and judge stay the same. V3 starts a separate run and
is first compared on the same 25-question batch before continuing. This is further
development on BEAM, not an independent test set.
