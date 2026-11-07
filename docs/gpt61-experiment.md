# GPT-6.1 Sol run

The complete 200-question run scored **64.61%** on the primary metric and
**62.56%** with event F1 included. Its 95% conversation-bootstrap interval is
61.08–68.16%. The reader was GPT-6.1 Sol with medium reasoning; the judge was
GPT-4.1-mini.

## Batch results

Each batch finished and was reviewed before the next began. All eight used the
same source, prompt, and configuration. No answer was replaced for scoring poorly.

| Batch | Questions | Score |
| --- | ---: | ---: |
| 1 | 25 | 68.51% |
| 2 | 25 | 65.19% |
| 3 | 25 | 59.21% |
| 4 | 25 | 56.68% |
| 5 | 25 | 57.59% |
| 6 | 25 | 71.76% |
| 7 | 25 | 64.93% |
| 8 | 25 | 73.02% |
| **All** | **200** | **64.61%** |

There were no question-level errors, context evictions, or tool-budget
exhaustions. The agent made 1,233 searches and 596 original-message reads.
See the [full results](../RESULTS.md) and
[frozen protocol](../results/gpt-6.1-sol-medium-v3/protocol.json).

## Reproduce the run

Use `config/gpt61-medium-v3.toml` with a completed Voyage 4 Large index,
Voyage Rerank 3, and a GoodMem proxy registration that can access `gpt-6.1-sol`.
The configuration uses the Responses API with medium reasoning, 128 retrieval
candidates reranked to 20 chunks, 32 tool rounds, a 2,800,000-character context
limit, and 16,000 output tokens per call.

```bash
BEAM_READER_MODEL=gpt-6.1-sol uv run python tools/run_batch.py \
  --config config/gpt61-medium-v3.toml --run runs/my-sol-run --batch 0
```

The command answers and grades one batch of 25, then exits. Review its answers,
traces, and score before incrementing `--batch`. Run batches 0 through 7 to cover
all 200 questions. If a batch has infrastructure failures, rerun the same command;
successful answers are preserved.

Question order is fixed before the first batch: round-robin conversations, with
each conversation's questions rotated by twice its conversation index. Each batch
contains two or three questions from every conversation and ability. Every
question appears exactly once across the eight batches.

Changing source or configuration requires a new run directory. Scores from
different configurations must not be combined into one full-run result. Missing
answers or judgments prevent reporting a complete score.

The agent receives only the question, conversation-scoped evidence tools, and
general instructions. Ability labels and rubrics are used outside the agent for
batch selection and grading. Stopping requires three distinct searches, a factual
checklist citing returned evidence, and a fresh search or read after drafting.
The [methodology](methodology.md) describes these checks and the scoring rules.

Verify the published run without credentials:

```bash
uv run python tools/verify_results.py results/gpt-6.1-sol-medium-v3
uv run python tools/audit_agent_trace.py results/gpt-6.1-sol-medium-v3
```
