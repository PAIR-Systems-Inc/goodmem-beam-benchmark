# GoodMem on BEAM-10M

**GoodMem + GPT-6.1 Sol, medium reasoning, scored 64.61% on all 200 BEAM-10M
questions.** The score is **62.56%** when event ordering includes event F1,
which accounts for missing and extra events.

This repository runs a ReAct agent over [GoodMem](https://github.com/PAIR-Systems-Inc/goodmem).
The agent searches repeatedly, reads nearby messages, and checks its evidence
before answering. The full run used one fixed configuration, with a review
after every 25 questions.

BEAM comes from [*Beyond a Million Tokens: Benchmarking and Enhancing Long-Term
Memory in LLMs*](https://arxiv.org/abs/2510.27246), Tavakoli et al., ICLR 2026.
This evaluates GoodMem on the paper's benchmark; it does not implement LIGHT.

| Run | Configuration |
| --- | --- |
| Reader | GPT-6.1 Sol, medium reasoning |
| Judge | GPT-4.1-mini |
| Embeddings / reranker | Voyage 4 Large / Voyage Rerank 3 |
| Corpus | All ten BEAM-10M conversations, 208,696 messages |
| Measured content tokens | 108,051,936 under `o200k_base` |
| Questions | 200/200 |

See [results](RESULTS.md) for scores by ability, retrieval statistics, and
artifacts; [methodology](docs/methodology.md) for the evaluation protocol; and
[published BEAM-10M scores](docs/published-beam-results.md) for other systems.

## Run it

Requires Python 3.11+, `uv`, and a running GoodMem instance with Voyage 4 Large
and Voyage Rerank 3 registered, plus credentials for the reader and judge.
Model inference uses hosted providers through GoodMem's proxy.

```bash
uv sync --locked
cp .env.example .env
# Fill in the GoodMem credentials and model registration IDs.
uv run beam models
uv run beam download
uv run beam prepare
uv run beam ingest

BEAM_READER_MODEL=gpt-6.1-sol uv run python tools/run_batch.py \
  --config config/gpt61-medium-v3.toml --run runs/my-sol-run --batch 0
```

The batch command answers and grades 25 questions, then exits. Review the result
before running the next batch, numbered 0 through 7. Rerunning a batch retries
failed items and preserves successful answers. See the [Sol run record](docs/gpt61-experiment.md).
After all eight batches:

```bash
uv run beam --run runs/my-sol-run report
uv run beam --run runs/my-sol-run export results/my-sol-run
```

Credentials belong in environment variables or the ignored `.env` file. Model
registration IDs are specific to your GoodMem instance. Set `GOODMEM_TLS_VERIFY=false`
only for a trusted local instance with a development certificate.

Ingestion is resumable. Every memory must reach `COMPLETED` before answering.
`beam repair` waits for processing to finish, then recreates failed memories
belonging to this index. Use a new index directory if the index settings change,
and a new run directory if the reader, prompt, or agent settings change.

## How it works

1. Read the canonical `chat` field in chronological order, including assistant
   messages. Keep questions and grading rubrics separate from the indexed text.
2. Pack adjacent messages into approximately 48,000-character memories, one
   GoodMem space per conversation. Index 3,200-character chunks with 400-character
   overlap, retaining message IDs and explicit dates.
3. For each search, retrieve 128 candidates and rerank to 20 chunks. The agent
   can issue new searches and read up to 31 neighboring source messages at a time.
4. Allow up to 32 tool rounds. Before submitting, require three distinct queries,
   a factual checklist citing returned evidence, and a fresh search or read after
   the first draft. These checks do not guarantee that the answer is correct.
5. Grade the completed answers with GPT-4.1-mini and the upstream BEAM prompts.

The agent receives the question and conversation-scoped evidence. It receives
no ability label, rubric, reference answer, separate user profile, or generation
plan. Original chat text is preserved, including any inline markers in the release.
BEAM permits repeated retrieval; the agent averaged 6.165 searches per question.

## Verify the results

Nine abilities use nugget scores of 0, 0.5, or 1. Event ordering uses normalized
Kendall tau. We report both the upstream reporting metric and the variant that
multiplies event-ordering scores by event F1. Overall scores are means across
questions, not percentages of wholly correct answers.

Public artifacts include answers, rubrics, judge outputs, retrieval queries,
evidence identifiers, dataset checksums, and the frozen configuration. Full
source excerpts remain in ignored local traces. Verification needs no API key:

```bash
uv run python tools/verify_results.py results/gpt-6.1-sol-medium-v3
uv run python tools/audit_agent_trace.py results/gpt-6.1-sol-medium-v3
uv run pytest -q
uv run ruff check src tests tools
```

## License and attribution

Harness: [MIT](LICENSE). BEAM-derived results: CC BY-SA 4.0. Download the source
corpus from its authors; it is not included here. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the upstream prompt's license.

```bibtex
@article{tavakoli2025beyond,
  title={Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs},
  author={Tavakoli, Mohammad and Salemi, Alireza and Ye, Carrie and Abdalla, Mohamed
          and Zamani, Hamed and Mitchell, J. Ross},
  journal={arXiv preprint arXiv:2510.27246},
  year={2025}
}
```
