# GoodMem on BEAM-10M

Evaluate [GoodMem](https://github.com/PAIR-Systems-Inc/goodmem) on the complete
BEAM-10M histories with a small **ReAct agent that retrieves repeatedly**.
The agent can rewrite searches, investigate contradictory statements, and read
neighboring messages before answering. Each question has its own agent state.

BEAM comes from [*Beyond a Million Tokens: Benchmarking and Enhancing Long-Term
Memory in LLMs*](https://arxiv.org/abs/2510.27246), Tavakoli et al., ICLR 2026.
This repository evaluates a GoodMem-based method on BEAM; it does not implement
the paper's LIGHT architecture. It is independent of `goodmem-memory-benchmarks`.

## Scope and results

The pinned [BEAM-10M release](https://huggingface.co/datasets/Mohammadta/BEAM-10M)
contains ten conversations and 200 questions covering ten memory abilities.
All conversation messages are ingested, including the assistant turns.
The downloaded release has **208,696 messages and 108,051,936 content tokens**
under `o200k_base`. Individual conversations measure 9.66–12.93 million tokens
with that tokenizer. “10M” denotes the authors' split, not a filter we apply.

Measured on all 200 questions: **44.49% with GPT-4.1-mini** and **46.37% with
GPT-5.4-mini** on the upstream primary metric. Both use the same GoodMem index
and GPT-4.1-mini judge. See [RESULTS.md](RESULTS.md) for category scores, uncertainty,
retrieval-loop statistics, downloadable evidence, and limitations, and
[the methodology](docs/methodology.md) for scoring and data isolation.

[Failure analysis](FAILURE_ANALYSIS.md) separates missing evidence from reader
errors, including a controlled replay with the exact same retrieved evidence.

## Run it

Requires Python 3.11+, `uv`, and a running GoodMem instance with an embedder,
reranker, and OpenAI-compatible LLM registrations. These may use hosted providers;
a local GoodMem server does not imply local model inference.

```bash
uv sync --locked
cp .env.example .env
# Fill in the instance credentials and model registration IDs.
uv run beam models
uv run beam download
uv run beam prepare
uv run beam ingest
uv run beam run
uv run beam export results/my-run
```

All credentials live in environment variables or the ignored `.env` file.
The provider calls go through GoodMem's proxy using its stored credentials.
Model IDs are instance-specific. The measured configurations use Voyage 4 Large,
Voyage Rerank 3, and either GPT-4.1-mini or GPT-5.4-mini as reader. The judge is
GPT-4.1-mini in both runs.

Set `GOODMEM_TLS_VERIFY=false` only for a trusted local instance with a development
certificate. Normal connections verify TLS by default.

Ingestion, answering, and judging are resumable. Accepted memories must reach `COMPLETED` before the
agent can run. `beam repair` waits for the queue to drain, then recreates only
this index's failed memories with the same content and deterministic IDs. It
never replaces completed memories. An interrupted delete/create can be recovered
by running `beam ingest` again.

Use separate directories for different experiments:

```bash
uv run beam --index runs/index-voyage4 --run runs/react-v1 answer
uv run beam --run runs/react-v1 judge
uv run beam --run runs/react-v1 report
uv run beam --run runs/react-v1 export results/react-v1
```

For the second reader, set `BEAM_READER_MODEL=gpt-5.4-mini` and use a new run
directory. `BEAM_READER_ID` must identify a proxy registration whose credentials
can access that model. The harness uses `reasoning_effort=none` for this model's
Chat Completions function calls; this condition is recorded in the protocol.

If provider throttling leaves answer errors after a pass, retry only those answers
with paced retrieval, then rerun grading. Successful answers stay fixed:

```bash
uv run python tools/retry_errors.py --run runs/react-v1
uv run beam --run runs/react-v1 judge
```

The helper reads the original model/settings from the frozen run, records each
retry, and retains failed attempt files. Let the original answer pass finish first.

`--conversations 1,2` selects a subset and must appear before the command. Label
such results as a subset; the report includes its exact denominator. A changed
index configuration requires a new index directory. Changing the reader, prompts,
or agent settings requires a new run directory. The protocol is frozen before
the first answer and includes model descriptions, settings, and source hashes.

## How it works

1. `prepare` reads only the canonical `chat` field, preserving chronological message
   IDs and explicit chat dates. It stores questions/rubrics in separate files.
2. `ingest` packs adjacent messages into approximately 48,000-character memories
   in one space per conversation. GoodMem creates 3,200-character chunks with
   400-character overlap, embeds them, and stores the index.
3. On every `search_memory` call, GoodMem retrieves 128 dense candidates and
   reranks them to 12 chunks. `read_messages` can inspect up to 31 adjacent
   original messages from an immutable local copy of that same conversation.
4. The reader chooses its next tool call after observing the previous result.
   By default it has up to 16 tool rounds, then a final answer opportunity.
   It can finish earlier. The limits are experiment settings, not BEAM restrictions.
5. The judge scores the frozen answers. Every decision, tool result, stop reason,
   provider token count, and per-question judgment is checkpointed under `runs/`.

The model never receives the question category, reference answer, rubric, source
annotations, user profile, or generation plans. Retrieval is scoped to the
question's conversation. It has no shell, internet, or cross-question memory tool.

The active context is bounded by characters; if necessary, old tool-result bodies
are removed while retaining tool protocol and the question. The full trace stays
on disk. Compaction and budget exhaustion are counted in answer records.

## Scoring and audit

Nine abilities use the official nugget prompt with scores of 0, 0.5, or 1.
Event ordering follows the released greedy event alignment and normalized
Kendall tau calculation. The upstream reporting script uses `tau_norm`; the
underlying scorer also computes `tau_norm × F1`. **Both overall variants are
reported**. This distinction matters when events are omitted.

Exports include answers, rubrics, raw judge outputs, retrieval queries and evidence
identifiers, dataset checksums, and the frozen configuration. Full source excerpts
remain in ignored local traces. Missing answers, service errors, and ungraded
questions prevent a complete overall score from being reported.

```bash
uv run pytest -q
uv run ruff check src tests tools
uv run python tools/verify_results.py results/gpt-4.1-mini results/gpt-5.4-mini
```

The tests exercise iterative tool use, context limits, corpus ordering, read-tool
isolation, retrieval stream failures, and event-ordering edge cases without network
access. Live model behavior is validated separately by the recorded run.

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
