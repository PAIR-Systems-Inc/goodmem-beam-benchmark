# Published BEAM-10M results

Checked on 7 November 2026. This page collects results for the **10M split**;
100K and 1M scores are not included. The scores below were published by their
authors or by the named evaluator. We checked the sources but did not rerun
these systems.

GoodMem's **64.61%** sits between Mastra's **66.4%** and Hindsight's **64.08%**
when sorting the published headline scores. Five configurations report higher
numbers. Those runs used different readers, judges, prompts, and scoring rules,
so their order does not establish how the memory backends would compare under
one protocol.

## Reports from system authors

These reports cover the 10M split, which contains 200 questions. The Qwen paper
identifies the split but does not state its question count. A reader produces
answers; a judge grades them. These columns do not list models used to build the
memory.

| System or configuration | Reported score | Reader | Judge | Source and details |
| --- | ---: | --- | --- | --- |
| past.dev | 85.03% | GPT-5.6 Luna | GPT-5.6 Luna | [Report][past]; question annotations enter the answer prompt |
| Stash | 72.56% | Kimi K3 | Gemini 3 Flash Preview | [Report][stash]; event ordering uses normalized tau × F1 |
| Exabase M-1 | 68.0% | Gemini 3 Flash | Gemini 3 Flash, per report | [Report][exabase]; category-specific prompts |
| Qwen-Planner-Agent 27B + Persistent Memory Manager | 67.24% | Qwen-Planner-Model 27B | Not identified in the evaluation section | [Paper, Table 3][qwen]; Mem0 rubric-scoring protocol |
| Mastra Observational Memory | 66.4% | Claude Opus 5.5 | Gemini 3 Flash | [Report][mastra]; mean of two runs |
| **GoodMem** | **64.61%** | **GPT-6.1 Sol, medium** | **GPT-4.1-mini** | [Results](../RESULTS.md); **62.56%** with event F1 |
| Hindsight | 64.08% | Gemini 3.1 Pro Preview | Gemini 2.5 Flash Lite | [Published run artifact][hindsight]; exact aggregate 64.079772% |
| krusch-context-mcp v2 | 58.71% / 48.18% | GPT-OSS-120B | Hermes 3 / Qwen2.5:32B | [Report][krusch]; score depends on which judge is used |
| Mnemon | 51.2% / 48.8% | GPT-4.1-mini | GPT-4.1-mini / DeepSeek V4.1 Flash | [Paper, Table 4][mnemon]; alternate judge shown alongside the standard result |
| MnemosyneOS | 49.2% | Gemini 2.5 Pro | GPT-4.1-mini | [Run summary][mnemosyne] |
| Mem0 | 48.60% | GPT-5 | GPT-5 | [Published run artifact][mem0]; top-200 retrieval, mean rubric score |
| Honcho | 40.9% | Claude Haiku 4.5 | Not specified on the current results page | [Current results][honcho]; the earlier December 2025 report was 40.6% |
| HivemindOS Shared Brain | 37.04% | GPT-5.4-mini | GPT-5.4-mini | [Report][hivemind]; top-50 retrieval, mean rubric score |

## Runs published by OmniMemEval

[OmniMemEval's pinned results][omni] provide another comparison across memory
backends. Its BEAM runs use GPT-4.1-mini as reader and GPT-4o-mini as judge and
report nugget scores. GoodMem has not been evaluated in that harness. These
numbers should therefore stay separate from our run and the providers' own
reports above.

| Backend | Published 10M score |
| --- | ---: |
| LCM-X | 63.32% |
| Hindsight | 59.75% |
| Viking | 58.14% |
| mem9 | 57.30% |
| MemOS | 56.75% |
| Graphiti / Zep | 56.11% |
| Cognee | 56.02% |
| Supermemory | 52.49% |
| Letta | 52.30% |
| MemMachine | 51.90% |
| EverOS | 47.73% |
| Mem0 | 43.33% |

Each row covers 200 questions. LCM-X is a single contributor run; its reader was
routed through OpenRouter, and the dated GPT-4.1-mini snapshot was not confirmed.
See its [run notes][lcmx]. The spread shown with scores in the source is a
per-question standard deviation, not a confidence interval.

## Differences that affect the comparison

**Event scoring is not uniform.** Our primary score uses normalized Kendall tau
for the 20 event-ordering questions, following the upstream reporting script.
Multiplying those question scores by event F1 gives our secondary overall score
of **62.56%**. F1 accounts for both missing and extra events. Stash explicitly
uses that product in its headline score; Mem0 uses rubric grading. Mastra also
reports a separate strict rubric metric, which is a different calculation.

**Some answer prompts include question annotations.** past.dev documents this
and publishes the [prompt builder][past-prompts]: it supplies fields such as the
preference being tested, event topics, and summary requirements to the reader.
Its README attributes those prompts to Exabase. Our reader receives the question
text and retrieved conversation evidence, with no category, rubric, or reference
answer. This is a difference in the information available to the answerer; the
published scores alone do not measure its effect.

**Reader and judge choice matter.** The two krusch and Mnemon scores show how
changing a judge changes the reported result. Our 95% interval of 61.08–68.16%
measures variation across ten conversations. It does not account for different
judges or establish a significant lead over Hindsight or a gap behind Mastra.

For comparisons over time, the Hindsight, Mem0, past.dev prompt, and OmniMemEval
links below pin the revisions inspected. Other links are dated papers or live
publisher pages and may change. This is a survey of verified public sources,
not an exhaustive registry of every BEAM experiment.

[past]: https://past.dev/benchmarks
[past-prompts]: https://github.com/pastdotdev/benchmarks/blob/a712cd993e249cadd214168c3c3ee674f89bd5e4/beam/exabase_prompts.py
[stash]: https://www.joinstash.ai/blog/state-of-the-art-beam-10m
[exabase]: https://exabase.io/research/exabase-achieves-state-of-the-art-on-beam-benchmark
[qwen]: https://arxiv.org/html/2609.29892v1#S3.SS3
[mastra]: https://mastra.ai/research/observational-memory-beam
[hindsight]: https://github.com/vectorize-io/agent-memory-benchmark/blob/28b644bd4dd653c0207c7507280439390c829155/outputs/beam/hindsight/single-query/10m.json.gz
[krusch]: https://krusch.dev/articles/iclr-2026-beam-closed-system-benchmark
[mnemon]: https://arxiv.org/html/2609.36059v1#S6.SS4
[mnemosyne]: https://mnemosyne-os.github.io/MnemosyneOS---benchmarks/beam-2026-09/SUMMARY.html
[mem0]: https://github.com/mem0ai/memory-benchmarks/blob/4b61c5d31b9c668a12b4f5e78064248a02c82d2b/results/platform/beam_10m_results.json
[honcho]: https://honcho.dev/evals/
[hivemind]: https://hivemindos.app/docs/for-users/features/shared-brain-benchmarks
[omni]: https://github.com/MemTensor/OmniMemEval/blob/7c79d46e31dd85f3da2509b430bba9406656dd28/docs/user_memory/results.md#beam
[lcmx]: https://github.com/MemTensor/OmniMemEval/blob/7c79d46e31dd85f3da2509b430bba9406656dd28/docs/user_memory/lcmx.md#beam-10m-run-2026-10-22
