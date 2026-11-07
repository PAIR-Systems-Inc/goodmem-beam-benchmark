"""Audit the recorded Responses stopping procedure; this does not prove factual correctness."""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def audit(directory):
    protocol = json.loads((directory / "protocol.json").read_text())
    assert protocol["config"]["agent"]["api"] == "responses"
    with gzip.open(directory / "outcomes.jsonl.gz", "rt") as handle:
        outcomes = [json.loads(line) for line in handle]
    grouped = defaultdict(list)
    with gzip.open(directory / "retrieval-traces.jsonl.gz", "rt") as handle:
        for line in handle:
            step = json.loads(line)
            assert "encrypted_content" not in line
            grouped[step["id"]].append(step)
    assert set(grouped) == {row["id"] for row in outcomes}
    counts, model_names, effort_names, reasoning_contexts = Counter(), set(), set(), set()
    for row in outcomes:
        answer = row["answer"]
        steps = grouped[row["id"]]
        assert [s["step"] for s in steps] == list(range(len(steps)))
        queries, evidence = set(), set()
        eligible_draft = None
        verified_after_draft = False
        accepted_answer = None
        usage, rejected = Counter(), 0
        for step in steps:
            model_names.add(step["model"])
            effort = step["reasoning"]["effort"]
            effort_names.add(effort)
            assert effort == protocol["config"]["agent"]["reasoning_effort"]
            reasoning_contexts.add(step["reasoning"]["context"])
            assert step["reasoning"]["context"] == protocol["reader_inference"]["reasoning"]["context"]
            assert step["model"].startswith(protocol["requested_models"]["reader"])
            calls = step["assistant"].get("tool_calls", [])
            assert len(calls) == len(step["observations"])
            raw = step["usage"]
            usage["prompt_tokens"] += raw["input_tokens"]
            usage["completion_tokens"] += raw["output_tokens"]
            usage["reasoning_tokens"] += raw.get("output_tokens_details", {}).get("reasoning_tokens", 0)
            usage["cached_tokens"] += raw.get("input_tokens_details", {}).get("cached_tokens", 0)
            if step.get("stopping_rejected"):
                rejected += 1
            for call, observation in zip(calls, step["observations"], strict=True):
                name, result = observation["tool"], observation["result"]
                assert call["function"]["name"] == name
                assert call["function"]["arguments"] == observation["arguments"]
                args = json.loads(observation["arguments"])
                if "tool_input_error" in result:
                    counts["tool_input_errors"] += 1
                    continue
                counts[name] += 1
                if name == "search_memory":
                    queries.add(" ".join(args["query"].casefold().split()))
                    for hit in result["hits"]:
                        evidence.add(hit["evidence_id"])
                        counts["retrieved_hits"] += 1
                        counts["exact_source_matches"] += hit["source_messages"]["exact_match"]
                elif name == "read_messages":
                    assert 0 <= args["start"] <= args["end"] <= args["start"] + 30
                    for message in result["messages"]:
                        evidence.add(message["evidence_id"])
                    counts["empty_reads"] += not result["messages"]
                if name in {"search_memory", "read_messages"} and eligible_draft is not None:
                    verified_after_draft = True
                if name == "submit_answer":
                    assert args["answer"].strip() and args["coverage"]
                    for fact in args["coverage"]:
                        assert set(fact["evidence_ids"]) <= evidence
                        if fact["status"] == "supported":
                            assert fact["evidence_ids"]
                    if result["accepted"]:
                        assert len(queries) >= protocol["config"]["agent"]["min_distinct_searches"]
                        assert eligible_draft is not None and verified_after_draft
                        assert step["step"] == steps[-1]["step"]
                        accepted_answer = args["answer"]
                    else:
                        rejected += 1
                        if (eligible_draft is None
                                and len(queries) >= protocol["config"]["agent"]["min_distinct_searches"]):
                            eligible_draft = step["step"]
        assert dict(usage) == answer["usage"]
        assert rejected == answer["rejected_submissions"]
        assert len(queries) == answer["distinct_searches"]
        if answer["stop_reason"] == "verified_submission":
            assert accepted_answer == answer["answer"]
        else:
            assert answer["stop_reason"] == "budget_exhausted"
            assert steps[-1]["step"] == protocol["config"]["agent"]["max_steps"]
            assert steps[-1]["assistant"]["content"] == answer["answer"]
        counts[answer["stop_reason"]] += 1
        counts["context_evictions"] += answer["context_compactions"]
    return {"questions": len(outcomes), "observed_models": sorted(model_names),
            "reasoning_contexts": sorted(reasoning_contexts),
            "reasoning_efforts": sorted(effort_names), "procedure_counts": dict(counts),
            "scope": "Procedure and reported usage audit; not an evidence-entailment audit"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
