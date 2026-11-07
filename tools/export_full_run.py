"""Export a complete run, operational counts, and a paired baseline comparison.

This writes local artifacts only. --verify-index additionally reads the local
GoodMem instance and checks readiness plus the stored ingestion metadata digests.
"""

import argparse
import gzip
import json
import shutil
import subprocess
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from goodmem_beam.client import GoodMem
from goodmem_beam.config import Connection, load_env
from goodmem_beam.io import digest, read_json, write_json
from goodmem_beam.report import report


def verify_index(prepared, index_dir, export):
    load_env(Path(".env"))
    index = read_json(index_dir / "index.json")
    client = GoodMem(Connection.from_env())
    rows = []
    try:
        for conv in index["identity"]["conversations"]:
            cid = conv["id"]
            space = index["spaces"][cid]
            blocks = read_json(prepared / f"{cid}.blocks.json")
            assert digest(blocks) == conv["blocks_sha256"]
            memories = client.memories(space)
            assert len(memories) == len(blocks) == conv["memory_blocks"]
            by_id = {m["memoryId"]: m for m in memories}
            assert len(by_id) == len(blocks)
            for i, block in enumerate(blocks):
                memory = by_id[str(uuid.uuid5(uuid.UUID(space), str(i)))]
                assert memory["processingStatus"] == "COMPLETED"
                meta = memory["metadata"]
                assert meta["sha256"] == digest(block)
                assert meta["block"] == i and meta["conversation"] == cid
                assert (meta["start"], meta["end"]) == (block["start"], block["end"])
            rows.append({"conversation_id": cid, "expected_memories": len(blocks),
                         "statuses": dict(Counter(m["processingStatus"] for m in memories)),
                         "matching_ingestion_metadata_digests": len(blocks)})
        result = {"verified_at": datetime.now(UTC).isoformat(),
                  "index_fingerprint": index["fingerprint"], "conversations": rows,
                  "expected_memories": sum(r["expected_memories"] for r in rows),
                  "all_completed": True, "server": client.get("/v1/system/info"),
                  "scope": "Readiness, deterministic IDs, source preparation hashes and stored ingestion metadata; no re-embedding or mutations"}
        write_json(export / "index-verification.json", result)
    finally:
        client.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--baseline", type=Path, default=Path("results/gpt-5.4-mini"))
    p.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    p.add_argument("--index", type=Path, default=Path("runs/index-voyage4"))
    p.add_argument("--verify-index", action="store_true")
    args = p.parse_args()
    args.source_commit = subprocess.check_output(
        ["git", "rev-parse", args.source_commit], text=True).strip()
    protocol = read_json(args.run / "protocol.json")
    committed = {}
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", args.source_commit,
                                     "src/goodmem_beam"], text=True).splitlines()
    for path in files:
        if path.endswith(".py"):
            committed[path.removeprefix("src/goodmem_beam/")] = subprocess.check_output(
                ["git", "show", f"{args.source_commit}:{path}"], text=True)
    assert digest(committed) == protocol["source_sha256"], "Source commit does not match the frozen package"
    source_dir = Path("src/goodmem_beam")
    current = {str(path.relative_to(source_dir)): path.read_text()
               for path in source_dir.rglob("*.py")}
    assert digest(current) == protocol["source_sha256"], "Current package differs from the frozen run"
    summary = report(args.prepared, args.run)
    if not summary["complete"]:
        raise ValueError("Refusing a full-result export for an incomplete run")
    report(args.prepared, args.run, args.export)
    if (args.run / "batches").exists():
        shutil.copytree(args.run / "batches", args.export / "batches", dirs_exist_ok=True)
    with gzip.open(args.export / "outcomes.jsonl.gz", "rt") as f:
        outcomes = [json.loads(line) for line in f]
    reader_models, tool_errors, usage = Counter(), Counter(), Counter()
    trace_steps, payloads = [], 0
    for row in outcomes:
        path = args.run / "traces" / (row["id"].replace(":", "_") + ".jsonl")
        steps = [json.loads(line) for line in path.read_text().splitlines()]
        trace_steps.append(len(steps))
        for step in steps:
            reader_models[step["model"]] += 1
            for observation in step["observations"]:
                assert digest(observation["result"]) == observation["sha256"]
                payloads += 1
                error = observation["result"].get("tool_input_error")
                if error:
                    tool_errors[error] += 1
        usage.update(row["answer"]["usage"])
    judge_models = Counter(read_json(path)["model"] for path in (args.run / "judge-cache").glob("*.json"))
    normalized = [{"id": row["id"], "rubric_index": i, "score": n["score"],
                   "raw_sha256": digest(n["raw"])} for row in outcomes
                  for i, n in enumerate(row["judgment"].get("nuggets", []))
                  if n.get("json_escape_normalized")]
    operations = {
        "answer_source_commit": args.source_commit, "source_commit_verified": True,
        "reader_response_models": dict(reader_models), "judge_response_models": dict(judge_models),
        "final_answer_errors": 0, "final_judgment_errors": 0,
        "saved_interrupted_trace_attempts": len(list((args.run / "traces").glob("*.attempt-*.jsonl"))),
        "internal_http_retries": "not instrumented",
        "search_count_distribution": dict(sorted(Counter(r["answer"]["searches"] for r in outcomes).items())),
        "read_count_distribution": dict(sorted(Counter(r["answer"]["reads"] for r in outcomes).items())),
        "questions_with_multiple_searches": sum(r["answer"]["searches"] > 1 for r in outcomes),
        "questions_with_reads": sum(r["answer"]["reads"] > 0 for r in outcomes),
        "total_reads": sum(r["answer"]["reads"] for r in outcomes),
        "context_compactions": sum(r["answer"]["context_compactions"] for r in outcomes),
        "stop_reasons": dict(Counter(r["answer"]["stop_reason"] for r in outcomes)),
        "recoverable_tool_input_errors": dict(tool_errors), "max_model_calls_used": max(trace_steps),
        "normalized_nuggets": normalized,
        "undefined_event_tau_questions": [r["id"] for r in outcomes
                                           if r["judgment"].get("event_metrics", {}).get("undefined_tau")],
        "reader_usage_successful_attempts": dict(usage), "tool_payload_hashes_verified": payloads,
        "usage_scope": "Successful answer trajectories only; excludes development runs, probes, diagnostics and judging; not a billing estimate",
    }
    write_json(args.export / "operations.json", operations)
    baseline = read_json(args.baseline / "summary.json")
    assert read_json(args.export / "dataset-manifest.json") == read_json(args.baseline / "dataset-manifest.json")
    assert baseline["complete"] and baseline["expected_questions"] == summary["expected_questions"]
    assert set(baseline["per_conversation"]) == set(summary["per_conversation"])
    deltas = np.array([summary["per_conversation"][cid] - baseline["per_conversation"][cid]
                       for cid in protocol["conversations"]])
    rng = np.random.default_rng(20261107)
    sampled = rng.choice(deltas, size=(10000, len(deltas)), replace=True).mean(axis=1)
    comparison = {"comparison": f"{args.export.name} minus {args.baseline.name}",
                  "primary_score_difference": summary["primary_score"] - baseline["primary_score"],
                  "paired_conversation_bootstrap_95ci": [float(x) for x in np.quantile(sampled, [.025, .975])],
                  "bootstrap_samples": 10000, "bootstrap_seed": 20261107,
                  "conversations": len(deltas), "questions_per_method": summary["expected_questions"],
                  "scope": "Reader and harness both differ; development-tuned on BEAM, not held-out validation"}
    write_json(args.export / "comparison.json", comparison)
    if args.verify_index:
        verify_index(args.prepared, args.index, args.export)
    print(json.dumps({"score": summary["primary_score"], "comparison": comparison,
                      "operations": operations}, indent=2))


if __name__ == "__main__":
    main()
