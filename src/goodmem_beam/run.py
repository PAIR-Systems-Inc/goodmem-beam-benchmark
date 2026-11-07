"""Freeze configuration, isolate questions, checkpoint answers, then grade separately."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .agent import SYSTEM, TOOLS, answer
from .dataset import ChatReader
from .io import append_json, digest, read_json, write_json
from .judge import NUGGET_PROMPT, Judge


def all_questions(prepared, ids):
    manifest = read_json(prepared / "manifest.json")
    expected = {c["id"]: c["questions_sha256"] for c in manifest["conversations"]}
    result = []
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate requested conversation IDs")
    for cid in ids:
        qs = read_json(prepared / f"{cid}.questions.json")
        if digest(qs) != expected[cid]:
            raise ValueError("Question-file digest mismatch")
        result.extend(qs)
    return result


def filename(q):
    return q["id"].replace(":", "_") + ".json"


def freeze(client, model_ids, config, index, run_dir, ids):
    if not index.get("ready"):
        raise ValueError("Index is not fully COMPLETED; refusing a partial-history evaluation")
    if set(ids) - set(index["spaces"]):
        raise ValueError("Requested conversations missing from index")
    sizes = {c["id"]: c["memory_blocks"] for c in index["identity"]["conversations"]}
    for cid in ids:
        records = client.memories(index["spaces"][cid])
        if len(records) != sizes[cid] or any(m["processingStatus"] != "COMPLETED" for m in records):
            raise ValueError("Live index is no longer fully searchable")
    model_names = {}
    for key, kind in [("embedder", "embedders"), ("reranker", "rerankers"), ("reader", "llms"), ("judge", "llms")]:
        m = client.get(f"/v1/{kind}/{model_ids[key]}")
        model_names[key] = {k: m.get(k) for k in ["modelIdentifier", "providerType", "displayName"]}
    protocol = {"config": config, "models": model_names, "model_ids": model_ids,
                "index_fingerprint": index["fingerprint"], "conversations": ids,
                "agent_prompt_sha256": digest(SYSTEM), "tools_sha256": digest(TOOLS),
                "judge_prompt_sha256": digest(NUGGET_PROMPT)}
    # Include code identity so resumed runs cannot silently mix implementations.
    source_dir = Path(__file__).parent
    protocol["source_sha256"] = digest({str(p.relative_to(source_dir)): p.read_text()
                                        for p in sorted(source_dir.rglob("*.py"))})
    path = run_dir / "protocol.json"
    if path.exists() and read_json(path) != protocol:
        raise ValueError("Frozen protocol changed; use a new result directory")
    write_json(path, protocol)
    return protocol


def run_answers(client, model_ids, cfg, prepared, index_dir, run_dir, ids):
    index = read_json(index_dir / "index.json")
    freeze(client, model_ids, cfg, index, run_dir, ids)
    questions = all_questions(prepared, ids)

    def one(q):
        path = run_dir / "answers" / filename(q)
        if path.exists() and read_json(path).get("status") == "ok":
            return q["id"], "cached"
        trace_path = run_dir / "traces" / (filename(q) + "l")
        if trace_path.exists():
            trace_path.rename(trace_path.with_suffix(f".attempt-{time.time_ns()}.jsonl"))
        try:
            result = answer(client, q["question"], index["spaces"][q["conversation_id"]],
                            ChatReader(prepared / f"{q['conversation_id']}.sqlite"), model_ids,
                            cfg["agent"], cfg["retrieval"], on_step=lambda s: append_json(trace_path, s))
            result.pop("trace")
            result.update(id=q["id"], conversation_id=q["conversation_id"], ability=q["ability"],
                          question=q["question"], status="ok")
        except Exception as e:  # noqa: BLE001 - preserve explicit per-question failure records
            result = {"id": q["id"], "status": "error", "error": str(e)}
        write_json(path, result)
        return q["id"], result["status"]

    with ThreadPoolExecutor(max_workers=cfg["agent"]["workers"]) as pool:
        for future in as_completed([pool.submit(one, q) for q in questions]):
            print("answer", *future.result(), flush=True)


def run_judge(client, model_ids, cfg, prepared, run_dir, ids):
    protocol = read_json(run_dir / "protocol.json")
    if (any(protocol["model_ids"][k] != model_ids[k] for k in ["judge", "judge_model"])
            or protocol["conversations"] != ids or protocol["config"]["judge"] != cfg["judge"]
            or protocol["judge_prompt_sha256"] != digest(NUGGET_PROMPT)):
        raise ValueError("Judge or selected conversations differ from the frozen run")
    judge = Judge(client, model_ids, run_dir / "judge-cache", cfg["judge"]["max_output_tokens"])

    def one(q):
        answer_path = run_dir / "answers" / filename(q)
        path = run_dir / "judgments" / filename(q)
        if not answer_path.exists():
            return q["id"], "missing_answer"
        a = read_json(answer_path)
        if a.get("status") != "ok":
            return q["id"], "answer_error"
        key = digest({"question": q, "answer": a["answer"]})
        if path.exists() and read_json(path).get("input_sha256") == key and read_json(path).get("status") == "ok":
            return q["id"], "cached"
        try:
            result = judge.evaluate(q, a["answer"])
            result.update(status="ok", input_sha256=key)
        except Exception as e:  # noqa: BLE001 - preserve explicit per-question failure records
            result = {"status": "error", "error": str(e)}
        result.update(id=q["id"], conversation_id=q["conversation_id"], ability=q["ability"])
        write_json(path, result)
        return q["id"], result["status"]

    with ThreadPoolExecutor(max_workers=cfg["judge"]["workers"]) as pool:
        for future in as_completed([pool.submit(one, q) for q in all_questions(prepared, ids)]):
            print("judge", *future.result(), flush=True)
