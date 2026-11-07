"""Resumable ingestion with deterministic IDs and an exhaustive readiness gate."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .client import GoodMemError
from .io import digest, read_json, write_json


class SubmissionPacer:
    """Conservative input-token pacing; includes a margin for chunk overlap."""

    def __init__(self, tokens_per_minute=3_000_000):
        import tiktoken
        self.encoding = tiktoken.get_encoding("o200k_base")
        self.rate = tokens_per_minute / 60
        self.next_time = time.monotonic()
        self.lock = threading.Lock()

    def wait(self, batch):
        tokens = sum(len(self.encoding.encode(r["originalContent"], disallowed_special=()))
                     for r in batch)
        with self.lock:
            start = max(time.monotonic(), self.next_time)
            self.next_time = start + tokens * 1.4 / self.rate
        time.sleep(max(0, start - time.monotonic()))


def space_body(space_id, name, embedder, config):
    return {
        "spaceId": space_id, "name": name,
        "labels": {"benchmark": "beam-10m", "harness": "goodmem-beam-benchmark"},
        "spaceEmbedders": [{"embedderId": embedder, "defaultRetrievalWeight": 1.0}],
        "defaultChunkingConfig": {"recursive": {
            "chunkSize": config["chunk_chars"], "chunkOverlap": config["overlap_chars"],
            "separators": ["\n\n", "\n", ". ", " ", ""], "keepStrategy": "KEEP_END",
            "lengthMeasurement": "CHARACTER_COUNT",
        }},
    }


def ingest(client, prepared: Path, run_dir: Path, ids, embedder, config):
    started = time.monotonic()
    pacer = SubmissionPacer()
    manifest = read_json(prepared / "manifest.json")
    selected = [c for c in manifest["conversations"] if c["id"] in ids]
    if {c["id"] for c in selected} != set(ids):
        raise ValueError("Prepare all requested conversations first")
    state_path = run_dir / "index.json"
    identity = {"dataset_revision": manifest["revision"], "conversations": selected,
                "embedder": embedder, "config": config}
    fingerprint = digest(identity)
    if state_path.exists():
        state = read_json(state_path)
        if state["fingerprint"] != fingerprint:
            raise ValueError("Index configuration/data changed; use a new index directory")
    else:
        state = {"fingerprint": fingerprint, "identity": identity, "spaces": {},
                 "ready": False, "ingest_seconds": 0}
        write_json(state_path, state)
    # Checkpoints are serialized; only independent network batches run concurrently.
    for conv in selected:
        cid = conv["id"]
        blocks = read_json(prepared / f"{cid}.blocks.json")
        if digest(blocks) != conv["blocks_sha256"]:
            raise ValueError("Prepared corpus digest mismatch")
        space = str(uuid.uuid5(uuid.NAMESPACE_URL, f"beam:{run_dir.name}:{fingerprint}:{cid}"))
        if cid not in state["spaces"]:
            body = space_body(space, f"beam-10m-{run_dir.name}-{cid}", embedder, config)
            try:
                client.post("/v1/spaces", body)
            except GoodMemError as e:
                if "HTTP 409" not in str(e):
                    raise
                found = client.get(f"/v1/spaces/{space}")
                if found["name"] != body["name"]:
                    raise GoodMemError("Existing space identity mismatch")
            state["spaces"][cid] = space
            write_json(state_path, state)
        existing = {m["memoryId"]: m for m in client.memories(space)}
        requests = []
        for i, block in enumerate(blocks):
            mid = str(uuid.uuid5(uuid.UUID(space), str(i)))
            content_hash = digest(block)
            if mid in existing:
                if existing[mid].get("metadata", {}).get("sha256") != content_hash:
                    raise GoodMemError("Existing memory content identity mismatch")
                continue
            requests.append({"memoryId": mid, "spaceId": space, "contentType": "text/plain",
                             "originalContent": block["text"],
                             "metadata": {"conversation": cid, "block": i, "start": block["start"],
                                          "end": block["end"], "sha256": content_hash}})
        batch_size = config["batch_size"]
        batches = [requests[i:i+batch_size] for i in range(0, len(requests), batch_size)]

        def submit(batch):
            pacer.wait(batch)
            response = client.post("/v1/memories:batchCreate", {"requests": batch})
            results = response.get("results", [])
            if len(results) != len(batch):
                raise GoodMemError("Partial batch response")
            for req, result in zip(batch, results, strict=True):
                memory = result.get("memory")
                if memory is None:
                    status = result.get("status", {})
                    if status.get("code") != 6:  # ALREADY_EXISTS on replay after timeout
                        raise GoodMemError(f"Memory create failed: {status.get('code')}")
                    memory = client.get(f"/v1/memories/{req['memoryId']}")
                if (memory["memoryId"] != req["memoryId"] or
                        memory.get("metadata", {}).get("sha256") != req["metadata"]["sha256"]):
                    raise GoodMemError("Memory write identity mismatch")
            return len(batch)

        done = len(existing)
        with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
            for future in as_completed([pool.submit(submit, batch) for batch in batches]):
                done += future.result()
                print(f"ingest conversation={cid} accepted={done}/{len(blocks)}", flush=True)
        state["ingest_seconds"] += time.monotonic() - started
        started = time.monotonic()
        write_json(state_path, state)
    deadline = time.monotonic() + config["ready_timeout"]
    while True:
        total, complete, status_counts = 0, 0, {}
        for conv in selected:
            records = client.memories(state["spaces"][conv["id"]])
            expected = conv["memory_blocks"]
            if len(records) != expected:
                raise GoodMemError(f"Index has {len(records)} memories; expected {expected}")
            counts = Counter(m["processingStatus"] for m in records)
            status_counts[conv["id"]] = dict(counts)
            total += expected
            complete += counts["COMPLETED"]
        state["statuses"] = status_counts
        state["ready"] = complete == total
        write_json(state_path, state)
        print(json.dumps({"ready": complete, "expected": total, "statuses": status_counts}), flush=True)
        if any(c.get("FAILED", 0) for c in status_counts.values()):
            raise GoodMemError("Some memories FAILED processing; the index is not eligible for evaluation")
        if state["ready"]:
            state["ingest_seconds"] += time.monotonic() - started
            write_json(state_path, state)
            return state
        if time.monotonic() >= deadline:
            raise GoodMemError("Index readiness timeout; resume ingestion after checking GoodMem")
        time.sleep(15)


def repair_failed(client, prepared: Path, run_dir: Path, *, rounds=3):
    """Recreate only this index's FAILED memories, after queued work has drained.

    GoodMem currently has no public reprocess endpoint. A deterministic ID and
    exact content digest fence every delete/recreate. Completed memories are untouched.
    """
    from .io import append_json

    path = run_dir / "index.json"
    state = read_json(path)
    started = time.monotonic()
    deadline = started + 7200
    pacer = SubmissionPacer()
    for attempt in range(rounds + 1):
        while True:
            failed, active, complete, total = [], 0, 0, 0
            for conv in state["identity"]["conversations"]:
                cid, expected = conv["id"], conv["memory_blocks"]
                sid = state["spaces"][cid]
                records = client.memories(sid)
                if len(records) != expected:
                    raise GoodMemError("Missing memory: resume ingest before repair")
                for record in records:
                    status = record["processingStatus"]
                    if status == "FAILED":
                        failed.append((cid, sid, record))
                    elif status == "COMPLETED":
                        complete += 1
                    else:
                        active += 1
                total += expected
            state.update(ready=complete == total, completed=complete, failed=len(failed))
            write_json(path, state)
            print(f"index completed={complete}/{total} active={active} failed={len(failed)} retry_round={attempt}", flush=True)
            if active == 0:
                break
            if time.monotonic() > deadline:
                raise GoodMemError("Repair wait timed out")
            time.sleep(20)
        if not failed:
            state["repair_seconds"] = state.get("repair_seconds", 0) + time.monotonic() - started
            write_json(path, state)
            return state
        if attempt == rounds:
            raise GoodMemError("Failed memories remain after bounded repair attempts")
        # Provider rolling windows must clear before resubmission starts.
        time.sleep(30)
        cached_blocks = {}
        for cid, sid, record in failed:
            if cid not in cached_blocks:
                cached_blocks[cid] = read_json(prepared / f"{cid}.blocks.json")
            i = int(record["metadata"]["block"])
            block = cached_blocks[cid][i]
            mid = str(uuid.uuid5(uuid.UUID(sid), str(i)))
            if record["memoryId"] != mid or record["metadata"]["sha256"] != digest(block):
                raise GoodMemError("Repair ownership/content verification failed")
            latest = client.get(f"/v1/memories/{mid}")
            if latest["processingStatus"] != "FAILED" or latest["spaceId"] != sid:
                raise GoodMemError("Memory state changed during repair")
            req = {"memoryId": mid, "spaceId": sid, "originalContent": block["text"],
                   "contentType": "text/plain", "metadata": record["metadata"]}
            pacer.wait([req])
            client.request("DELETE", f"/v1/memories/{mid}")
            result = client.post("/v1/memories:batchCreate", {"requests": [req]})
            if not result.get("results", [{}])[0].get("memory"):
                raise GoodMemError("Repair create failed; resume ingest to restore missing memory")
            append_json(run_dir / "repairs.jsonl", {"conversation": cid, "block": i,
                                                     "round": attempt + 1, "reason": "FAILED"})
