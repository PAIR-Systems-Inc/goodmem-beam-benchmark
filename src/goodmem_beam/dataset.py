"""Read the canonical 10M chat once; never index plans, profiles or answer rubrics."""

from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq

from .io import digest, write_json

REVISION = "9b2096193fe74e2837e4713e483351e19817773c"
UPSTREAM_COMMIT = "b2da22eac88bb0874c64665f13457eb99835774a"
ABILITIES = (
    "abstention", "contradiction_resolution", "event_ordering", "information_extraction",
    "instruction_following", "knowledge_update", "multi_session_reasoning",
    "preference_following", "summarization", "temporal_reasoning",
)


def rows(data_dir: Path, ids: list[str] | None = None):
    seen = set()
    for path in sorted(data_dir.glob("10M-*.parquet")):
        # Read no redundant plans, profiles, or generation scaffolding.
        f = pq.ParquetFile(path)
        for batch in f.iter_batches(batch_size=1, columns=[
            "conversation_id", "conversation_seed", "chat", "probing_questions"
        ]):
            row = batch.to_pylist()[0]
            cid = str(row["conversation_id"])
            if ids is None or cid in ids:
                if cid in seen:
                    raise ValueError(f"Duplicate conversation {cid}")
                seen.add(cid)
                yield row
    if ids and seen != set(ids):
        raise ValueError(f"Missing conversations: {set(ids) - seen}")


def messages(chat):
    last = -1
    for plan in chat:
        populated = [(k, v) for k, v in plan.items() if v is not None]
        if len(populated) != 1:
            raise ValueError("Expected one populated canonical plan per element")
        plan_name, batches = populated[0]
        for batch in batches:
            stamp = batch.get("time_anchor") or ""
            for turn in batch["turns"]:
                for m in turn:
                    mid = int(m["id"])
                    if mid != last + 1:
                        raise ValueError(f"Noncontiguous message IDs {last} -> {mid}")
                    last = mid
                    # Carry only explicit chat dates; never use conversation-plan dates.
                    stamp = m.get("time_anchor") or stamp
                    yield {
                        "id": mid, "role": m["role"], "content": m["content"],
                        "time": stamp, "plan": plan_name,
                    }


def questions(raw, cid):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = ast.literal_eval(raw)
    if set(raw) != set(ABILITIES):
        raise ValueError("Unexpected BEAM ability set")
    result = []
    for ability in ABILITIES:
        if len(raw[ability]) != 2:
            raise ValueError(f"Expected two questions for {ability}")
        for i, q in enumerate(raw[ability]):
            rubric = q["rubric"]
            if not rubric or not all(isinstance(x, str) for x in rubric):
                raise ValueError("Invalid rubric")
            result.append({"id": f"{cid}:{ability}:{i}", "conversation_id": cid,
                           "ability": ability, "question": q["question"], "rubric": rubric})
    return result


def render(m):
    return f"[message {m['id']} | {m['time']} | {m['role']}]\n{m['content']}"


def blocks(msgs, target_chars):
    texts, start, size, end = [], 0, 0, 0
    for m in msgs:
        text = render(m)
        if texts and size + len(text) > target_chars:
            yield {"start": start, "end": end, "text": "\n\n".join(texts)}
            texts, size = [], 0
        if not texts:
            start = m["id"]
        texts.append(text)
        size += len(text) + 2
        end = m["id"]
    if texts:
        yield {"start": start, "end": end, "text": "\n\n".join(texts)}


def prepare(data_dir: Path, ids, memory_chars=48000, count_tokens=True):
    tokenizer = None
    if count_tokens:
        import tiktoken
        tokenizer = tiktoken.get_encoding("o200k_base")
    out = data_dir / "prepared"
    out.mkdir(exist_ok=True)
    manifest = {"dataset": "Mohammadta/BEAM-10M", "revision": REVISION,
                "tokenizer": "o200k_base" if tokenizer else None, "conversations": [],
                "files": {p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
                          for p in sorted(data_dir.glob("10M-*.parquet"))}}
    for row in rows(data_dir, ids):
        cid = str(row["conversation_id"])
        msgs = list(messages(row["chat"]))
        qs = questions(row["probing_questions"], cid)
        # Physically separate chat storage from evaluation-only labels.
        dbpath = out / f"{cid}.sqlite"
        with sqlite3.connect(dbpath) as db:
            db.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, text TEXT)")
            db.execute("DELETE FROM messages")
            db.executemany("INSERT INTO messages VALUES (?,?)", [(m["id"], render(m)) for m in msgs])
        b = list(blocks(msgs, memory_chars))
        write_json(out / f"{cid}.blocks.json", b)
        write_json(out / f"{cid}.questions.json", qs)
        tokens = sum(len(tokenizer.encode(m["content"], disallowed_special=()))
                     for m in msgs) if tokenizer else None
        item = {"id": cid, "messages": len(msgs), "memory_blocks": len(b),
                "content_chars": sum(len(m["content"]) for m in msgs),
                "content_tokens": tokens, "chat_sha256": digest(msgs),
                "blocks_sha256": digest(b), "questions_sha256": digest(qs),
                "questions": len(qs), "category": row["conversation_seed"]["category"]}
        manifest["conversations"].append(item)
        write_json(out / "manifest.json", manifest)
        print(json.dumps(item), flush=True)
    return manifest


class ChatReader:
    """Read-only transcript tool. No question labels or generation scaffolding are accessible."""

    def __init__(self, path: Path):
        self.path = path

    def locate_chunk(self, start, end, text):
        """Restore source headers only when the retrieved chunk matches exactly.

        Chunk boundaries can omit a message's header. This does not search the
        corpus: it checks only the already-retrieved parent block's message range.
        """
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as db:
            rows = db.execute("SELECT id,text FROM messages WHERE id BETWEEN ? AND ? ORDER BY id",
                              (start, end)).fetchall()
        block = "\n\n".join(value for _, value in rows)
        needle = text.strip()
        offset = block.find(needle) if needle else -1
        if offset < 0 or block.find(needle, offset + 1) >= 0:
            return {"exact_match": False, "messages": []}
        matched, position = [], 0
        for mid, value in rows:
            if position < offset + len(needle) and position + len(value) > offset:
                matched.append({"message_id": mid, "header": value.split("\n", 1)[0]})
            position += len(value) + 2
        return {"exact_match": True, "messages": matched}

    def read(self, start: int, end: int, char_budget=24000):
        if start < 0 or end < start or end - start > 30:
            raise ValueError("Read a range of at most 31 messages, with nonnegative IDs")
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as db:
            records = db.execute(
                "SELECT id,text FROM messages WHERE id BETWEEN ? AND ? ORDER BY id", (start, end)
            ).fetchall()
        result, used = [], 0
        for mid, text in records:
            remaining = char_budget - used
            if remaining <= 0:
                break
            result.append({"message_id": mid, "text": text[:remaining],
                           "truncated": len(text) > remaining})
            used += len(text)
        return {"messages": result, "requested_end": end,
                "delivered_end": result[-1]["message_id"] if result else None}
