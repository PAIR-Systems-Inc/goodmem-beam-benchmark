"""Retry only missing/error answers, without changing the frozen inference protocol.

Use after the initial answer pass. Transport pacing reduces provider throttles;
completed answers are never replaced or selected by their score.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from goodmem_beam.agent import answer
from goodmem_beam.client import GoodMem
from goodmem_beam.config import Connection, load_env
from goodmem_beam.dataset import ChatReader
from goodmem_beam.io import append_json, digest, read_json, write_json
from goodmem_beam.run import all_questions, filename, freeze


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--index", type=Path, default=Path("runs/index-voyage4"))
    p.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--attempts", type=int, default=3)
    args = p.parse_args()
    protocol = read_json(args.run / "protocol.json")
    cfg, model_ids = protocol["config"], protocol["model_ids"]
    index = read_json(args.index / "index.json")
    qs = all_questions(args.prepared, protocol["conversations"])
    load_env(Path(".env"))
    client = GoodMem(Connection.from_env())
    freeze(client, model_ids, cfg, index, args.run, protocol["conversations"])
    previous = [0.0]

    def pace(request):
        if request.url.path.endswith("/memories:retrieve"):
            time.sleep(max(0.0, previous[0] + args.interval - time.monotonic()))
            previous[0] = time.monotonic()

    client.http.event_hooks["request"].append(pace)
    append_json(args.run / "retry-history.jsonl", {
        "event": "start", "transport_interval_seconds": args.interval,
        "helper_sha256": digest(Path(__file__).read_text()), "time": time.time(),
    })
    try:
        for attempt in range(1, args.attempts + 1):
            pending = [q for q in qs if not (args.run / "answers" / filename(q)).exists()
                       or read_json(args.run / "answers" / filename(q)).get("status") != "ok"]
            if not pending:
                print("All planned answers are complete.", flush=True)
                return
            print(f"Retry pass {attempt}: {len(pending)} answers", flush=True)
            for q in pending:
                path = args.run / "answers" / filename(q)
                trace = args.run / "traces" / (filename(q) + "l")
                stamp = str(time.time_ns())
                if path.exists():
                    backup = args.run / "failed-attempts" / (filename(q) + f".{stamp}")
                    backup.parent.mkdir(exist_ok=True)
                    shutil.copyfile(path, backup)
                if trace.exists():
                    trace.rename(trace.with_suffix(f".attempt-{stamp}.jsonl"))
                try:
                    result = answer(client, q["question"], index["spaces"][q["conversation_id"]],
                                    ChatReader(args.prepared / f"{q['conversation_id']}.sqlite"),
                                    model_ids, cfg["agent"], cfg["retrieval"],
                                    on_step=lambda step, t=trace: append_json(t, step))
                    result.pop("trace")
                    result.update(id=q["id"], conversation_id=q["conversation_id"],
                                  ability=q["ability"], question=q["question"], status="ok")
                except Exception as e:  # noqa: BLE001 - retain recoverable per-question failures
                    result = {"id": q["id"], "status": "error", "error": str(e)}
                write_json(path, result)
                append_json(args.run / "retry-history.jsonl", {
                    "id": q["id"], "attempt": attempt, "status": result["status"], "time": time.time(),
                })
                print("retry", q["id"], result["status"], flush=True)
            if attempt < args.attempts:
                time.sleep(30)
        remaining = [q["id"] for q in qs
                     if read_json(args.run / "answers" / filename(q)).get("status") != "ok"]
        if remaining:
            raise RuntimeError(f"Answers remain incomplete: {json.dumps(remaining)}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
