"""Answer and grade exactly one balanced batch, then stop for inspection.

Changing source/configuration requires a new run directory. Never combine scores
from different protocols. The question ordering uses no gold answers or scores.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from goodmem_beam.client import GoodMem
from goodmem_beam.config import Connection, load_env, models, settings
from goodmem_beam.io import digest, read_json, write_json
from goodmem_beam.report import report
from goodmem_beam.run import all_questions, filename, run_answers, run_judge


def batch_order(questions, ids):
    by_conversation = {cid: [q for q in questions if q["conversation_id"] == cid] for cid in ids}
    lengths = {len(qs) for qs in by_conversation.values()}
    if len(lengths) != 1:
        raise ValueError("Balanced order requires equal question counts per conversation")
    count = lengths.pop()
    return [by_conversation[cid][(round_no + 2 * i) % count]
            for round_no in range(count) for i, cid in enumerate(ids)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True, help="Zero-based batch index")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--baseline", type=Path, default=Path("runs/react-gpt54-mini"))
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--index", type=Path, default=Path("runs/index-voyage4"))
    args = parser.parse_args()
    cfg = settings(args.config)
    ids = cfg["dataset"]["conversations"]
    ordered = batch_order(all_questions(args.prepared, ids), ids)
    selected = ordered[args.batch * args.batch_size:(args.batch + 1) * args.batch_size]
    if args.batch < 0 or args.batch_size < 1 or not selected:
        raise ValueError("Invalid batch selection")
    plan = {"question_ids": [q["id"] for q in ordered], "batch_size": args.batch_size,
            "ordering": "Conversation round-robin; question rotation by twice conversation index",
            "helper_sha256": digest(Path(__file__).read_text())}
    path = args.run / "batches" / "plan.json"
    if path.exists() and read_json(path) != plan:
        raise ValueError("Batch plan changed")
    write_json(path, plan)
    selected_ids = {q["id"] for q in selected}
    load_env(Path(".env"))
    client = GoodMem(Connection.from_env())
    try:
        run_answers(client, models(), cfg, args.prepared, args.index, args.run, ids,
                    question_ids=selected_ids)
        run_judge(client, models(), cfg, args.prepared, args.run, ids, question_ids=selected_ids)
        report(args.prepared, args.run)
    finally:
        client.close()
    rows, errors, by_ability = [], [], defaultdict(list)
    for q in selected:
        judgment = read_json(args.run / "judgments" / filename(q)) if (args.run / "judgments" / filename(q)).exists() else {}
        if judgment.get("status") != "ok":
            errors.append(q["id"])
            continue
        base = read_json(args.baseline / "judgments" / filename(q))
        row = {"id": q["id"], "ability": q["ability"], "score": judgment["score"],
               "baseline_score": base["score"], "delta": judgment["score"] - base["score"]}
        rows.append(row)
        by_ability[q["ability"]].append(row)
    complete = len(rows) == len(selected)
    result = {"batch": args.batch, "expected": len(selected), "scored": len(rows),
              "complete": complete, "errors": errors,
              "question_ids": [q["id"] for q in selected],
              "score": mean(r["score"] for r in rows) if complete else None,
              "baseline_score": mean(r["baseline_score"] for r in rows) if complete else None,
              "paired_delta": mean(r["delta"] for r in rows) if complete else None,
              "per_ability": {a: {"count": len(rs), "score": mean(r["score"] for r in rs),
                                  "baseline": mean(r["baseline_score"] for r in rs)}
                              for a, rs in by_ability.items()}, "rows": rows}
    write_json(args.run / "batches" / f"batch-{args.batch:02d}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2), flush=True)
    if not complete:
        raise RuntimeError("Batch is incomplete; resolve errors before continuing")


if __name__ == "__main__":
    main()
