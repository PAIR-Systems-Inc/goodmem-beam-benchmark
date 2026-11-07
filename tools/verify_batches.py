"""Verify published batch checkpoints against their outcomes and matched baseline."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import mean

from goodmem_beam.judge import event_metrics, parse_json


def read(path):
    return json.loads(path.read_text())


def outcomes(directory):
    with gzip.open(directory / "outcomes.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert len({row["id"] for row in rows}) == len(rows)
    return {row["id"]: row for row in rows}


def verify(directory, baseline):
    records = outcomes(directory)
    plan = read(directory / "batches/plan.json")
    assert len(set(plan["question_ids"])) == len(plan["question_ids"])
    checked = set()
    for path in sorted((directory / "batches").glob("batch-*.json")):
        batch = read(path)
        start = batch["batch"] * plan["batch_size"]
        assert batch["question_ids"] == plan["question_ids"][start:start + plan["batch_size"]]
        assert batch["complete"] and not batch["errors"]
        assert len(batch["rows"]) == batch["expected"] == batch["scored"]
        assert {r["id"] for r in batch["rows"]} == set(batch["question_ids"])
        for row in batch["rows"]:
            assert row["id"] not in checked
            checked.add(row["id"])
            actual = records[row["id"]]
            assert actual["answer"]["status"] == actual["judgment"]["status"] == "ok"
            assert row["score"] == actual["judgment"]["score"]
            assert row["baseline_score"] == baseline[row["id"]]["judgment"]["score"]
            assert row["delta"] == row["score"] - row["baseline_score"]
            judgment = actual["judgment"]
            if actual["ability"] == "event_ordering":
                assert event_metrics(actual["rubric"], judgment["aligned"]) == judgment["event_metrics"]
                assert judgment["score"] == judgment["event_metrics"]["tau_norm"]
            else:
                assert judgment["score"] == mean(n["score"] for n in judgment["nuggets"])
                for nugget in judgment["nuggets"]:
                    assert parse_json(nugget["raw"])[0]["score"] == nugget["score"]
        assert batch["score"] == mean(row["score"] for row in batch["rows"])
        assert batch["baseline_score"] == mean(row["baseline_score"] for row in batch["rows"])
        assert batch["paired_delta"] == mean(row["delta"] for row in batch["rows"])
    assert checked == set(records)
    summary = read(directory / "summary.json")
    assert summary["scored_questions"] == len(records)
    if len(records) < summary["expected_questions"]:
        assert not summary["complete"] and summary["primary_score"] is None
    print(f"{directory}: verified {len(records)} checkpoint questions and matched baseline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("results/gpt-5.4-mini"))
    parser.add_argument("directories", type=Path, nargs="+")
    args = parser.parse_args()
    baseline = outcomes(args.baseline)
    for directory in args.directories:
        verify(directory, baseline)
