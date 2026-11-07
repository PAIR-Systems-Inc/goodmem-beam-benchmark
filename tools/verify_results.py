"""Recompute published scores and tool counts without a server or credentials."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

from goodmem_beam.dataset import ABILITIES


def verify(directory):
    summary = json.loads((directory / "summary.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    with gzip.open(directory / "outcomes.jsonl.gz", "rt") as handle:
        rows = [json.loads(line) for line in handle]
    with gzip.open(directory / "retrieval-traces.jsonl.gz", "rt") as handle:
        traces = [json.loads(line) for line in handle]
    expected = {(cid, ability): 2 for cid in protocol["conversations"] for ability in ABILITIES}
    assert Counter((r["conversation_id"], r["ability"]) for r in rows) == expected
    assert len({r["id"] for r in rows}) == len(rows) == summary["expected_questions"]
    assert summary["complete"] and not summary["errors_or_missing"]
    assert summary["scored_questions"] == len(rows)
    assert all(r["answer"]["status"] == r["judgment"]["status"] == "ok" for r in rows)
    tools = Counter((step["id"], observation["tool"])
                    for step in traces for observation in step["observations"]
                    if "tool_input_error" not in observation["result"])
    for row in rows:
        assert tools[row["id"], "search_memory"] == row["answer"]["searches"]
        assert tools[row["id"], "read_messages"] == row["answer"]["reads"]
    for field, metric in [("per_ability", "score"),
                          ("per_ability_recall_adjusted", "recall_adjusted_score")]:
        expected_scores = {a: mean(r["judgment"][metric] for r in rows if r["ability"] == a)
                           for a in ABILITIES}
        assert summary[field] == expected_scores
    assert summary["primary_score"] == mean(summary["per_ability"].values())
    assert summary["recall_adjusted_score"] == mean(summary["per_ability_recall_adjusted"].values())
    per_conversation = {cid: mean(r["judgment"]["score"] for r in rows if r["conversation_id"] == cid)
                        for cid in protocol["conversations"]}
    np.testing.assert_allclose(list(per_conversation.values()),
                               list(summary["per_conversation"].values()), rtol=0, atol=1e-15)
    rng = np.random.default_rng(20261107)
    samples = rng.choice(list(per_conversation.values()), size=(10000, len(per_conversation)),
                         replace=True).mean(axis=1)
    np.testing.assert_allclose(np.quantile(samples, [0.025, 0.975]),
                               summary["conversation_bootstrap_95ci"], rtol=0, atol=1e-15)
    print(f"{directory}: verified {len(rows)} questions, primary={summary['primary_score']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    for directory in parser.parse_args().directories:
        verify(directory)
