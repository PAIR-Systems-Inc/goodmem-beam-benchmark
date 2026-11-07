"""Macro scores, conversation-cluster intervals, and a public evidence export."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

from .dataset import ABILITIES
from .io import read_json, write_json
from .run import all_questions, filename


def report(prepared: Path, run_dir: Path, export_dir: Path | None = None):
    protocol = read_json(run_dir / "protocol.json")
    ids = protocol["conversations"]
    qs = all_questions(prepared, ids)
    outcomes, errors = [], []
    for q in qs:
        a_path, j_path = run_dir / "answers" / filename(q), run_dir / "judgments" / filename(q)
        a = read_json(a_path) if a_path.exists() else {"status": "missing"}
        j = read_json(j_path) if j_path.exists() else {"status": "missing"}
        if a["status"] != "ok" or j["status"] != "ok":
            errors.append({"id": q["id"], "answer": a["status"], "judgment": j["status"]})
            continue
        outcomes.append({**q, "answer": a, "judgment": j})
    complete = len(outcomes) == len(qs)
    primary, adjusted, per_conversation = {}, {}, {}
    for cid in ids:
        by_ability = {}
        for ability in ABILITIES:
            values = [o["judgment"]["score"] for o in outcomes
                      if o["conversation_id"] == cid and o["ability"] == ability]
            if len(values) == 2:
                by_ability[ability] = mean(values)
        if len(by_ability) == len(ABILITIES):
            per_conversation[cid] = mean(by_ability.values())
    for ability in ABILITIES:
        selected = [o for o in outcomes if o["ability"] == ability]
        if selected:
            primary[ability] = mean(o["judgment"]["score"] for o in selected)
            adjusted[ability] = mean(o["judgment"]["recall_adjusted_score"] for o in selected)
    interval = None
    if complete and len(per_conversation) > 1:
        rng = np.random.default_rng(20261107)
        values = np.array(list(per_conversation.values()))
        sample = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
        interval = [float(x) for x in np.quantile(sample, [0.025, 0.975])]
    summary = {
        "complete": complete, "expected_questions": len(qs), "scored_questions": len(outcomes),
        "errors_or_missing": errors,
        "primary_score": mean(primary.values()) if complete else None,
        "recall_adjusted_score": mean(adjusted.values()) if complete else None,
        "conversation_bootstrap_95ci": interval, "per_ability": primary,
        "per_ability_recall_adjusted": adjusted, "per_conversation": per_conversation,
        "metric": "Mean of conversation/ability means; event ordering uses upstream tau_norm",
        "searches_mean": mean(o["answer"]["searches"] for o in outcomes) if outcomes else None,
        "searches_max": max((o["answer"]["searches"] for o in outcomes), default=0),
        "reads_mean": mean(o["answer"]["reads"] for o in outcomes) if outcomes else None,
        "answer_seconds_mean": mean(o["answer"]["seconds"] for o in outcomes) if outcomes else None,
        "stop_reasons": dict(Counter(o["answer"]["stop_reason"] for o in outcomes)),
        "reader_usage": {k: sum(o["answer"]["usage"][k] for o in outcomes)
                         for k in ["prompt_tokens", "completion_tokens"]},
    }
    write_json(run_dir / "summary.json", summary)
    if export_dir:
        export_dir.mkdir(parents=True, exist_ok=True)
        write_json(export_dir / "summary.json", summary)
        public_protocol = {k: v for k, v in protocol.items() if k != "model_ids"}
        write_json(export_dir / "protocol.json", public_protocol)
        write_json(export_dir / "dataset-manifest.json", read_json(prepared / "manifest.json"))
        with gzip.open(export_dir / "outcomes.jsonl.gz", "wt", encoding="utf-8") as f:
            for outcome in outcomes:
                f.write(json.dumps(outcome, ensure_ascii=False) + "\n")
        with gzip.open(export_dir / "retrieval-traces.jsonl.gz", "wt", encoding="utf-8") as f:
            for q in qs:
                p = run_dir / "traces" / (filename(q) + "l")
                if not p.exists():
                    continue
                for line in p.read_text().splitlines():
                    step = json.loads(line)
                    # Publish tool decisions and evidence identifiers/digests,
                    # not hundreds of pages of source conversation excerpts.
                    for obs in step["observations"]:
                        result = obs["result"]
                        if "hits" in result:
                            for hit in result["hits"]:
                                hit.pop("text", None)
                                hit.pop("memory_id", None)
                        if "messages" in result:
                            for msg in result["messages"]:
                                msg.pop("text", None)
                    f.write(json.dumps({"id": q["id"], **step}, ensure_ascii=False) + "\n")
    return summary
