"""A standalone batch must not depend on another experiment's results."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from goodmem_beam.io import read_json, write_json


@pytest.mark.parametrize("with_baseline", [False, True])
def test_batch_runs_without_requiring_a_baseline(tmp_path, monkeypatch, with_baseline):
    script = Path(__file__).resolve().parents[1] / "tools" / "run_batch.py"
    spec = importlib.util.spec_from_file_location("run_batch", script)
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    run = tmp_path / "sol-run"
    question = {"id": "1:q", "conversation_id": "1", "ability": "abstention"}
    write_json(run / "judgments" / "1_q.json", {"status": "ok", "score": 0.75})

    monkeypatch.setattr(batch, "settings", lambda _: {"dataset": {"conversations": ["1"]}})
    monkeypatch.setattr(batch, "all_questions", lambda *_: [question])
    monkeypatch.setattr(batch, "load_env", lambda _: None)
    monkeypatch.setattr(batch, "Connection", SimpleNamespace(from_env=lambda: None))
    monkeypatch.setattr(batch, "GoodMem", lambda _: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(batch, "models", dict)
    monkeypatch.setattr(batch, "run_answers", lambda *_, **__: None)
    monkeypatch.setattr(batch, "run_judge", lambda *_, **__: None)
    monkeypatch.setattr(batch, "report", lambda *_: None)
    argv = [str(script), "--run", str(run), "--config", "unused.toml", "--batch", "0"]
    if with_baseline:
        baseline = tmp_path / "comparison-run"
        write_json(baseline / "judgments" / "1_q.json", {"status": "ok", "score": 0.25})
        argv += ["--baseline", str(baseline)]
    monkeypatch.setattr(sys, "argv", argv)

    batch.main()

    result = read_json(run / "batches" / "batch-00.json")
    assert result["complete"] and result["score"] == 0.75
    if with_baseline:
        assert result["baseline_score"] == 0.25
        assert result["paired_delta"] == 0.5
    else:
        assert "baseline_score" not in result and "paired_delta" not in result
        assert "baseline" not in result["per_ability"]["abstention"]
        assert "baseline_score" not in result["rows"][0]
