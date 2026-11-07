import json

import pytest

from goodmem_beam.agent import answer, trim_context
from goodmem_beam.client import GoodMemError, parse_retrieval
from goodmem_beam.dataset import ChatReader, blocks, messages, questions
from goodmem_beam.judge import event_metrics, parse_json


def test_judge_normalizes_only_literal_invalid_escapes():
    raw = r'{"score": 0.5, "reason": "Uses \(x\), \\alpha and \"quoted\" text.\nDone."}'
    verdict, normalized = parse_json(raw)
    assert normalized
    assert verdict == {"score": 0.5,
                       "reason": 'Uses \\(x\\), \\alpha and "quoted" text.\nDone.'}
    valid = json.dumps(verdict)
    assert parse_json(valid) == (verdict, False)
    assert parse_json('```json\n' + valid + '\n```') == (verdict, False)


@pytest.mark.parametrize("raw", [
    r'{"score": 1, "reason": "truncated \(x\)"',
    r'{"score": 1, "reason": "invalid unicode \uZZZZ"}',
    r'{"score": 1 "reason": "missing comma"}',
])
def test_judge_refuses_structural_json_repair(raw):
    with pytest.raises(json.JSONDecodeError):
        parse_json(raw)


def test_canonical_chat_no_duplicate_plans_or_metadata():
    chat = [{"plan-1": [{"time_anchor": None, "turns": [[
        {"id": 0, "role": "user", "content": "First fact", "time_anchor": "2024-01-01"},
        {"id": 1, "role": "assistant", "content": "Response", "time_anchor": None},
    ]]}], "plan-10": None}, {"plan-1": None, "plan-10": [
        {"time_anchor": None, "turns": [[
            {"id": 2, "role": "user", "content": "Later fact", "time_anchor": "2024-09-01"}
        ]]}
    ]}]
    flattened = list(messages(chat))
    assert [m["id"] for m in flattened] == [0, 1, 2]
    assert flattened[1]["time"] == "2024-01-01"
    assert flattened[2]["plan"] == "plan-10"
    packed = list(blocks(flattened, 60))
    assert packed[0]["start"] == 0
    assert packed[-1]["end"] == 2
    assert all("rubric" not in block["text"] for block in packed)
    chat[1]["plan-10"][0]["turns"][0][0]["id"] = 0
    with pytest.raises(ValueError, match="Noncontiguous"):
        list(messages(chat))


def test_question_loader_uses_safe_literal_parser():
    with pytest.raises((ValueError, SyntaxError)):
        questions("__import__('os').system('false')", "1")


def test_event_ranking_matches_released_semantics():
    assert event_metrics(["a", "b", "c"], ["a", "b", "c"])["tau_norm"] == 1
    assert event_metrics(["a", "b", "c"], ["c", "b", "a"])["tau_norm"] == 0
    missing = event_metrics(["a", "b", "c"], ["a", "b"])
    assert missing["tau_norm"] == 1
    assert missing["final_score"] == pytest.approx(0.8)
    assert event_metrics(["a", "b"], ["I don't know"])["final_score"] == 0
    degenerate = event_metrics(["a", "b"], [])
    assert degenerate["undefined_tau"]
    assert degenerate["tau_norm"] == 0


class FakeClient:
    def __init__(self, rounds=3):
        self.calls, self.queries, self.rounds = [], [], rounds

    def chat(self, registration, model, messages, **kwargs):
        self.calls.append(json.dumps(messages))
        step = len(self.calls)
        if step <= self.rounds and kwargs.get("tools"):
            msg = {"role": "assistant", "tool_calls": [{"id": f"t{step}", "type": "function",
                "function": {"name": "search_memory", "arguments": json.dumps({"query": f"query {step}"})}}]}
        else:
            msg = {"role": "assistant", "content": "The evidence supports the answer."}
        return {"message": msg, "usage": {"prompt_tokens": 10, "completion_tokens": 2}, "model": model}

    def retrieve(self, query, space, reranker, **kwargs):
        self.queries.append(query)
        return [{"memory_id": "m", "chunk": len(self.queries), "text": f"evidence {query}"}]


def test_agent_keeps_looping_and_observes_prior_results():
    client = FakeClient()
    config = {"max_steps": 8, "max_context_chars": 100000, "max_output_tokens": 500,
              "temperature": 0}
    result = answer(client, "A question", "isolated-space", None,
                    {"reader": "r", "reader_model": "model", "reranker": "rr"}, config, {})
    assert result["searches"] == 3
    assert result["stop_reason"] == "answered"
    assert result["usage"]["prompt_tokens"] == 40
    assert "evidence query 1" in client.calls[1]
    assert "evidence query 2" in client.calls[2]
    assert len(result["trace"]) == 4


def test_agent_budget_is_explicit_and_still_generates_answer():
    client = FakeClient(rounds=50)
    config = {"max_steps": 2, "max_context_chars": 100000, "max_output_tokens": 500,
              "temperature": 0}
    result = answer(client, "A question", "s", None,
                    {"reader": "r", "reader_model": "model", "reranker": "rr"}, config, {})
    assert result["stop_reason"] == "budget_exhausted"
    assert result["searches"] == 2
    assert result["answer"]


def test_agent_does_not_treat_service_failure_as_no_evidence():
    class Broken(FakeClient):
        def retrieve(self, *args, **kwargs):
            raise GoodMemError("provider failed")
    config = {"max_steps": 8, "max_context_chars": 100000, "max_output_tokens": 500,
              "temperature": 0}
    with pytest.raises(GoodMemError, match="provider failed"):
        answer(Broken(), "q", "s", None,
               {"reader": "r", "reader_model": "model", "reranker": "rr"}, config, {})


def test_stream_validation_rejects_incomplete_or_degraded_retrieval():
    begin = {"resultSetBoundary": {"kind": "BEGIN", "stageName": "rerank", "resultSetId": "s"}}
    end = {"resultSetBoundary": {"kind": "END", "resultSetId": "s"}}
    assert parse_retrieval("\n".join(map(json.dumps, [begin, end]))) == []
    with pytest.raises(GoodMemError, match="complete"):
        parse_retrieval(json.dumps(begin))
    with pytest.raises(GoodMemError, match="RERANKING_FAILED"):
        parse_retrieval(json.dumps({"status": {"code": "RERANKING_FAILED"}}))
    with pytest.raises(GoodMemError, match="Malformed"):
        parse_retrieval('{"truncated":')


def test_read_tool_is_readonly_bounded_and_chronological(tmp_path):
    import sqlite3
    path = tmp_path / "chat.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, text TEXT)")
        db.executemany("INSERT INTO messages VALUES (?,?)", [(i, "fact" * 50) for i in range(4)])
    reader = ChatReader(path)
    result = reader.read(0, 3, char_budget=250)
    assert [m["message_id"] for m in result["messages"]] == [0, 1]
    assert result["messages"][-1]["truncated"]
    with pytest.raises(ValueError):
        reader.read(0, 1000)


def test_compaction_preserves_tool_protocol():
    history = [{"role": "user", "content": "question"},
               {"role": "assistant", "tool_calls": [{"id": "a"}]},
               {"role": "tool", "tool_call_id": "a", "content": "x" * 1000},
               {"role": "assistant", "content": "continue"}]
    assert trim_context(history, 500) == 1
    assert history[0]["content"] == "question"
    assert history[2]["tool_call_id"] == "a"


def test_report_never_reports_a_complete_score_for_missing_questions(tmp_path):
    from goodmem_beam.dataset import ABILITIES
    from goodmem_beam.io import digest, write_json
    from goodmem_beam.report import report
    from goodmem_beam.run import filename

    prepared, run = tmp_path / "prepared", tmp_path / "run"
    qs = [{"id": f"1:{ability}:{i}", "conversation_id": "1", "ability": ability,
           "question": "q", "rubric": ["r"]} for ability in ABILITIES for i in range(2)]
    write_json(prepared / "1.questions.json", qs)
    write_json(prepared / "manifest.json", {"conversations": [{"id": "1", "questions_sha256": digest(qs)}]})
    write_json(run / "protocol.json", {"conversations": ["1"]})
    for q in qs:
        write_json(run / "answers" / filename(q), {"status": "ok", "searches": 3, "reads": 0,
                   "seconds": 1, "stop_reason": "answered", "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        write_json(run / "judgments" / filename(q), {"status": "ok", "score": 1, "recall_adjusted_score": 1})
    assert report(prepared, run)["primary_score"] == 1
    (run / "judgments" / filename(qs[0])).unlink()
    partial = report(prepared, run)
    assert partial["primary_score"] is None
    assert partial["expected_questions"] == 20
    assert partial["scored_questions"] == 19


def test_repair_refuses_to_delete_a_memory_outside_its_deterministic_ids(tmp_path, monkeypatch):
    from goodmem_beam import ingest as module
    from goodmem_beam.io import digest, write_json

    prepared, run = tmp_path / "prepared", tmp_path / "run"
    block = {"start": 0, "end": 1, "text": "synthetic conversation"}
    write_json(prepared / "1.blocks.json", [block])
    write_json(run / "index.json", {"identity": {"conversations": [{"id": "1", "memory_blocks": 1}]},
                                    "spaces": {"1": "00000000-0000-4000-8000-000000000000"}})

    class FakePacer:
        def wait(self, batch):
            pass

    class Client:
        def memories(self, space):
            return [{"memoryId": "unrelated", "processingStatus": "FAILED",
                     "metadata": {"block": 0, "sha256": digest(block)}}]

        def request(self, *args, **kwargs):
            pytest.fail("An unrelated memory must not be deleted")

    monkeypatch.setattr(module, "SubmissionPacer", FakePacer)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    with pytest.raises(GoodMemError, match="ownership/content"):
        module.repair_failed(Client(), prepared, run)
