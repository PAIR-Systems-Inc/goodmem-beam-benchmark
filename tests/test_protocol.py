import json

import pytest

from goodmem_beam.agent import answer, trim_context
from goodmem_beam.client import GoodMemError, parse_retrieval
from goodmem_beam.dataset import ChatReader, blocks, messages, questions
from goodmem_beam.judge import event_metrics


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
