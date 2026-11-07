import copy
import json
import sqlite3

import pytest

from goodmem_beam.agent import answer
from goodmem_beam.agent_v2 import trim_context
from goodmem_beam.client import GoodMemError
from goodmem_beam.dataset import ChatReader


class Client:
    def __init__(self, calls):
        self.pending = iter(calls)
        self.inputs = []

    def responses(self, registration, model, items, **kwargs):
        self.inputs.append(copy.deepcopy(items))
        name, args = next(self.pending)
        output = [{"type": "reasoning", "id": f"r{len(self.inputs)}", "summary": [],
                   "encrypted_content": "opaque-private-state"}]
        if name is None:
            output.append({"type": "message", "role": "assistant", "phase": "final_answer",
                           "content": [{"type": "output_text", "text": args}]})
        else:
            output.append({"type": "function_call", "name": name, "call_id": f"c{len(self.inputs)}",
                           "arguments": json.dumps(args)})
        return {"output": output, "model": model, "reasoning": {"effort": "medium"},
                "usage": {"input_tokens": 10, "output_tokens": 4,
                          "output_tokens_details": {"reasoning_tokens": 2}}}

    def retrieve(self, *args, **kwargs):
        return [{"memory_id": "m", "chunk": 0, "start": 0, "end": 0,
                 "text": "The source fact"}]


class Transcript:
    def locate_chunk(self, *args):
        return {"exact_match": True, "messages": [{"message_id": 0, "header": "user"}]}

    def read(self, *args):
        return {"messages": [{"message_id": 0, "text": "A fresh source check"}]}


def run(client, max_steps=15):
    return answer(client, "Question only", "space", Transcript(),
                  {"reader": "reg", "reader_model": "gpt-6.1-sol", "reranker": "rr"},
                  {"api": "responses", "max_steps": max_steps, "min_distinct_searches": 3,
                   "max_context_chars": 100000, "max_output_tokens": 1000,
                   "reasoning_effort": "medium"}, {})


def submission(evidence="S1.1"):
    return {"answer": "The supported answer.", "coverage": [
        {"requirement": "Requested fact", "finding": "Source fact", "status": "supported",
         "evidence_ids": [evidence]}], "remaining_uncertainties": []}


def test_stop_gate_requires_distinct_queries_and_postdraft_evidence():
    client = Client([
        ("search_memory", {"query": "one"}),
        (None, "An early answer"),
        ("search_memory", {"query": "  ONE "}),
        ("submit_answer", submission()),
        ("search_memory", {"query": "two"}),
        ("search_memory", {"query": "three"}),
        ("submit_answer", submission()),
        ("submit_answer", submission()),
        ("read_messages", {"start": 0, "end": 0}),
        ("submit_answer", submission("M0")),
    ])
    result = run(client)
    assert result["stop_reason"] == "verified_submission"
    assert (result["searches"], result["distinct_searches"], result["reads"]) == (4, 3, 1)
    assert result["rejected_submissions"] == 4
    assert result["usage"]["reasoning_tokens"] == 20
    assert "opaque-private-state" not in json.dumps(result)
    assert any(i.get("encrypted_content") == "opaque-private-state" for i in client.inputs[1])
    assert any(i.get("phase") == "final_answer" for i in client.inputs[2])
    assert any(i.get("type") == "function_call_output" for i in client.inputs[1])


def test_citations_must_refer_to_returned_evidence_and_budget_remains_bounded():
    result = run(Client([("search_memory", {"query": "one"}),
                         ("submit_answer", submission("S99.1")), (None, "Uncertain answer")]),
                 max_steps=2)
    assert "never returned" in result["trace"][1]["observations"][0]["result"]["tool_input_error"]
    assert result["stop_reason"] == "budget_exhausted"


def test_responses_preserves_service_failures():
    class Broken(Client):
        def retrieve(self, *args, **kwargs):
            raise GoodMemError("Provider unavailable")
    with pytest.raises(GoodMemError, match="unavailable"):
        run(Broken([("search_memory", {"query": "q"})]))


def test_responses_compaction_preserves_opaque_items_and_call_ids():
    items = [{"type": "reasoning", "encrypted_content": "opaque"},
             {"type": "function_call", "call_id": "c", "name": "search_memory"},
             {"type": "function_call_output", "call_id": "c", "output": "x" * 5000},
             {"role": "user", "content": "continue"}]
    assert trim_context(items, 1000) == 1
    assert items[0]["encrypted_content"] == "opaque"
    assert items[2]["call_id"] == "c"


def test_chunk_metadata_restoration_never_guesses_source(tmp_path):
    path = tmp_path / "messages.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, text TEXT)")
        db.executemany("INSERT INTO messages VALUES (?,?)", [
            (0, "[message 0 | 2024-01-01 | user]\nI started in January."),
            (1, "[message 1 | 2024-02-01 | assistant]\nAn example in February.")])
    reader = ChatReader(path)
    result = reader.locate_chunk(0, 1, "started in January.")
    assert result == {"exact_match": True, "messages": [
        {"message_id": 0, "header": "[message 0 | 2024-01-01 | user]"}]}
    assert not reader.locate_chunk(0, 1, "not in the source")["exact_match"]
    assert not reader.locate_chunk(1, 1, "started in January.")["exact_match"]
