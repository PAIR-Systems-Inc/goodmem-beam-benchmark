"""Compare retained versus truncated evidence using a fixed tool trajectory.

This is a post-hoc diagnostic, not an end-to-end BEAM score. Neither condition
gets the original answer, coverage drafts, rubric, ability label, or new retrieval.
Both reconstruct the same public tool history without private reasoning state.
"""

import argparse
import copy
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from goodmem_beam.agent_v2 import SYSTEM, trim_context
from goodmem_beam.client import GoodMem
from goodmem_beam.config import Connection, load_env
from goodmem_beam.io import digest, read_json, write_json
from goodmem_beam.judge import Judge
from goodmem_beam.run import all_questions, filename


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--questions", nargs="+", required=True)
    args = p.parse_args()
    source = read_json(args.source / "protocol.json")
    questions = [q for q in all_questions(Path("data/prepared"), source["conversations"])
                 if q["id"] in args.questions]
    if len(questions) != len(set(args.questions)):
        raise ValueError("Unknown selected question")
    conditions = {"trimmed": source["config"]["agent"]["max_context_chars"], "retained": 2800000}
    protocol = {"source_protocol_sha256": digest(source), "conditions": conditions,
                "questions": args.questions, "reader": source["requested_models"]["reader"],
                "reasoning_effort": "medium", "helper_sha256": digest(Path(__file__).read_text()),
                "method": "Same fixed evidence and reader; vary only active context character limit; original drafts/answers and opaque reasoning omitted in both conditions"}
    write_json(args.out / "protocol.json", protocol)
    load_env(Path(".env"))
    client = GoodMem(Connection.from_env())
    judge = Judge(client, source["model_ids"], args.out / "judge-cache", 1200)

    def one(q, condition, budget):
        directory = args.out / condition
        path = directory / "answers" / filename(q)
        items = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["question"]}]
        for line in (args.source / "traces" / (filename(q) + "l")).read_text().splitlines():
            step = json.loads(line)
            calls = step["assistant"].get("tool_calls", [])
            for call, observation in zip(calls, step["observations"], strict=True):
                if call["function"]["name"] == "submit_answer":
                    continue
                items.append({"type": "function_call", "call_id": call["id"],
                              "name": call["function"]["name"],
                              "arguments": call["function"]["arguments"]})
                items.append({"type": "function_call_output", "call_id": call["id"],
                              "output": json.dumps(observation["result"], ensure_ascii=False)})
        items.append({"role": "user", "content": "Using only the historical evidence already provided, answer the original question now. No further tools are available."})
        input_before = digest(items)
        dropped = trim_context(items, budget)
        if not path.exists():
            response = client.responses(source["model_ids"]["reader"], protocol["reader"], items,
                                        max_tokens=16000, reasoning_effort="medium")
            text = "\n".join(part["text"] for item in response["output"] if item["type"] == "message"
                             for part in item.get("content", []) if part["type"] == "output_text")
            if not text.strip():
                raise ValueError("Empty diagnostic answer")
            write_json(path, {"id": q["id"], "answer": text, "status": "ok", "model": response["model"],
                              "usage": response["usage"], "reasoning": response["reasoning"],
                              "context_compactions": dropped, "input_sha256": digest(items),
                              "original_input_sha256": input_before})
        answer = read_json(path)
        grade_path = directory / "judgments" / filename(q)
        if not grade_path.exists():
            write_json(grade_path, {**judge.evaluate(q, answer["answer"]), "id": q["id"], "status": "ok"})
        return {"id": q["id"], "condition": condition, "score": read_json(grade_path)["score"],
                "original_score": read_json(args.source / "judgments" / filename(q))["score"],
                "compactions": dropped}

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            tasks = [pool.submit(one, copy.deepcopy(q), name, budget)
                     for q in questions for name, budget in conditions.items()]
            results = []
            for future in as_completed(tasks):
                result = future.result()
                results.append(result)
                print(json.dumps(result), flush=True)
        write_json(args.out / "summary.json", results)
    finally:
        client.close()


if __name__ == "__main__":
    main()
