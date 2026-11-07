"""Answer with a different reader using exactly the original tool observations.

This diagnostic makes no new retrieval calls and provides no rubrics/categories
to the reader. It is not a new end-to-end benchmark result.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from goodmem_beam.agent import SYSTEM
from goodmem_beam.client import GoodMem
from goodmem_beam.config import Connection, load_env
from goodmem_beam.io import digest, read_json, write_json
from goodmem_beam.judge import Judge
from goodmem_beam.run import all_questions, filename

CATEGORIES = {"information_extraction", "knowledge_update", "temporal_reasoning",
              "multi_session_reasoning", "contradiction_resolution"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('runs/react-gpt54-mini'))
    parser.add_argument('--out', type=Path, default=Path('runs/diagnostic-replay-gpt52'))
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    load_env(Path('.env'))
    source = read_json(args.source / 'protocol.json')
    questions = [q for q in all_questions(Path('data/prepared'), source['conversations'])
                 if q['ability'] in CATEGORIES]
    protocol = {"method": "fixed original tool trajectory, different final reader",
                "reader": "gpt-5.2", "reasoning_effort": "low", "max_output_tokens": 10000,
                "system_prompt_sha256": digest(SYSTEM), "source_protocol_sha256": digest(source),
                "categories": sorted(CATEGORIES), "questions": len(questions),
                "final_instruction": "Using only the historical evidence already provided, answer the original question now. No further tools are available."}
    write_json(args.out / 'protocol.json', protocol)
    client = GoodMem(Connection.from_env())
    judge = Judge(client, source['model_ids'], args.out / 'judge-cache', 1200)

    def one(q):
        path = args.out / 'answers' / filename(q)
        grade_path = args.out / 'judgments' / filename(q)
        if not path.exists() or read_json(path).get('status') != 'ok':
            steps = [json.loads(line) for line in
                     (args.source / 'traces' / (filename(q) + 'l')).read_text().splitlines()]
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": q['question']}]
            for step in steps:
                message = step['assistant']
                if not message.get('tool_calls'):
                    continue  # Never show the original final answer.
                messages.append(message)
                for call, obs in zip(message['tool_calls'], step['observations'], strict=True):
                    messages.append({"role": "tool", "tool_call_id": call['id'],
                                     "content": json.dumps(obs['result'], ensure_ascii=False)})
            messages.append({"role": "user", "content": protocol['final_instruction']})
            try:
                response = client.chat(source['model_ids']['reader'], 'gpt-5.2', messages,
                                       max_tokens=10000)
                result = {"status": "ok", "id": q['id'], "answer": response['message']['content'],
                          "response": response, "input_sha256": digest(messages)}
            except Exception as error:  # noqa: BLE001 - preserve explicit per-question failures
                result = {"status": "error", "id": q['id'], "error": str(error)}
            write_json(path, result)
        result = read_json(path)
        if result['status'] != 'ok':
            return q['id'], 'answer_error'
        if grade_path.exists() and read_json(grade_path).get('status') == 'ok':
            return q['id'], 'cached'
        try:
            grade = {**judge.evaluate(q, result['answer']), "id": q['id'], "status": "ok"}
        except Exception as error:  # noqa: BLE001 - preserve explicit per-question failures
            grade = {"status": "error", "id": q['id'], "error": str(error)}
        write_json(grade_path, grade)
        return q['id'], grade['status']

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(one, q) for q in questions]):
                print(*future.result(), flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
