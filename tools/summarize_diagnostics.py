"""Export the failure analysis separately from the frozen benchmark results."""

import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from replay_reader import CATEGORIES

from goodmem_beam.io import read_json, write_json


def main():
    out = Path('diagnostics/2026-11-07')
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open('results/gpt-5.4-mini/outcomes.jsonl.gz', 'rt') as handle:
        original = {r['id']: r for r in map(json.loads, handle)}
    audit_dir = Path('runs/diagnostic-evidence-gpt52')
    audit = [read_json(p) for p in sorted((audit_dir / 'judgments').glob('*.json'))]
    assert len(audit) == 180
    counts, lost_credit, per_question = defaultdict(Counter), Counter(), Counter()
    non_event_questions = 0
    missed_supported = set()
    for record in audit:
        if record['status'] != 'ok':
            continue
        q = original[record['id']]
        for item in record['items']:
            counts[q['ability']][item['label']] += 1
        if q['ability'] == 'event_ordering':
            continue
        non_event_questions += 1
        for item, nugget in zip(record['items'], q['judgment']['nuggets'], strict=True):
            lost_credit[item['label']] += (1 - nugget['score']) / len(record['items'])
            if item['label'] == 'supported' and nugget['score'] < 1:
                missed_supported.add(q['id'])
        if q['judgment']['score'] < 1:
            labels = {i['label'] for i in record['items']}
            if labels <= {'supported', 'behavioral'}:
                per_question['all_factual_targets_supported'] += 1
            elif labels & {'absent', 'partial'}:
                per_question['at_least_one_factual_target_missing_or_partial'] += 1
            else:
                per_question['uncertain'] += 1
    assert non_event_questions == 160
    coverage = {"attempted_questions": 180, "successful_audits": sum(a['status'] == 'ok' for a in audit),
                "errors": [{k: a[k] for k in ['id', 'status', 'error']} for a in audit if a['status'] != 'ok'],
                "note": "Post-hoc model judgments, not a measured causal split. A supported label requires literal verified quotes; uncertain includes citation failures.",
                "by_ability_rubric_counts": dict(counts),
                "non_event_non_abstention_questions": non_event_questions,
                "non_event_questions_losing_credit": dict(per_question),
                "non_event_questions_missing_at_least_one_supported_target": len(missed_supported),
                "non_event_lost_question_score_units_by_coverage": dict(lost_credit),
                "citation_validation_failures": sum(not i['citations_verified'] for a in audit
                                                     if a['status'] == 'ok' for i in a['items'])}
    write_json(out / 'coverage-summary.json', coverage)
    write_json(out / 'coverage-protocol.json', read_json(audit_dir / 'audit-protocol.json'))
    with gzip.open(out / 'coverage-audits.jsonl.gz', 'wt') as handle:
        for record in audit:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    replay_dir = Path('runs/diagnostic-replay-gpt52')
    replay = []
    for q in original.values():
        if q['ability'] not in CATEGORIES:
            continue
        name = q['id'].replace(':', '_') + '.json'
        answer = read_json(replay_dir / 'answers' / name)
        grade = read_json(replay_dir / 'judgments' / name)
        assert answer['status'] == grade['status'] == 'ok'
        replay.append({"id": q['id'], "conversation_id": q['conversation_id'],
                       "ability": q['ability'], "question": q['question'], "rubric": q['rubric'],
                       "original_answer": q['answer'], "original_judgment": q['judgment'],
                       "replay_answer": answer, "replay_judgment": grade})
    assert len(replay) == 100
    per_ability = {}
    for ability in sorted(CATEGORIES):
        selected = [r for r in replay if r['ability'] == ability]
        assert len(selected) == 20
        per_ability[ability] = {"original": mean(r['original_judgment']['score'] for r in selected),
                               "replay": mean(r['replay_judgment']['score'] for r in selected)}
    deltas = [mean(r['replay_judgment']['score'] - r['original_judgment']['score']
                   for r in replay if r['conversation_id'] == str(cid)) for cid in range(1, 11)]
    samples = np.random.default_rng(20261107).choice(deltas, size=(10000, 10), replace=True).mean(axis=1)
    replay_summary = {"questions": 100, "new_retrievals": 0, "reference_information_given_to_reader": False,
                      "original_score": mean(r['original_judgment']['score'] for r in replay),
                      "replay_score": mean(r['replay_judgment']['score'] for r in replay),
                      "per_ability": per_ability,
                      "improved_questions": sum(r['replay_judgment']['score'] > r['original_judgment']['score'] for r in replay),
                      "worsened_questions": sum(r['replay_judgment']['score'] < r['original_judgment']['score'] for r in replay),
                      "paired_difference_95ci": [float(x) for x in np.quantile(samples, [.025, .975])],
                      "bootstrap_samples": 10000, "bootstrap_seed": 20261107,
                      "actual_reader_models": dict(Counter(r['replay_answer']['response']['model'] for r in replay)),
                      "limitations": "Final-reader/model/reasoning/instruction change with fixed earlier tool trajectory, not a full end-to-end run. Judge sampling variability is not estimated."}
    write_json(out / 'replay-summary.json', replay_summary)
    write_json(out / 'replay-protocol.json', read_json(replay_dir / 'protocol.json'))
    with gzip.open(out / 'replay-outcomes.jsonl.gz', 'wt') as handle:
        for record in replay:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    observations = []
    for path in Path('runs/react-gpt54-mini/traces').glob('*.jsonl'):
        if '.attempt-' in path.name:
            continue
        for step in map(json.loads, path.read_text().splitlines()):
            observations.extend(o for o in step['observations'] if 'hits' in o['result'])
    trace_stats = {"searches": len(observations),
                   "chunks_returned": sum(len(o['result']['hits']) for o in observations),
                   "chunks_without_an_inline_message_header": sum(not re.search(r'\[message \d+ \|', h['text'])
                       for o in observations for h in o['result']['hits']),
                   "mean_distinct_memory_blocks_per_search": mean(len({h['block'] for h in o['result']['hits']}) for o in observations),
                   "by_ability": {}}
    for ability in sorted({q['ability'] for q in original.values()}):
        qs = [q for q in original.values() if q['ability'] == ability]
        failed = [q for q in qs if q['judgment']['score'] < 1]
        trace_stats['by_ability'][ability] = {"questions": len(qs), "not_full_credit": len(failed),
            "not_full_credit_and_one_search": sum(q['answer']['searches'] == 1 for q in failed)}
    write_json(out / 'trace-statistics.json', trace_stats)
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'artifacts.sha256')
    (out / 'artifacts.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p}\n' for p in files))
    print(json.dumps({"coverage": coverage, "replay": replay_summary}, indent=2))


if __name__ == '__main__':
    main()
