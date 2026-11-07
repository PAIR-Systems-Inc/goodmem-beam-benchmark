"""BEAM's released nugget judge and event alignment/ranking protocol."""

from __future__ import annotations

import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scipy.stats import kendalltau

from .io import digest, read_json, write_json

NUGGET_PROMPT = (Path(__file__).parent / "prompts/nugget_judge.txt").read_text()
EQUIVALENCE_SYSTEM = """
            You are a binary classifier.
            If the TWO snippets describe the SAME event/fact, reply **YES**
            Otherwise reply **NO**. No extra words.
            DO NOT provide any exaplanation.
        """


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


def event_metrics(reference, aligned):
    """Exact released ranking formula, with undefined tau explicitly scored zero."""
    tp = len(set(reference) & set(aligned))
    fp = sum(x not in reference for x in aligned)
    fn = sum(x not in aligned for x in reference)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = list(dict.fromkeys(reference + aligned))
    tie_rank = len(union) + 1

    def ranks(seq):
        rank = {item: i + 1 for i, item in enumerate(seq)}
        return [rank.get(item, tie_rank) for item in union]

    tau = float(kendalltau(ranks(reference), ranks(aligned), variant="b", method="auto").statistic)
    undefined = not math.isfinite(tau)
    normalized = 0.0 if undefined else (tau + 1) / 2
    return {"precision": precision, "recall": recall, "f1": f1,
            "tau_b": None if undefined else tau, "tau_norm": normalized,
            "undefined_tau": undefined, "final_score": normalized * f1}


class Judge:
    def __init__(self, client, model_ids, cache_dir, max_tokens=1200):
        self.client, self.models, self.cache = client, model_ids, cache_dir
        self.max_tokens = max_tokens
        self.lock = threading.Lock()

    def completion(self, messages, max_tokens=None):
        key = digest({"messages": messages, "model": self.models["judge_model"],
                      "registration": self.models["judge"], "max_tokens": max_tokens or self.max_tokens})
        path = self.cache / f"{key}.json"
        if path.exists():
            return read_json(path)
        response = self.client.chat(self.models["judge"], self.models["judge_model"],
                                    messages, max_tokens=max_tokens or self.max_tokens)
        # Each prompt has a deterministic content key. Lock only atomic writes.
        with self.lock:
            write_json(path, response)
        return response

    def equivalent(self, reference, candidate):
        messages = [
            {"role": "system", "content": EQUIVALENCE_SYSTEM},
            {"role": "user", "content": f"First snippet: {reference} \n\n                       Second snippet: {candidate}\n                    "},
        ]
        response = self.completion(messages, max_tokens=12)
        raw = response["message"]["content"].strip()
        # Upstream uses substring membership; preserve it and retain the raw output.
        return {"match": "yes" in raw.lower(), "response": raw, "usage": response["usage"]}

    def evaluate(self, q, answer):
        rubric = q["rubric"]
        if q["ability"] == "event_ordering":
            # Preserve upstream newline splitting, including blank lines.
            candidates = answer.split("\n")
            used, aligned, comparisons = set(), [], []
            with ThreadPoolExecutor(max_workers=4) as pool:
                for candidate in candidates:
                    remaining = [i for i in range(len(rubric)) if i not in used]
                    match = None
                    # Parallelize independent comparisons in small batches; select
                    # the first positive in reference order, exactly as upstream.
                    for begin in range(0, len(remaining), 4):
                        indices = remaining[begin:begin+4]
                        verdicts = list(pool.map(lambda i, c=candidate: self.equivalent(rubric[i], c), indices))
                        for i, verdict in zip(indices, verdicts, strict=True):
                            comparisons.append({"reference_index": i, "candidate": candidate, **verdict})
                            if verdict["match"] and match is None:
                                match = i
                        if match is not None:
                            break
                    if match is None:
                        aligned.append(candidate)
                    else:
                        aligned.append(rubric[match])
                        used.add(match)
            metrics = event_metrics(rubric, aligned)
            return {"score": metrics["tau_norm"], "recall_adjusted_score": metrics["final_score"],
                    "event_metrics": metrics, "aligned": aligned, "comparisons": comparisons}
        verdicts = []
        for item in rubric:
            # Exact released prompt: it does not interpolate the question.
            prompt = NUGGET_PROMPT.replace("<rubric_item>", item).replace("<llm_response>", answer)
            result = self.completion([{"role": "user", "content": prompt}])
            raw = result["message"]["content"]
            verdict = parse_json(raw)
            if verdict.get("score") not in {0, 0.5, 1} or not isinstance(verdict.get("reason"), str):
                raise ValueError("Invalid nugget judgment; refusing to invent a grade")
            verdicts.append({"rubric": item, **verdict, "raw": raw, "usage": result["usage"]})
        score = sum(v["score"] for v in verdicts) / len(verdicts)
        return {"score": score, "recall_adjusted_score": score, "nuggets": verdicts}
