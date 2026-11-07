"""Small REST client: GoodMem owns embedding, reranking and provider credentials."""

from __future__ import annotations

import json
import time

import httpx

from .config import Connection


class GoodMemError(RuntimeError):
    pass


def inference_parameters(model, max_tokens, temperature=0):
    if model.startswith(("gpt-5", "o3", "o4")):
        # GPT-5.4-mini's Chat Completions API only supports function tools
        # with reasoning_effort=none. Keep this setting for all its rounds.
        effort = "none" if model.startswith("gpt-5.4-mini") else "low"
        return {"max_completion_tokens": max_tokens, "reasoning_effort": effort}
    return {"max_tokens": max_tokens, "temperature": temperature}


class GoodMem:
    def __init__(self, conn: Connection):
        self.http = httpx.Client(
            base_url=conn.base_url.rstrip("/"), headers={"x-api-key": conn.api_key},
            verify=conn.verify, timeout=httpx.Timeout(180, connect=15),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )

    def close(self):
        self.http.close()

    def request(self, method, path, **kwargs):
        # Creates have deterministic client IDs; other POSTs are read-only inference.
        for attempt in range(7):
            try:
                r = self.http.request(method, path, **kwargs)
                if r.status_code in {429, 500, 502, 503, 504} and attempt < 6:
                    delay = r.headers.get("Retry-After", "")
                    time.sleep(min(float(delay), 60) if delay.isdigit() else min(2 ** attempt, 30))
                    continue
                if r.status_code >= 400:
                    # Do not expose provider error bodies or embedded credentials in logs.
                    raise GoodMemError(f"{method} {path} returned HTTP {r.status_code}")
                return r
            except httpx.TransportError as e:
                if attempt == 6:
                    raise GoodMemError(f"Transport failure on {method} {path}: {type(e).__name__}") from e
                time.sleep(min(2 ** attempt, 30))
        raise AssertionError("unreachable")

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs).json()

    def post(self, path, body):
        return self.request("POST", path, json=body).json()

    def inventory(self, kind):
        result, token = [], None
        while True:
            params = {"nextToken": token} if token else {}
            page = self.get(f"/v1/{kind}", params=params)
            result.extend(page.get(kind, []))
            token = page.get("nextToken")
            if not token:
                return result

    def memories(self, space):
        result, token = [], None
        while True:
            params = {"pageSize": 500}
            if token:
                params["nextToken"] = token
            page = self.get(f"/v1/spaces/{space}/memories", params=params)
            result.extend(page.get("memories", []))
            token = page.get("nextToken")
            if not token:
                return result

    def chat(self, registration, model, messages, *, max_tokens=5000, temperature=0,
             tools=None, tool_choice=None):
        body = {"model": model, "messages": messages}
        body.update(inference_parameters(model, max_tokens, temperature))
        if tools:
            body.update(tools=tools, parallel_tool_calls=False)
            if tool_choice:
                body["tool_choice"] = tool_choice
        data = self.post(f"/v1/proxy/llms/{registration}/chat/completions", body)
        choices = data.get("choices") or []
        if not choices:
            raise GoodMemError("LLM returned no choices")
        choice = choices[0]
        if choice.get("finish_reason") not in {"stop", "tool_calls"}:
            raise GoodMemError(f"Incomplete LLM output: {choice.get('finish_reason')}")
        message = choice["message"]
        if not message.get("content") and not message.get("tool_calls"):
            raise GoodMemError("LLM returned empty output")
        return {"message": message, "usage": data.get("usage", {}),
                "model": data.get("model"), "finish_reason": choice["finish_reason"]}

    def retrieve(self, query, space, reranker, *, candidates=128, top_k=12, ef_search=256):
        body = {
            "message": query, "spaceKeys": [{"spaceId": space}],
            "requestedSize": candidates, "fetchMemory": True, "fetchMemoryContent": False,
            "hnsw": {"efSearch": ef_search, "iterativeScan": "ITERATIVE_SCAN_RELAXED_ORDER",
                     "maxScanTuples": 100000},
            "postProcessor": {
                "name": "com.goodmem.retrieval.postprocess.ChatPostProcessorFactory",
                "config": {"reranker_id": reranker, "max_results": top_k},
            },
        }
        for attempt in range(5):
            response = self.request("POST", "/v1/memories:retrieve", json=body,
                                    headers={"Accept": "application/x-ndjson"})
            try:
                return parse_retrieval(response.text)
            except GoodMemError as e:
                transient = any(s in str(e).lower() for s in
                                ["rate_limit", "resource_exhausted", "rate limit", "timeout", "429", "503"])
                if not transient or attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 30))
        raise AssertionError("unreachable")


def parse_retrieval(text):
    try:
        return _parse_retrieval(text)
    except (ValueError, KeyError, TypeError) as e:
        raise GoodMemError("Malformed retrieval response") from e


def _parse_retrieval(text):
    memories, items, stages, closed, statuses = [], [], {}, set(), []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("event:", ":", "id:")):
            continue
        event = json.loads(line.removeprefix("data:").strip())
        if "memoryDefinition" in event:
            memories.append(event["memoryDefinition"])
        elif "resultSetBoundary" in event:
            b = event["resultSetBoundary"]
            if b["kind"] == "BEGIN":
                stages[b["resultSetId"]] = b
            elif b["kind"] == "END":
                closed.add(b["resultSetId"])
        elif "retrievedItem" in event:
            items.append(event["retrievedItem"]["chunk"])
        elif "status" in event:
            status = event["status"]
            statuses.append(status)
            if status.get("code") not in {"FEATURE_DISABLED", "LLM_CAPABILITY_INFERRED",
                                          "RETRIEVAL_COMPLETE", "NO_RESULTS", "RERANKING_COMPLETE",
                                          "EMBEDDING_COMPLETE", "RETRIEVAL_STARTED"}:
                raise GoodMemError(f"Retrieval status {status.get('code')}: {status.get('message', '')[:300]}")
    stage_ids = [sid for sid, b in stages.items() if b.get("stageName") == "rerank"]
    if len(stage_ids) != 1 or stage_ids[0] not in closed:
        raise GoodMemError("Rerank stream missing a complete BEGIN/END pair")
    sid = stage_ids[0]
    ranked = [x for x in items if x.get("resultSetId") == sid]
    expected = stages[sid].get("expectedItems")
    if expected and len(ranked) != expected:
        raise GoodMemError("Incomplete rerank result set")
    result = []
    for item in ranked:
        idx = int(item.get("memoryIndex", -1))
        if not 0 <= idx < len(memories):
            raise GoodMemError("Missing parent memory metadata")
        chunk, memory = item["chunk"], memories[idx]
        meta = memory.get("metadata") or {}
        result.append({"text": chunk["chunkText"], "score": item.get("relevanceScore"),
                       "block": meta.get("block"), "start": meta.get("start"),
                       "end": meta.get("end"), "chunk": chunk.get("chunkSequenceNumber"),
                       "memory_id": memory["memoryId"]})
    return result
