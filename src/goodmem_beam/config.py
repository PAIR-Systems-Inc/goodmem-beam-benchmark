"""Public experiment settings and environment-only credentials."""

from __future__ import annotations

import os
import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        parts = shlex.split(value, comments=True)
        os.environ.setdefault(key.strip(), parts[0] if parts else "")


@dataclass(frozen=True)
class Connection:
    base_url: str
    api_key: str = field(repr=False)
    verify: bool = True

    @classmethod
    def from_env(cls):
        return cls(
            os.environ.get("GOODMEM_BASE_URL", "https://localhost:8080"),
            os.environ["GOODMEM_API_KEY"],
            os.environ.get("GOODMEM_TLS_VERIFY", "true").lower() not in {"0", "false"},
        )


def settings(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def models() -> dict:
    return {
        "embedder": os.environ["BEAM_EMBEDDER_ID"],
        "reranker": os.environ["BEAM_RERANKER_ID"],
        "reader": os.environ["BEAM_READER_ID"],
        "reader_model": os.environ.get("BEAM_READER_MODEL", "gpt-4.1-mini"),
        "judge": os.environ["BEAM_JUDGE_ID"],
        "judge_model": os.environ.get("BEAM_JUDGE_MODEL", "gpt-4.1-mini"),
    }
