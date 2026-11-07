"""All network operations are explicit commands; downloads are pinned by revision."""

import argparse
import json
import urllib.request
from pathlib import Path

from .client import GoodMem
from .config import Connection, load_env, models, settings
from .dataset import REVISION, prepare
from .ingest import ingest, repair_failed
from .report import report
from .run import run_answers, run_judge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--index", type=Path, default=Path("runs/index-voyage4"))
    parser.add_argument("--run", type=Path, default=Path("runs/react-v1"))
    parser.add_argument("--conversations", help="Comma-separated IDs; default: every configured conversation")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["download", "prepare", "models", "ingest", "repair", "answer", "judge", "run", "report"]:
        sub.add_parser(name)
    exp = sub.add_parser("export")
    exp.add_argument("destination", type=Path)
    args = parser.parse_args()
    cfg = settings(args.config)
    ids = args.conversations.split(",") if args.conversations else cfg["dataset"]["conversations"]
    if cfg["dataset"]["revision"] != REVISION:
        raise ValueError("This adapter is pinned to the documented dataset revision")
    prepared = args.data / "prepared"
    if args.command == "download":
        args.data.mkdir(parents=True, exist_ok=True)
        for name in ["10M-00000-of-00002.parquet", "10M-00001-of-00002.parquet"]:
            path = args.data / name
            temp = path.with_suffix(".part")
            urllib.request.urlretrieve(f"https://huggingface.co/datasets/Mohammadta/BEAM-10M/resolve/{REVISION}/data/{name}", temp)
            temp.replace(path)
            print(path)
        return
    if args.command == "prepare":
        prepare(args.data, ids, cfg["ingest"]["memory_chars"])
        return
    if args.command in {"report", "export"}:
        summary = report(prepared, args.run, args.destination if args.command == "export" else None)
        print(json.dumps(summary, indent=2))
        return
    load_env(args.env)
    client = GoodMem(Connection.from_env())
    try:
        if args.command == "models":
            for kind in ["embedders", "rerankers", "llms"]:
                for m in client.inventory(kind):
                    print(json.dumps({k: v for k, v in m.items() if k in {
                        "embedderId", "rerankerId", "llmId", "modelIdentifier", "displayName", "providerType"}}))
            return
        model_ids = models()
        if args.command == "ingest":
            ingest(client, prepared, args.index, ids, model_ids["embedder"], cfg["ingest"])
        if args.command == "repair":
            repair_failed(client, prepared, args.index)
        if args.command in {"answer", "run"}:
            run_answers(client, model_ids, cfg, prepared, args.index, args.run, ids)
        if args.command in {"judge", "run"}:
            run_judge(client, model_ids, cfg, prepared, args.run, ids)
        if args.command == "run":
            print(json.dumps(report(prepared, args.run), indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
