#!/usr/bin/env python3
"""Evaluate a JSONL ranking run against graded qrels without model-specific assumptions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evidence_retrieval.evaluate import evaluate_rankings, qrels_from_pairs
from evidence_retrieval.reproducibility import runtime_metadata, write_json


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rankings", type=Path, required=True, help="JSONL with query_id and ranked_passage_ids"
    )
    parser.add_argument(
        "--qrels", type=Path, required=True, help="JSONL with query_id, passage_id, relevance"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--latency-ms", type=float, default=None, help="Measured end-to-end mean latency per query"
    )
    args = parser.parse_args()

    start = time.perf_counter()
    rankings = {str(row["query_id"]): list(row["ranked_passage_ids"]) for row in read_jsonl(args.rankings)}
    qrels = qrels_from_pairs(
        [
            (str(row["query_id"]), str(row["passage_id"]), float(row["relevance"]))
            for row in read_jsonl(args.qrels)
        ]
    )
    metrics = evaluate_rankings(rankings, qrels)
    metrics["evaluation_wall_time_ms"] = (time.perf_counter() - start) * 1000
    if args.latency_ms is not None:
        metrics["latency_ms"] = args.latency_ms
    payload = {"metadata": runtime_metadata(args.seed, args.manifest), "metrics": metrics}
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
