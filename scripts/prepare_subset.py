#!/usr/bin/env python3
"""Prepare a deterministic local T2Ranking subset without downloading or uploading data."""

from __future__ import annotations

import argparse
from pathlib import Path

from evidence_retrieval.data import build_subset_manifest

PRESETS = {
    "smoke": {"train_query_limit": 64, "dev_query_limit": 32},
    "small": {"train_query_limit": 2_000, "dev_query_limit": 500},
    "medium": {"train_query_limit": 20_000, "dev_query_limit": 2_000},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, required=True, help="Directory containing official T2Ranking data files"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Local directory for processed JSONL files"
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hard-negative-file", default="train.mined.tsv")
    args = parser.parse_args()
    settings = PRESETS[args.preset]
    manifest_path = build_subset_manifest(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_query_limit=settings["train_query_limit"],
        dev_query_limit=settings["dev_query_limit"],
        seed=args.seed,
        hard_negative_file=args.hard_negative_file,
    )
    print(f"Prepared {args.preset} subset. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
