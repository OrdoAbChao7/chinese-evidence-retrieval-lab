#!/usr/bin/env python3
"""Aggregate versioned per-seed metric JSON files into a Markdown summary."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", default="artifacts/runs/**/metrics.json")
    parser.add_argument("--output", type=Path, default=Path("artifacts/metrics/results_summary.md"))
    args = parser.parse_args()

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    files = sorted(Path(".").glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No metric files found for glob: {args.input_glob}")
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        experiment = str(payload.get("experiment") or path.parents[1].name)
        for metric, value in payload.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                grouped[experiment][metric].append(float(value))

    lines = ["# Experiment Results", "", "Results are aggregated across completed, versioned seed runs.", ""]
    for experiment, metrics in sorted(grouped.items()):
        lines.extend([f"## {experiment}", "", "| Metric | Runs | Mean | Std. dev. |", "|---|---:|---:|---:|"])
        for metric, values in sorted(metrics.items()):
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            lines.append(f"| {metric} | {len(values)} | {mean:.6f} | {std:.6f} |")
        lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
