"""T2Ranking parsing, deterministic smoke subsets, and split-integrity checks."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .reproducibility import sha256_file


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


@dataclass(frozen=True)
class Passage:
    passage_id: str
    text: str


def read_tsv_mapping(path: str | Path) -> dict[str, str]:
    """Read a two-column UTF-8 TSV file keyed by its first column."""
    rows: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t", maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Expected two TSV columns at {path}:{line_number}")
            key, text = parts
            if key in rows:
                raise ValueError(f"Duplicate identifier {key!r} in {path}")
            rows[key] = text
    return rows


def read_retrieval_qrels(path: str | Path) -> dict[str, set[str]]:
    """Read T2Ranking retrieval qrels (qid<TAB>pid), tolerating extra columns."""
    qrels: dict[str, set[str]] = defaultdict(set)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                raise ValueError(f"Expected qid and pid at {path}:{line_number}")
            qrels[parts[0]].add(parts[1])
    return dict(qrels)


def read_hard_negatives(path: str | Path) -> dict[str, list[str]]:
    """Read qid/pid negatives from either BM25 or mined-negative files."""
    negatives: dict[str, list[str]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                raise ValueError(f"Expected qid and pid at {path}:{line_number}")
            negatives[parts[0]].append(parts[1])
    return dict(negatives)


def assert_disjoint_query_sets(train_ids: Iterable[str], dev_ids: Iterable[str]) -> None:
    overlap = set(train_ids).intersection(dev_ids)
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"Train/dev query leakage detected ({len(overlap)} IDs): {preview}")


def assert_negatives_exclude_positives(
    positives: dict[str, set[str]], negatives: dict[str, list[str]]
) -> None:
    collisions: list[tuple[str, str]] = []
    for query_id, candidate_ids in negatives.items():
        for passage_id in candidate_ids:
            if passage_id in positives.get(query_id, set()):
                collisions.append((query_id, passage_id))
    if collisions:
        preview = ", ".join(f"{qid}/{pid}" for qid, pid in collisions[:5])
        raise ValueError(f"Positive passages sampled as negatives ({len(collisions)}): {preview}")


def build_subset_manifest(
    source_dir: str | Path,
    output_dir: str | Path,
    train_query_limit: int,
    dev_query_limit: int,
    seed: int,
    hard_negative_file: str = "train.mined.tsv",
) -> Path:
    """Create a deterministic local subset and write a JSON manifest.

    The function expects the official T2Ranking file names and never downloads data.
    Raw source files are referenced by hash rather than copied into version control.
    """
    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    required = {
        "collection": source / "collection.tsv",
        "train_queries": source / "queries.train.tsv",
        "dev_queries": source / "queries.dev.tsv",
        "train_qrels": source / "qrels.retrieval.train.tsv",
        "dev_qrels": source / "qrels.retrieval.dev.tsv",
        "hard_negatives": source / hard_negative_file,
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing official source files: " + ", ".join(missing))

    train_queries = read_tsv_mapping(required["train_queries"])
    dev_queries = read_tsv_mapping(required["dev_queries"])
    train_qrels = read_retrieval_qrels(required["train_qrels"])
    dev_qrels = read_retrieval_qrels(required["dev_qrels"])
    hard_negatives = read_hard_negatives(required["hard_negatives"])
    assert_disjoint_query_sets(train_queries, dev_queries)
    assert_negatives_exclude_positives(train_qrels, hard_negatives)

    rng = random.Random(seed)
    train_candidates = sorted(set(train_queries).intersection(train_qrels))
    dev_candidates = sorted(set(dev_queries).intersection(dev_qrels))
    selected_train = sorted(rng.sample(train_candidates, min(train_query_limit, len(train_candidates))))
    selected_dev = sorted(rng.sample(dev_candidates, min(dev_query_limit, len(dev_candidates))))
    assert_disjoint_query_sets(selected_train, selected_dev)

    selected_passage_ids = set()
    for query_id in [*selected_train, *selected_dev]:
        selected_passage_ids.update(train_qrels.get(query_id, set()))
        selected_passage_ids.update(dev_qrels.get(query_id, set()))
    for query_id in selected_train:
        for passage_id in hard_negatives.get(query_id, [])[:3]:
            selected_passage_ids.add(passage_id)

    collection = read_tsv_mapping(required["collection"])
    missing_passages = selected_passage_ids.difference(collection)
    if missing_passages:
        raise ValueError(f"Subset references {len(missing_passages)} passages absent from collection")

    manifest = {
        "dataset": "T2Ranking",
        "seed": seed,
        "source_dir": str(source),
        "source_hashes": {name: sha256_file(path) for name, path in required.items()},
        "selection": {
            "train_query_ids": selected_train,
            "dev_query_ids": selected_dev,
            "passage_ids": sorted(selected_passage_ids),
        },
        "counts": {
            "train_queries": len(selected_train),
            "dev_queries": len(selected_dev),
            "passages": len(selected_passage_ids),
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    write_rows(
        output / "train_queries.jsonl",
        (asdict(Query(query_id, train_queries[query_id])) for query_id in selected_train),
    )
    write_rows(
        output / "dev_queries.jsonl",
        (asdict(Query(query_id, dev_queries[query_id])) for query_id in selected_dev),
    )
    write_rows(
        output / "passages.jsonl",
        (asdict(Passage(passage_id, collection[passage_id])) for passage_id in sorted(selected_passage_ids)),
    )
    write_rows(
        output / "train_pairs.jsonl",
        (
            {
                "query_id": query_id,
                "query": train_queries[query_id],
                "positive_passage_id": passage_id,
                "positive_passage": collection[passage_id],
                "hard_negative_passage_ids": [
                    pid
                    for pid in hard_negatives.get(query_id, [])
                    if pid in collection and pid not in train_qrels[query_id]
                ][:3],
            }
            for query_id in selected_train
            for passage_id in sorted(train_qrels[query_id])[:1]
        ),
    )
    return manifest_path
