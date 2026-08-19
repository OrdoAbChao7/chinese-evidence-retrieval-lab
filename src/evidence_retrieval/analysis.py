"""Helpers for transparent failure-case export and analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def first_missed_relevant_rank(
    ranking: Sequence[str], relevance_by_passage: Mapping[str, float], cutoff: int = 10
) -> int | None:
    """Return the first rank containing a relevant passage, or None if it is missed."""
    for rank, passage_id in enumerate(ranking[:cutoff], start=1):
        if float(relevance_by_passage.get(passage_id, 0.0)) > 0:
            return rank
    return None


def build_error_records(
    queries: Mapping[str, str],
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
    passage_texts: Mapping[str, str],
    cutoff: int = 10,
) -> list[dict[str, Any]]:
    """Export missed queries for human annotation; no automatic cause is fabricated."""
    records: list[dict[str, Any]] = []
    for query_id, relevance_by_passage in qrels.items():
        ranking = rankings.get(query_id, [])
        if first_missed_relevant_rank(ranking, relevance_by_passage, cutoff) is not None:
            continue
        records.append(
            {
                "query_id": query_id,
                "query": queries.get(query_id, ""),
                "cutoff": cutoff,
                "retrieved": [
                    {"passage_id": passage_id, "text": passage_texts.get(passage_id, "")}
                    for passage_id in ranking[:cutoff]
                ],
                "relevant": [
                    {
                        "passage_id": passage_id,
                        "relevance": relevance,
                        "text": passage_texts.get(passage_id, ""),
                    }
                    for passage_id, relevance in sorted(relevance_by_passage.items())
                    if float(relevance) > 0
                ],
                "human_error_category": None,
                "notes": None,
            }
        )
    return records
