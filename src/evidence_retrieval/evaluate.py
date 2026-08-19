"""Pure-Python retrieval metrics with graded relevance support."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

Qrels = Mapping[str, Mapping[str, float]]
Rankings = Mapping[str, Sequence[str]]


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    return sum(
        (2.0**relevance - 1.0) / math.log2(position + 2) for position, relevance in enumerate(relevances[:k])
    )


def ndcg_at_k(rankings: Rankings, qrels: Qrels, k: int = 10) -> float:
    scores: list[float] = []
    for query_id, relevance_by_passage in qrels.items():
        ranked = rankings.get(query_id, [])
        observed = [float(relevance_by_passage.get(passage_id, 0.0)) for passage_id in ranked]
        ideal = sorted((float(value) for value in relevance_by_passage.values()), reverse=True)
        denominator = dcg_at_k(ideal, k)
        scores.append(0.0 if denominator == 0 else dcg_at_k(observed, k) / denominator)
    return sum(scores) / len(scores) if scores else 0.0


def mrr_at_k(rankings: Rankings, qrels: Qrels, k: int = 10, min_relevance: float = 1.0) -> float:
    scores: list[float] = []
    for query_id, relevance_by_passage in qrels.items():
        reciprocal_rank = 0.0
        for rank, passage_id in enumerate(rankings.get(query_id, [])[:k], start=1):
            if float(relevance_by_passage.get(passage_id, 0.0)) >= min_relevance:
                reciprocal_rank = 1.0 / rank
                break
        scores.append(reciprocal_rank)
    return sum(scores) / len(scores) if scores else 0.0


def recall_at_k(rankings: Rankings, qrels: Qrels, k: int = 50, min_relevance: float = 1.0) -> float:
    scores: list[float] = []
    for query_id, relevance_by_passage in qrels.items():
        relevant = {
            passage_id for passage_id, value in relevance_by_passage.items() if float(value) >= min_relevance
        }
        if not relevant:
            continue
        retrieved = set(rankings.get(query_id, [])[:k])
        scores.append(len(relevant.intersection(retrieved)) / len(relevant))
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_rankings(rankings: Rankings, qrels: Qrels) -> dict[str, float]:
    """Return the fixed project metrics for one ranking run."""
    return {
        "ndcg@10": ndcg_at_k(rankings, qrels, k=10),
        "mrr@10": mrr_at_k(rankings, qrels, k=10),
        "recall@50": recall_at_k(rankings, qrels, k=50),
        "recall@100": recall_at_k(rankings, qrels, k=100),
    }


def qrels_from_pairs(pairs: Sequence[tuple[str, str, float]]) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for query_id, passage_id, relevance in pairs:
        qrels[query_id][passage_id] = float(relevance)
    return dict(qrels)
