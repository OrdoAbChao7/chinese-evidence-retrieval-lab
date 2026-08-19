"""Reproducible building blocks for Chinese evidence retrieval experiments."""

from .evaluate import evaluate_rankings, mrr_at_k, ndcg_at_k, recall_at_k
from .reproducibility import runtime_metadata, set_global_seed

__all__ = [
    "evaluate_rankings",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "runtime_metadata",
    "set_global_seed",
]
