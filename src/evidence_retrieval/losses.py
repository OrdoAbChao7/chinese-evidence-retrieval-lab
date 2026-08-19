"""Explicit contrastive objectives for dense retrieval."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def in_batch_contrastive_loss(
    query_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    temperature: float = 0.05,
    hard_negative_embeddings: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute cross-entropy over in-batch positives and optional per-query negatives.

    For a batch of B aligned query/positive pairs, diagonal entries are positives and all
    other columns serve as in-batch negatives. If hard negatives are supplied, the B hard
    passages are appended as additional columns; they are never targets.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if query_embeddings.ndim != 2 or positive_embeddings.ndim != 2:
        raise ValueError("Embeddings must have shape [batch, hidden]")
    if query_embeddings.shape != positive_embeddings.shape:
        raise ValueError("Query and positive embeddings must have identical shape")
    logits = query_embeddings @ positive_embeddings.T / temperature
    if hard_negative_embeddings is not None:
        if hard_negative_embeddings.shape != positive_embeddings.shape:
            raise ValueError("Hard-negative embeddings must match positive embedding shape")
        hard_logits = query_embeddings @ hard_negative_embeddings.T / temperature
        logits = torch.cat([logits, hard_logits], dim=1)
    targets = torch.arange(query_embeddings.shape[0], device=query_embeddings.device)
    loss = F.cross_entropy(logits, targets)
    retrieval_accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
    return loss, {"batch_retrieval_accuracy": retrieval_accuracy, "logit_columns": float(logits.shape[1])}
