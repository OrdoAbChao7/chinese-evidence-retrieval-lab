# ruff: noqa: E402

import pytest

torch = pytest.importorskip("torch")

from evidence_retrieval.losses import in_batch_contrastive_loss


def test_contrastive_loss_prefers_aligned_pairs_and_backpropagates():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    positives = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    loss, metrics = in_batch_contrastive_loss(queries, positives, temperature=0.1)
    assert loss.item() < 0.01
    assert metrics["batch_retrieval_accuracy"] == 1.0
    loss.backward()
    assert queries.grad is not None


def test_hard_negatives_extend_logit_columns():
    queries = torch.eye(2, requires_grad=True)
    positives = torch.eye(2, requires_grad=True)
    negatives = torch.tensor([[0.9, 0.1], [0.1, 0.9]], requires_grad=True)
    _, metrics = in_batch_contrastive_loss(queries, positives, hard_negative_embeddings=negatives)
    assert metrics["logit_columns"] == 4.0
