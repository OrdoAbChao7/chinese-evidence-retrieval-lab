# ruff: noqa: E402

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from evidence_retrieval.losses import in_batch_contrastive_loss
from evidence_retrieval.model import DualEncoder


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)

    def forward(self, input_ids, attention_mask):
        class Output:
            pass

        output = Output()
        output.pooler_output = None
        output.last_hidden_state = self.embedding(input_ids)
        return output


def test_dual_encoder_smoke_step():
    model = DualEncoder(TinyEncoder())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    query_inputs = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.ones((2, 2), dtype=torch.long),
    }
    passage_inputs = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.ones((2, 2), dtype=torch.long),
    }
    queries, positives = model(query_inputs, passage_inputs)
    loss, _ = in_batch_contrastive_loss(queries, positives)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
