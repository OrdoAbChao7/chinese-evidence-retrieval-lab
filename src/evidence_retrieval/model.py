"""PyTorch dual-encoder model with explicit pooling and normalization."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Attention-mask-aware mean pooling for [batch, sequence, hidden] tensors."""
    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must have shape [batch, sequence, hidden]")
    if attention_mask.shape != last_hidden_state.shape[:2]:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    weights = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * weights).sum(dim=1)
    denominator = weights.sum(dim=1).clamp_min(1e-9)
    return summed / denominator


class DualEncoder(nn.Module):
    """Shared-weights dense retriever with normalized query/passage representations."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs: Any = self.encoder(**inputs)
        if getattr(outputs, "pooler_output", None) is not None:
            representation = outputs.pooler_output
        else:
            representation = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        return F.normalize(representation, dim=-1)

    def forward(
        self, query_inputs: dict[str, torch.Tensor], passage_inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(query_inputs), self.encode(passage_inputs)


def load_transformer_dual_encoder(model_name: str) -> DualEncoder:
    """Lazily load a Transformer encoder so non-model utilities stay lightweight."""
    try:
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
        raise ImportError("Install the train extra: python -m pip install -e '.[train]'") from exc
    return DualEncoder(AutoModel.from_pretrained(model_name))
