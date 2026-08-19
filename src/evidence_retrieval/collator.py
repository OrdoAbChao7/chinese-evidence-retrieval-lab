"""Batch collation for transparent dual-encoder training."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch


class RetrievalCollator:
    """Tokenize query/positive/(optional) negative text without hidden state.

    ``tokenizer`` must accept a list of strings and return a mapping of tensors, matching
    the Hugging Face tokenizer interface. Keeping it injectable makes unit tests independent
    from a network-downloaded model.
    """

    def __init__(
        self,
        tokenizer: Callable[..., dict[str, torch.Tensor]],
        query_max_length: int = 32,
        passage_max_length: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length

    def _encode(self, texts: Sequence[str], max_length: int) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        queries = [str(example["query"]) for example in examples]
        positives = [str(example["positive_passage"]) for example in examples]
        output: dict[str, Any] = {
            "query_inputs": self._encode(queries, self.query_max_length),
            "positive_inputs": self._encode(positives, self.passage_max_length),
            "query_ids": [str(example.get("query_id", index)) for index, example in enumerate(examples)],
        }
        negatives = [example.get("hard_negative_passage") for example in examples]
        if any(negative is not None for negative in negatives):
            if not all(isinstance(negative, str) and negative for negative in negatives):
                raise ValueError(
                    "Hard-negative batches must provide one non-empty negative for every example"
                )
            output["negative_inputs"] = self._encode(negatives, self.passage_max_length)
        return output
