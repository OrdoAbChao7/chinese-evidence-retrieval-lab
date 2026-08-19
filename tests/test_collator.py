# ruff: noqa: E402

import pytest

torch = pytest.importorskip("torch")

from evidence_retrieval.collator import RetrievalCollator


class TinyTokenizer:
    def __call__(self, texts, **kwargs):
        max_length = kwargs["max_length"]
        width = min(max_length, max(len(text) for text in texts))
        return {
            "input_ids": torch.ones((len(texts), width), dtype=torch.long),
            "attention_mask": torch.ones((len(texts), width), dtype=torch.long),
        }


def test_collator_encodes_queries_and_positives():
    collator = RetrievalCollator(TinyTokenizer(), query_max_length=8, passage_max_length=16)
    batch = collator(
        [
            {"query_id": "q1", "query": "问题", "positive_passage": "正确段落"},
            {"query_id": "q2", "query": "另一个问题", "positive_passage": "另一个正确段落"},
        ]
    )
    assert batch["query_inputs"]["input_ids"].shape[0] == 2
    assert batch["positive_inputs"]["input_ids"].shape[0] == 2
    assert "negative_inputs" not in batch


def test_collator_requires_complete_hard_negative_batch():
    collator = RetrievalCollator(TinyTokenizer())
    batch = collator(
        [
            {"query": "问题", "positive_passage": "正例", "hard_negative_passage": "负例"},
            {"query": "问题二", "positive_passage": "正例二", "hard_negative_passage": "负例二"},
        ]
    )
    assert batch["negative_inputs"]["input_ids"].shape[0] == 2
