import pytest

from evidence_retrieval.data import assert_disjoint_query_sets, assert_negatives_exclude_positives


def test_train_and_dev_query_ids_must_be_disjoint():
    assert_disjoint_query_sets(["train-1", "train-2"], ["dev-1"])
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_query_sets(["shared"], ["shared"])


def test_positive_passages_cannot_be_hard_negatives():
    positives = {"q1": {"p-positive"}}
    assert_negatives_exclude_positives(positives, {"q1": ["p-negative"]})
    with pytest.raises(ValueError, match="Positive passages"):
        assert_negatives_exclude_positives(positives, {"q1": ["p-positive"]})
