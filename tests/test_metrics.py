from evidence_retrieval.evaluate import evaluate_rankings, mrr_at_k, ndcg_at_k, recall_at_k


def test_metrics_reward_correct_ranking():
    qrels = {"q1": {"p1": 3.0, "p2": 1.0}, "q2": {"p3": 2.0}}
    rankings = {"q1": ["p1", "p2", "p4"], "q2": ["p3", "p5"]}
    assert ndcg_at_k(rankings, qrels, k=10) == 1.0
    assert mrr_at_k(rankings, qrels, k=10) == 1.0
    assert recall_at_k(rankings, qrels, k=10) == 1.0


def test_metrics_penalize_missing_relevant_passages():
    qrels = {"q1": {"p1": 1.0}}
    rankings = {"q1": ["p2", "p3"]}
    report = evaluate_rankings(rankings, qrels)
    assert report["ndcg@10"] == 0.0
    assert report["mrr@10"] == 0.0
    assert report["recall@50"] == 0.0
