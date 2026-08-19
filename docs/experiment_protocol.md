# Experiment Protocol

## Objective

Measure ranking quality on public Chinese passages before considering a retrieval component for CountyResearchAI. The unit of evaluation is **query-to-passage relevance**, not generated report correctness.

## Dataset and splits

Use the official T2Ranking retrieval split. Training consumes `queries.train.tsv`, `qrels.retrieval.train.tsv`, and a documented negative source. Development consumes `queries.dev.tsv` and `qrels.retrieval.dev.tsv` only. The development set is used for checkpoint selection and final comparisons during this project stage; it must never enter training, hard-negative mining, or prompt construction.

## Systems under comparison

| ID | Retrieval stack | Purpose |
|---|---|---|
| E0 | BM25 | Establish lexical behavior and a non-neural floor. |
| E1 | Frozen pretrained dual encoder | Measure zero/few-shot semantic retrieval. |
| E2 | E1 fine-tuned with in-batch negatives | Measure the value of transparent contrastive learning. |
| E3 | E2 plus one verified hard negative per query | Measure the value and risk of difficult negatives. |
| E4 | E3 Top-100 candidates reranked by Cross-Encoder | Measure final ranking quality while holding recall candidates fixed. |

## Controlled variables

For comparisons within a scale preset, keep the source manifest, query and passage length caps, candidate depth, base checkpoint, development qrels, metric code, hardware descriptor, and seeds unchanged. E2 versus E3 differs only in the inclusion of the declared hard-negative policy.

## Metrics

The primary metric is nDCG@10. Report MRR@10, Recall@50, Recall@100, average end-to-end query latency, peak allocated GPU memory where relevant, model checkpoint size, and three-seed mean ± standard deviation. Report candidate recall separately from reranking quality; a reranker cannot improve recall outside its candidate set.

## Repeated runs

Run each completed training configuration under seeds `42`, `2026`, and `3407`. Store raw per-seed metric JSON and model/checkpoint metadata. Select no result by visual inspection. If only one seed has completed, label the result `preliminary`.

## Error analysis

For the strongest and weakest completed model, export at least 10 development queries that miss every relevant passage in the Top-10. Manually classify each into a documented category such as lexical mismatch, semantic ambiguity, long-document truncation, fine-grained relevance boundary, or likely annotation limitation. Never infer an error cause from a metric alone.

## Claim policy

The README may claim only what a committed, versioned metric artifact supports. It may not claim improvements to CountyResearchAI report quality until an independent, end-to-end application evaluation is designed and completed.
