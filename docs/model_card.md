# Model Card Template

> Do not complete quantitative fields until a corresponding versioned run artifact exists.

## Identity

| Field | Value |
|---|---|
| Experiment ID | `TBD` |
| Base checkpoint | `TBD` |
| Git commit | `TBD` |
| Data manifest SHA-256 | `TBD` |
| Training seeds | `TBD` |
| License | `TBD` |

## Intended use

This model is intended to rank public Chinese passages for controlled retrieval experiments. It is not a fact checker, citation validator, safety classifier, or autonomous report generator.

## Training

Document the query/passages length caps, loss function, optimizer, learning rate, batch/effective batch size, epochs, hard-negative policy, device, and package versions. Link to the exact config file and run metadata.

## Evaluation

| Metric | E0 | E1 | E2 | E3 | E4 |
|---|---:|---:|---:|---:|---:|
| nDCG@10 | TBD | TBD | TBD | TBD | TBD |
| MRR@10 | TBD | TBD | TBD | TBD | TBD |
| Recall@50 | TBD | TBD | TBD | TBD | TBD |
| Mean latency (ms/query) | TBD | TBD | TBD | TBD | TBD |

Report mean ± standard deviation across completed seeds. State the exact development split and candidate depth.

## Limitations

Benchmark relevance does not prove factual correctness, document freshness, source authority, or usefulness in CountyResearchAI. Include at least 10 manually reviewed error examples and describe known failure categories before proposing upstream integration.
