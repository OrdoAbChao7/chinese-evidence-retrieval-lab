# Data policy and local layout

This repository does **not** redistribute T2Ranking, private CountyResearchAI materials, API responses, model weights, or any raw production evidence. Obtain T2Ranking from its [official repository](https://github.com/THUIR/T2Ranking/) and follow the upstream Apache-2.0 license and citation requirements.

## Expected local layout

After downloading the upstream data, place only the `data/` directory contents here locally:

```text
data/
├── README.md                         # This file, version controlled
├── raw/                              # Ignored by Git
│   └── t2ranking/
│       └── data/
│           ├── collection.tsv
│           ├── queries.train.tsv
│           ├── queries.dev.tsv
│           ├── qrels.retrieval.train.tsv
│           ├── qrels.retrieval.dev.tsv
│           ├── train.bm25.tsv
│           └── train.mined.tsv
└── processed/                        # Ignored by Git
    └── smoke/
        ├── manifest.json
        ├── train_queries.jsonl
        ├── dev_queries.jsonl
        ├── passages.jsonl
        └── train_pairs.jsonl
```

## Manifest rule

Every prepared subset stores a `manifest.json` with source-file SHA-256 hashes, selected query/passage identifiers, seed, and counts. Commit a compact **example schema only** if useful; do not commit a manifest that reveals data text or a proprietary corpus.

## Data integrity rules

Training uses only official `queries.train.tsv` and `qrels.retrieval.train.tsv`. Development selection and final reporting use only the corresponding official development files. Never add development qrels to a negative-mining or training routine. The `test_no_leakage.py` suite protects the minimum invariants, but manual review remains required when data adapters change.
