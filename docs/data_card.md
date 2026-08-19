# Data Card: T2Ranking usage

## Intended use

T2Ranking is used solely as a public Chinese passage-retrieval and reranking benchmark for E0–E4. It is not a county-industry dataset and must not be presented as one. The project uses its official retrieval train/development files to learn and evaluate query-passage relevance.

## Source and license

The data source is the [official T2Ranking repository](https://github.com/THUIR/T2Ranking/), which documents an Apache-2.0 license, train/development file layout, retrieval qrels, and negative files. Users must cite the upstream benchmark and independently confirm they may use the data for their purpose.

## Processing

Raw source files stay in `data/raw/`, which is Git-ignored. `scripts/prepare_subset.py` records SHA-256 source hashes and deterministic selected IDs in a local manifest. It exports a local JSONL working subset; it does not download, upload, or transform data with an LLM.

## Split and leakage policy

The official query split is preserved. Query IDs must be disjoint between training and development. A passage that is an official positive for a query cannot be used as its hard negative. These invariants are covered by automated tests but must also be checked after any adapter change.

## Limitations and risks

The benchmark represents Chinese search-style passage ranking, not every domain that CountyResearchAI may encounter. Relevance labels do not prove factual accuracy, recency, provenance quality, safety, or suitability for a final research conclusion. Benchmark results should not be generalized to private or sensitive data.
