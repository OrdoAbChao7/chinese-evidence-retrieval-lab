# Project status and engineering notes

This document records the boundary between what this repository is intended to demonstrate, what has been verified in the repository, and what still needs evidence. It is deliberately more specific than a feature list.

## Current stage

**Experiment scaffold**

## Why this exists

I separated retrieval experiments from CountyResearchAI so that a ranking decision can be tested on public data before it changes an upstream research workflow.

## Scope and known limitations

The repository currently provides a protocol and scaffolded experiments; it does not claim benchmark gains until versioned runs are published. Ranking relevance is not factual correctness, and benchmark results do not automatically transfer to county research.

## Next evidence to collect

Run and publish E0 and E1 with environment manifests, seed-level metrics, latency, and at least ten manually classified retrieval failures.

## Maintenance rule

Future changes should describe one concrete behavior, include the smallest relevant verification step, and update this document whenever the project boundary changes.
