#!/usr/bin/env python3
"""Train a configured dual encoder on a prepared local subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from evidence_retrieval.collator import RetrievalCollator
from evidence_retrieval.model import load_transformer_dual_encoder
from evidence_retrieval.reproducibility import set_global_seed, write_json
from evidence_retrieval.train import save_checkpoint, save_run_metadata, train_one_epoch


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_examples(
    train_pairs_path: Path, passages_path: Path, use_hard_negatives: bool
) -> list[dict[str, str]]:
    passages = {row["passage_id"]: row["text"] for row in read_jsonl(passages_path)}
    examples: list[dict[str, str]] = []
    for row in read_jsonl(train_pairs_path):
        example = {
            "query_id": str(row["query_id"]),
            "query": str(row["query"]),
            "positive_passage": str(row["positive_passage"]),
        }
        if use_hard_negatives:
            candidates = [
                passage_id
                for passage_id in row.get("hard_negative_passage_ids", [])
                if passage_id in passages
            ]
            if not candidates:
                continue
            example["hard_negative_passage"] = passages[candidates[0]]
        examples.append(example)
    if not examples:
        raise ValueError(
            "No trainable examples were found. Check prepared data and hard-negative configuration."
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None, help="Defaults to cuda if available, otherwise cpu")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    experiment_id = config["experiment"]["id"]
    if experiment_id not in {"e2_inbatch", "e3_hard_negative"}:
        raise ValueError("train.py currently supports e2_inbatch and e3_hard_negative configurations")

    set_global_seed(args.seed, deterministic=bool(config["reproducibility"].get("deterministic", True)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_config = config["data"]
    train_pairs = Path(data_config["train_pairs"])
    processed_dir = Path(data_config["processed_dir"])
    examples = build_examples(
        train_pairs, processed_dir / "passages.jsonl", use_hard_negatives=experiment_id == "e3_hard_negative"
    )

    from transformers import AutoTokenizer  # Optional dependency, required only for full training.

    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_config["checkpoint"])
    collator = RetrievalCollator(
        tokenizer,
        query_max_length=int(model_config["query_max_length"]),
        passage_max_length=int(model_config["passage_max_length"]),
    )
    training_config = config["training"]
    loader = DataLoader(
        examples, batch_size=int(training_config["batch_size"]), shuffle=True, collate_fn=collator
    )
    model = load_transformer_dual_encoder(model_config["checkpoint"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and bool(training_config.get("amp", True))
    )
    run_dir = Path(config["output"]["run_dir"]) / f"seed_{args.seed}"
    manifest = Path(data_config["manifest"])
    save_run_metadata(run_dir / "run_metadata.json", args.seed, config, manifest)

    history: list[dict[str, float]] = []
    for epoch in range(1, int(training_config["epochs"]) + 1):
        metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            temperature=float(training_config["temperature"]),
            scaler=scaler,
        )
        metrics["epoch"] = float(epoch)
        history.append(metrics)
        save_checkpoint(run_dir / f"epoch_{epoch}.pt", model, optimizer, epoch, metrics, args.seed, manifest)
        print({"experiment": experiment_id, "seed": args.seed, **metrics})
    write_json(
        run_dir / "training_history.json",
        {"experiment": experiment_id, "seed": args.seed, "history": history},
    )


if __name__ == "__main__":
    main()
