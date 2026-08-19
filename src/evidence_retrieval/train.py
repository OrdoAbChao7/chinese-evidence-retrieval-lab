"""Minimal explicit PyTorch training loop for a dual-encoder retriever."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .losses import in_batch_contrastive_loss
from .reproducibility import runtime_metadata, write_json


def move_to_device(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    loss_values: list[float] = []
    accuracies: list[float] = []
    amp_enabled = scaler is not None and device.type == "cuda"
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        query_inputs = move_to_device(batch["query_inputs"], device)
        positive_inputs = move_to_device(batch["positive_inputs"], device)
        negative_inputs = (
            move_to_device(batch["negative_inputs"], device) if "negative_inputs" in batch else None
        )
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            query_embeddings, positive_embeddings = model(query_inputs, positive_inputs)
            negative_embeddings = model.encode(negative_inputs) if negative_inputs is not None else None
            loss, batch_metrics = in_batch_contrastive_loss(
                query_embeddings, positive_embeddings, temperature, negative_embeddings
            )
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        loss_values.append(float(loss.detach().cpu()))
        accuracies.append(batch_metrics["batch_retrieval_accuracy"])
    if not loss_values:
        raise ValueError("No batches were provided for training")
    return {
        "train_loss": sum(loss_values) / len(loss_values),
        "train_batch_retrieval_accuracy": sum(accuracies) / len(accuracies),
        "train_steps": float(len(loss_values)),
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    seed: int,
    data_manifest: str | Path | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": runtime_metadata(seed=seed, data_manifest=data_manifest),
        },
        output,
    )


def save_run_metadata(
    output_path: str | Path, seed: int, config: dict[str, Any], data_manifest: str | Path | None = None
) -> None:
    payload = runtime_metadata(seed=seed, data_manifest=data_manifest)
    payload["config"] = config
    write_json(output_path, payload)
