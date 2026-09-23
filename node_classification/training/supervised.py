from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
from torch import Tensor, nn

from ..data import GraphData


@dataclass
class TrainResult:
    best_epoch: int
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float


def accuracy(logits: Tensor, labels: Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def macro_f1(logits: Tensor, labels: Tensor) -> float:
    predictions = logits.argmax(dim=1)
    scores = []
    for class_id in torch.unique(labels):
        predicted = predictions == class_id
        actual = labels == class_id
        true_positive = (predicted & actual).sum().float()
        precision = true_positive / predicted.sum().clamp_min(1)
        recall = true_positive / actual.sum().clamp_min(1)
        scores.append(2 * precision * recall / (precision + recall).clamp_min(1e-12))
    return float(torch.stack(scores).mean().item())


@torch.no_grad()
def classification_metrics(model: nn.Module, data: GraphData) -> Dict[str, float]:
    model.eval()
    logits = model(data.x, data.edge_index)
    metrics: Dict[str, float] = {}
    for split, mask in (
        ("train", data.train_mask),
        ("val", data.val_mask),
        ("test", data.test_mask),
    ):
        metrics[f"{split}_accuracy"] = accuracy(logits[mask], data.y[mask])
        metrics[f"{split}_macro_f1"] = macro_f1(logits[mask], data.y[mask])
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, data: GraphData) -> Dict[str, float]:
    model.eval()
    logits = model(data.x, data.edge_index)
    return {
        split: accuracy(logits[mask], data.y[mask])
        for split, mask in (
            ("train", data.train_mask),
            ("val", data.val_mask),
            ("test", data.test_mask),
        )
    }


def train_model(
    model: nn.Module,
    data: GraphData,
    epochs: int = 300,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 50,
    verbose: bool = True,
    method: Literal["standard", "single-forward"] = "standard",
) -> TrainResult:
    """Train a node classifier using one of two validation strategies.

    ``standard`` evaluates validation loss with a second forward pass after the
    optimizer step. ``single-forward`` reuses the training forward pass, cutting
    the number of full-graph forward passes per epoch from two to one. In the
    latter case validation loss is measured before that epoch's optimizer step.
    """
    if method not in ("standard", "single-forward"):
        raise ValueError("method must be 'standard' or 'single-forward'")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, Tensor]] = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])

        if method == "single-forward":
            with torch.no_grad():
                val_loss = criterion(
                    logits[data.val_mask], data.y[data.val_mask]
                ).item()
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_epoch = epoch
                # The validation output belongs to the current, pre-update state.
                best_state = deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

        loss.backward()
        optimizer.step()

        if method == "standard":
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                val_loss = criterion(
                    logits[data.val_mask], data.y[data.val_mask]
                ).item()
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

        if verbose and (epoch == 1 or epoch % 25 == 0):
            scores = evaluate(model, data)
            print(
                f"method={method} epoch={epoch:03d} train_loss={loss.item():.4f} "
                f"val_loss={val_loss:.4f} val_acc={scores['val']:.3f}"
            )
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a model state")
    model.load_state_dict(best_state)
    scores = evaluate(model, data)
    return TrainResult(
        best_epoch=best_epoch,
        train_accuracy=scores["train"],
        val_accuracy=scores["val"],
        test_accuracy=scores["test"],
    )
