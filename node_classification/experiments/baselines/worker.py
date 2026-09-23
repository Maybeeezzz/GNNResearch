"""Run one isolated GNN optimization experiment and write a JSON record."""

import argparse
import json
import random
import threading
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import psutil
import torch
from torch import nn

from node_classification.data import load_planetoid_graph
from node_classification.models.gcn import GCN, ResidualGCN
from node_classification.models.forward_gnn import ForwardGNN, train_forwardgnn
from node_classification.training import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("Cora", "CiteSeer", "PubMed"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/Planetoid"))
    parser.add_argument("--model", choices=("gcn", "residual-gcn", "forward-gcn"), required=True)
    parser.add_argument("--method", choices=("bp", "forwardgnn"), required=True)
    parser.add_argument("--scope", choices=("all", "output"), default="all")
    parser.add_argument("--steps", type=int, default=500,
                        help="BP full-network updates; ForwardGNN updates PER LAYER")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--edge-direction", choices=("bidirection", "unidirection"), default="bidirection")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pretrain-steps", type=int, default=0)
    parser.add_argument(
        "--pretrain-fraction",
        type=float,
        default=0.0,
        help="stratified fraction of public training nodes reserved for pretraining",
    )
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.layers = args.layers if args.layers is not None else (8 if args.model == "residual-gcn" else 2)
    if args.steps < 1 or args.eval_every < 1:
        parser.error("steps and eval-every must be positive")
    if args.method == "forwardgnn" and args.model != "forward-gcn":
        parser.error("forwardgnn requires --model forward-gcn")
    if args.model == "forward-gcn":
        if args.scope != "all" or args.pretrain_steps or args.pretrain_fraction:
            parser.error("forward-gcn supports bp/forwardgnn from scratch with --scope all")
        if args.dropout:
            parser.error("forward-gcn follows SF-GCN without dropout")
    return args


def build_model(args: argparse.Namespace, in_features: int, classes: int) -> nn.Module:
    if args.model == "forward-gcn":
        return ForwardGNN(in_features, args.hidden, classes, num_layers=args.layers,
                          temperature=args.temperature, edge_direction=args.edge_direction)
    if args.model == "residual-gcn":
        return ResidualGCN(
            in_features,
            args.hidden,
            classes,
            num_layers=args.layers,
            dropout=args.dropout,
        )
    return GCN(in_features, args.hidden, classes, dropout=args.dropout)


def configure_scope(model: nn.Module, scope: str) -> int:
    selected = 0
    for name, parameter in model.named_parameters():
        is_output = "output_conv" in name or "conv2" in name
        parameter.requires_grad_(scope == "all" or is_output)
        if parameter.requires_grad:
            selected += parameter.numel()
    return selected


def staged_masks(data, fraction: float, seed: int):
    if not 0.0 < fraction < 1.0:
        raise ValueError("pretrain-fraction must be between 0 and 1")
    generator = torch.Generator().manual_seed(seed)
    pretrain_mask = torch.zeros_like(data.train_mask)
    finetune_mask = torch.zeros_like(data.train_mask)
    for class_id in torch.unique(data.y):
        indices = torch.where(data.train_mask & (data.y == class_id))[0]
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        split = max(1, min(indices.numel() - 1, int(indices.numel() * fraction)))
        pretrain_mask[indices[:split]] = True
        finetune_mask[indices[split:]] = True
    return pretrain_mask, finetune_mask


class MemoryMonitor:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.wait(0.002):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> "MemoryMonitor":
        self.thread.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    data = load_planetoid_graph(args.dataset, args.data_root)
    model = build_model(args, data.x.size(1), data.num_classes)
    if isinstance(model, ForwardGNN):
        model.bind_graph(data)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    finetune_mask = data.train_mask
    if args.pretrain_fraction:
        pretrain_mask, finetune_mask = staged_masks(
            data, args.pretrain_fraction, args.seed
        )
        data.train_mask = pretrain_mask
    if args.pretrain_steps:
        pretrain_optimizer = torch.optim.Adam(
            model.parameters(), lr=0.01, weight_decay=args.weight_decay
        )
        pretrain_criterion = nn.CrossEntropyLoss()
        for _ in range(args.pretrain_steps):
            model.train()
            pretrain_optimizer.zero_grad(set_to_none=True)
            pretrain_logits = model(data.x, data.edge_index)
            pretrain_loss = pretrain_criterion(
                pretrain_logits[data.train_mask], data.y[data.train_mask]
            )
            pretrain_loss.backward()
            pretrain_optimizer.step()
        del pretrain_optimizer
        for parameter in model.parameters():
            parameter.grad = None
    if args.pretrain_fraction:
        data.train_mask = finetune_mask
    trainable_parameters = configure_scope(model, args.scope)
    history: List[Dict[str, float]] = []
    total_updates = args.steps * (args.layers if args.method == "forwardgnn" else 1)

    def record(step: int, objective: float, current_model: nn.Module) -> None:
        if step != total_updates and step % args.eval_every:
            return
        metrics = classification_metrics(current_model, data)
        history.append({"step": step, "objective": objective,
                        "active_layers": int(current_model.trained_layers) if isinstance(current_model, ForwardGNN) else None,
                        **metrics})

    initial_metrics = classification_metrics(model, data)
    history.append({"step": 0, "objective": float("nan"), **initial_metrics})

    with MemoryMonitor() as memory:
        started = time.perf_counter()
        if args.method == "bp":
            optimizer = torch.optim.Adam(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            criterion = nn.CrossEntropyLoss()
            for step in range(1, args.steps + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(data.x, data.edge_index)
                loss = criterion(logits[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                record(step, float(loss.item()), model)
        elif args.method == "forwardgnn":
            forward_result = train_forwardgnn(
                model, data, epochs=args.steps, learning_rate=args.lr,
                weight_decay=args.weight_decay, patience=-1, verbose=False,
                callback=record,
            )
        wall_seconds = time.perf_counter() - started

    final_metrics = classification_metrics(model, data)
    optimization_forwards = total_updates
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "method": args.method,
        "scope": args.scope,
        "seed": args.seed,
        "steps": args.steps,
        "step_unit": "local_layer_update" if args.method == "forwardgnn" else "full_network_update",
        "steps_per_layer": args.steps if args.method == "forwardgnn" else None,
        "total_updates": total_updates,
        "eval_every": args.eval_every,
        "hidden": args.hidden,
        "layers": args.layers if args.model != "gcn" else 2,
        "temperature": args.temperature if args.model == "forward-gcn" else None,
        "edge_direction": args.edge_direction if args.model == "forward-gcn" else None,
        "dropout": args.dropout,
        "pretrain_steps": args.pretrain_steps,
        "pretrain_fraction": args.pretrain_fraction,
        "learning_rate": args.lr,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "nodes": data.x.size(0),
        "edges": data.edge_index.size(1),
        "features": data.x.size(1),
        "classes": data.num_classes,
        "optimization_forwards": optimization_forwards,
        "optimization_forward_unit": "single_layer" if args.method == "forwardgnn" else "full_network",
        "optimization_layer_forwards": optimization_forwards * (1 if args.method == "forwardgnn" else (args.layers if args.model != "gcn" else 2)),
        "backward_passes": args.steps if args.method == "bp" else 0,
        "local_backward_passes": forward_result.local_backward_passes if args.method == "forwardgnn" else 0,
        "cache_layer_forwards": forward_result.cache_layer_forwards if args.method == "forwardgnn" else 0,
        "checkpoint_selection": "final_fixed_budget",
        "timing_includes": "optimization, scheduled evaluation, and ForwardGNN cache/final evaluation",
        "wall_seconds": wall_seconds,
        "milliseconds_per_step": wall_seconds / total_updates * 1000,
        "peak_rss_mb": memory.peak / (1024**2),
        "incremental_peak_rss_mb": (memory.peak - memory.baseline) / (1024**2),
        **final_metrics,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "history"}))


if __name__ == "__main__":
    main()
