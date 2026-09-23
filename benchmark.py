"""Fair timing comparison of standard and single-forward GCN training."""

import argparse
import random
import statistics
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from node_classification.data import generate_sbm_graph
from node_classification.models.gcn import GCN, ResidualGCN
from node_classification.training import TrainResult, train_model
from train import choose_device


@dataclass
class Run:
    method: str
    seconds: float
    result: TrainResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=1200)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--model", choices=("gcn", "residual-gcn"), default="gcn")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_once(
    method: str,
    initial_state: dict,
    data,
    args: argparse.Namespace,
    device: torch.device,
) -> Run:
    model = build_model(args, data).to(device)
    model.load_state_dict(initial_state)
    synchronize(device)
    started = time.perf_counter()
    result = train_model(
        model,
        data,
        epochs=args.epochs,
        patience=args.epochs + 1,
        verbose=False,
        method=method,
    )
    synchronize(device)
    return Run(method, time.perf_counter() - started, result)


def build_model(args: argparse.Namespace, data) -> torch.nn.Module:
    if args.model == "residual-gcn":
        return ResidualGCN(
            data.x.size(1),
            args.hidden,
            data.num_classes,
            num_layers=args.layers,
            dropout=args.dropout,
        )
    return GCN(data.x.size(1), args.hidden, data.num_classes, dropout=args.dropout)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.repeats < 1:
        raise ValueError("epochs and repeats must be positive")
    set_seed(args.seed)
    device = choose_device(args.device)
    data = generate_sbm_graph(
        num_nodes=args.nodes,
        num_features=args.features,
        seed=args.seed,
    ).to(device)
    template = build_model(args, data).to(device)
    initial_state = template.state_dict()
    parameter_count = sum(parameter.numel() for parameter in template.parameters())

    # Untimed warm-up avoids charging one method for lazy runtime initialization.
    for method in ("single-forward", "standard"):
        warmup_args = argparse.Namespace(**vars(args))
        warmup_args.epochs = min(3, args.epochs)
        run_once(method, initial_state, data, warmup_args, device)
    runs: List[Run] = []
    # Alternate order to reduce bias from temperature and background activity.
    for repeat in range(args.repeats):
        methods = ("standard", "single-forward")
        if repeat % 2:
            methods = tuple(reversed(methods))
        for method in methods:
            set_seed(args.seed + repeat)
            runs.append(run_once(method, initial_state, data, args, device))

    print(
        f"device={device} model={args.model} layers="
        f"{args.layers if args.model == 'residual-gcn' else 2} "
        f"parameters={parameter_count} nodes={data.x.size(0)} "
        f"edges={data.edge_index.size(1)} epochs={args.epochs} "
        f"repeats={args.repeats} dropout={args.dropout}"
    )
    summaries = {}
    for method in ("standard", "single-forward"):
        selected = [run for run in runs if run.method == method]
        times = [run.seconds for run in selected]
        mean_seconds = statistics.mean(times)
        summaries[method] = mean_seconds
        print(
            f"{method:14s} mean={mean_seconds:.4f}s "
            f"stdev={statistics.stdev(times) if len(times) > 1 else 0.0:.4f}s "
            f"ms/epoch={mean_seconds / args.epochs * 1000:.3f} "
            f"epochs/s={args.epochs / mean_seconds:.2f} "
            f"test_acc={statistics.mean(run.result.test_accuracy for run in selected):.3f}"
        )
    standard = summaries["standard"]
    single = summaries["single-forward"]
    print(
        f"speedup={standard / single:.3f}x "
        f"time_saved={(standard - single) / standard * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
