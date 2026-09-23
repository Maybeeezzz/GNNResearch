"""Run isolated BP and ForwardGNN local-learning experiments."""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer"])
    parser.add_argument("--models", nargs="+", choices=("gcn", "residual-gcn", "forward-gcn"), default=["gcn", "forward-gcn"])
    parser.add_argument("--methods", nargs="+", choices=("bp", "forwardgnn"), default=["bp", "forwardgnn"])
    parser.add_argument("--scopes", nargs="+", choices=("all", "output"), default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--pretrain-steps", type=int, default=0)
    parser.add_argument("--pretrain-fraction", type=float, default=0.0)
    parser.add_argument("--results", type=Path, default=Path("results/forwardgnn"))
    parser.add_argument("--layers", type=int)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--forward-lr", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def learning_rate(method: str, model: str, scope: str) -> float:
    if method == "bp":
        return 0.01
    return 1e-3


def main() -> None:
    args = parse_args()
    combinations = list(
        itertools.product(args.datasets, args.models, args.methods, args.scopes, args.seeds)
    )
    combinations = [entry for entry in combinations if valid_combination(entry, args)]
    if not combinations:
        raise ValueError("no valid configurations: forwardgnn needs forward-gcn, scope=all, no pretraining")
    for index, (dataset, model, method, scope, seed) in enumerate(combinations, 1):
        layers = args.layers if args.layers is not None else (8 if model == "residual-gcn" else 2)
        if model == "gcn":
            layers = 2
        lr = args.forward_lr if method == "forwardgnn" else learning_rate(method, model, scope)
        output = args.results / (
            f"{dataset}_{model}_{method}_{scope}_pre{args.pretrain_steps}"
            f"_frac{args.pretrain_fraction:g}_L{layers}_H{args.hidden}_steps{args.steps}"
            f"_lr{lr:g}_tau{args.temperature:g}_eval{args.eval_every}_seed{seed}.json"
        )
        if output.exists() and not args.force:
            print(f"[{index}/{len(combinations)}] skip {output.name}")
            continue
        print(f"[{index}/{len(combinations)}] run  {output.name}", flush=True)
        command = [
            sys.executable,
            "-m", "node_classification.experiments.baselines.worker",
            "--dataset", dataset,
            "--model", model,
            "--method", method,
            "--scope", scope,
            "--seed", str(seed),
            "--steps", str(args.steps),
            "--eval-every", str(args.eval_every),
            "--pretrain-steps", str(args.pretrain_steps),
            "--pretrain-fraction", str(args.pretrain_fraction),
            "--lr", str(lr),
            "--layers", str(layers),
            "--hidden", str(args.hidden),
            "--temperature", str(args.temperature),
            "--output", str(output),
        ]
        subprocess.run(command, check=True)


def valid_combination(entry, args) -> bool:
    _, model, method, scope, _ = entry
    if method == "forwardgnn" and model != "forward-gcn":
        return False
    if model == "forward-gcn":
        return (method in ("bp", "forwardgnn") and scope == "all"
                and not args.pretrain_steps and not args.pretrain_fraction)
    return True


if __name__ == "__main__":
    main()
