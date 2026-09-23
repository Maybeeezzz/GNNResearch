import argparse
import random
from pathlib import Path

import numpy as np
import torch

from node_classification.data import (
    generate_sbm_graph,
    load_npz_graph,
    load_planetoid_graph,
)
from node_classification.models.gcn import GCN, ResidualGCN
from node_classification.models.forward_gnn import ForwardGNN, train_forwardgnn
from node_classification.training import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GCN for node classification")
    parser.add_argument("--data", type=Path, help="optional graph data in .npz format")
    parser.add_argument("--dataset", choices=("Cora", "CiteSeer", "PubMed"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--model", choices=("gcn", "residual-gcn", "forward-gcn"), default=None)
    parser.add_argument("--layers", type=int, help="default: residual-gcn=8, forward-gcn=2")
    parser.add_argument("--dropout", type=float, help="default: BP=0.5; ForwardGNN=0 (required)")
    parser.add_argument("--lr", type=float, help="default: BP=0.01, ForwardGNN=0.001")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--edge-direction", choices=("bidirection", "unidirection"), default="bidirection")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--method",
        choices=("standard", "single-forward", "forwardgnn"),
        default="standard",
        help="forwardgnn is layer-local learning; single-forward is the legacy BP validation reuse",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.data and args.dataset:
        parser.error("choose --data or --dataset, not both")
    args.model = args.model or ("forward-gcn" if args.method == "forwardgnn" else "gcn")
    if (args.method == "forwardgnn") != (args.model == "forward-gcn"):
        parser.error("--method forwardgnn requires --model forward-gcn (selected automatically)")
    args.layers = args.layers if args.layers is not None else (8 if args.model == "residual-gcn" else 2)
    args.lr = args.lr if args.lr is not None else (0.001 if args.method == "forwardgnn" else 0.01)
    args.dropout = args.dropout if args.dropout is not None else (0.0 if args.method == "forwardgnn" else 0.5)
    if args.method == "forwardgnn" and args.dropout:
        parser.error("forwardgnn follows SF-GCN without dropout; use --dropout 0")
    args.output = args.output or Path(f"checkpoints/{args.model}_{args.method}.pt")
    return args


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dataset:
        data = load_planetoid_graph(args.dataset)
    else:
        data = load_npz_graph(args.data, args.seed) if args.data else generate_sbm_graph(seed=args.seed)
    device = choose_device(args.device)
    data = data.to(device)
    model_class = ResidualGCN if args.model == "residual-gcn" else GCN
    model_kwargs = {
        "in_features": data.x.size(1),
        "hidden_features": args.hidden,
        "num_classes": data.num_classes,
        "dropout": args.dropout,
    }
    if args.model == "residual-gcn":
        model_kwargs["num_layers"] = args.layers
    if args.method == "forwardgnn":
        model_kwargs.pop("dropout")
        model = ForwardGNN(**model_kwargs, num_layers=args.layers,
                           temperature=args.temperature, edge_direction=args.edge_direction).to(device)
    else:
        model = model_class(**model_kwargs).to(device)

    print(
        f"device={device} nodes={data.x.size(0)} edges={data.edge_index.size(1)} "
        f"features={data.x.size(1)} classes={data.num_classes}"
    )
    if args.method == "forwardgnn":
        result = train_forwardgnn(model, data, epochs=args.epochs, learning_rate=args.lr,
                                  weight_decay=args.weight_decay, patience=args.patience)
        print(f"layer_epochs={result.layer_epochs} local_updates={result.local_backward_passes}")
    else:
        result = train_model(
            model,
            data,
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            method=args.method,
        )
    print(
        f"best_epoch={result.best_epoch} train_acc={result.train_accuracy:.3f} "
        f"val_acc={result.val_accuracy:.3f} test_acc={result.test_accuracy:.3f}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_features": data.x.size(1),
            "hidden_features": args.hidden,
            "num_classes": data.num_classes,
            "dropout": 0.0 if args.method == "forwardgnn" else args.dropout,
            "model": args.model,
            "method": args.method,
            "num_layers": args.layers if args.model != "gcn" else 2,
            "temperature": args.temperature,
            "edge_direction": args.edge_direction,
        },
        args.output,
    )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
