"""Node-classification optimization methods."""

from .supervised import (
    TrainResult,
    accuracy,
    classification_metrics,
    evaluate,
    macro_f1,
    train_model,
)
__all__ = [
    "TrainResult", "accuracy", "classification_metrics", "evaluate", "macro_f1",
    "train_model", "ForwardTrainResult", "train_forwardgnn",
]


def __getattr__(name):
    if name in {"ForwardTrainResult", "train_forwardgnn"}:
        from ..models.forward_gnn import ForwardTrainResult, train_forwardgnn

        return {"ForwardTrainResult": ForwardTrainResult, "train_forwardgnn": train_forwardgnn}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
