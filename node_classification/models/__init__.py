"""Node-classification GNN architectures."""

from .gcn import GCN, GraphConvolution, ResidualGCN
from .forward_gnn import ForwardGNN, SingleForwardLayer

__all__ = ["GCN", "GraphConvolution", "ResidualGCN", "ForwardGNN", "SingleForwardLayer"]
