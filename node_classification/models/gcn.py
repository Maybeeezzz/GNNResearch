import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """Sparse GCN layer implementing D^-1/2 (A + I) D^-1/2 X W."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes = x.size(0)
        loops = torch.arange(num_nodes, device=x.device)
        source = torch.cat([edge_index[0], loops])
        target = torch.cat([edge_index[1], loops])

        degree = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        degree.index_add_(0, target, torch.ones_like(target, dtype=x.dtype))
        inv_sqrt_degree = degree.clamp_min(1).pow(-0.5)
        norm = inv_sqrt_degree[source] * inv_sqrt_degree[target]

        transformed = self.linear(x)
        output = torch.zeros_like(transformed)
        output.index_add_(0, target, transformed[source] * norm.unsqueeze(1))
        return output


class GCN(nn.Module):
    """Two-layer GCN for node classification."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = GraphConvolution(in_features, hidden_features)
        self.conv2 = GraphConvolution(hidden_features, num_classes)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class ResidualGCN(nn.Module):
    """Deep GCN with residual blocks and LayerNorm.

    ``num_layers`` counts every graph-convolution layer, including the input
    and output projections, and must therefore be at least three.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,
        num_layers: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_layers < 3:
            raise ValueError("ResidualGCN requires at least 3 layers")
        self.input_conv = GraphConvolution(in_features, hidden_features)
        self.hidden_convs = nn.ModuleList(
            GraphConvolution(hidden_features, hidden_features)
            for _ in range(num_layers - 2)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden_features) for _ in range(num_layers - 2)
        )
        self.output_conv = GraphConvolution(hidden_features, num_classes)
        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.input_conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for convolution, normalization in zip(self.hidden_convs, self.norms):
            residual = x
            x = convolution(x, edge_index)
            x = normalization(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual
        return self.output_conv(x, edge_index)
