from typing import Sequence

from torch import nn
from torch_geometric.nn import (
    BatchNorm,
    ChebConv,
    GATConv,
    GCNConv,
    SAGEConv,
    Sequential,
    SGConv,
)


class GNN(nn.Module):
    """Configurable message-passing encoder used by UNREAL."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        batchnorm_mm: float = 0.99,
        *,
        net: str = "GCN",
        n_heads: int = 8,
        chebyshev_order: int = 2,
    ) -> None:
        super().__init__()
        self.input_size = layer_sizes[0]
        self.representation_size = layer_sizes[-1]
        self.net = net.upper()

        if self.net == "SGC":
            layers = self._build_sgc_layers(layer_sizes, batchnorm_mm)
        else:
            layers = self._build_message_passing_layers(
                layer_sizes,
                n_heads,
                chebyshev_order,
                batchnorm_mm,
            )

        self.model = Sequential("x, edge_index", layers)

    def _build_sgc_layers(
        self,
        layer_sizes: Sequence[int],
        batchnorm_momentum: float,
    ):
        convolution = SGConv(
            self.input_size,
            self.representation_size,
            K=len(layer_sizes) - 1,
            cached=True,
        )
        return [
            (convolution, "x, edge_index -> x"),
            BatchNorm(self.representation_size, momentum=batchnorm_momentum),
            nn.PReLU(),
        ]

    def _build_message_passing_layers(
        self,
        layer_sizes: Sequence[int],
        n_heads: int,
        chebyshev_order: int,
        batchnorm_momentum: float,
    ):
        layers = []
        for input_dim, output_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            convolution = self._make_convolution(
                input_dim,
                output_dim,
                n_heads,
                chebyshev_order,
            )
            layers.extend(
                [
                    (convolution, "x, edge_index -> x"),
                    BatchNorm(output_dim, momentum=batchnorm_momentum),
                    nn.PReLU(),
                ]
            )
        return layers

    def _make_convolution(
        self,
        input_dim: int,
        output_dim: int,
        n_heads: int,
        chebyshev_order: int,
    ):
        if self.net == "GCN":
            return GCNConv(input_dim, output_dim)
        if self.net == "GAT":
            return GATConv(input_dim, output_dim // n_heads, heads=n_heads)
        if self.net == "SAGE":
            return SAGEConv(input_dim, output_dim)
        if self.net == "CHEB":
            return ChebConv(input_dim, output_dim, chebyshev_order)
        raise ValueError(f"Unsupported GNN encoder: {self.net}")

    def forward(self, data):
        return self.model(data.x, data.edge_index)

    def reset_parameters(self) -> None:
        self.model.reset_parameters()
