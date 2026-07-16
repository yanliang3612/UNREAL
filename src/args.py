import argparse
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UNREAL for semi-supervised imbalanced node classification."
    )

    # Dataset and evaluation.
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        help="Dataset name: Cora, CiteSeer, or PubMed.",
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument(
        "--imb_ratio",
        type=float,
        default=10,
        help="Class imbalance ratio.",
    )

    # Encoder.
    parser.add_argument("--dim", type=int, default=128, help="Hidden dimension.")
    parser.add_argument("--layers", type=int, default=2, help="Number of GNN layers.")
    parser.add_argument(
        "--n_head",
        type=int,
        default=8,
        help="Number of attention heads for GAT.",
    )
    parser.add_argument(
        "--net",
        type=str,
        default="GCN",
        help="Encoder type: GCN, GAT, SAGE, SGC, or CHEB.",
    )
    parser.add_argument(
        "--chebgcn_para",
        type=int,
        default=2,
        help="Chebyshev filter size for ChebConv.",
    )

    # Optimization.
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=2000,
        help="Number of final training epochs.",
    )
    parser.add_argument(
        "--lr",
        "-lr",
        type=float,
        default=0.005,
        help="Learning rate.",
    )
    parser.add_argument("--decay", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--patience", type=int, default=300)

    # UNREAL.
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--clustering", action="store_true", default=True)
    parser.add_argument("--num_K", type=int, default=200)
    parser.add_argument(
        "--kmeans_backend",
        type=str,
        choices=("cpu", "gpu"),
        default="cpu",
        help="K-means backend: scikit-learn on CPU (default) or torch-kmeans on GPU.",
    )
    parser.add_argument(
        "--stride",
        "-s",
        type=float,
        default=1.0,
        help="Stride used by the optional round-weighted loss.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Geometric ambiguity filtering threshold.",
    )
    parser.add_argument("--rbo", type=float, default=0.5, help="RBO weight.")
    parser.add_argument(
        "--ad",
        type=float,
        default=4,
        help="Number of nodes added per class and round.",
    )

    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse known arguments while allowing launcher-specific extra flags."""
    return build_parser().parse_known_args(argv)[0]
