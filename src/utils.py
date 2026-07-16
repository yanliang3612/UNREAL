import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score


def set_random_seeds(seed: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def random_seed(repetition: int, dataset: Optional[str] = None) -> int:
    """Return the repetition seed used by the original experiments."""
    if dataset is None:
        # Retain compatibility with the original one-argument helper without
        # parsing command-line arguments at module import time.
        from src.args import parse_args

        dataset = parse_args().dataset

    if dataset in {"Cora", "CiteSeer"}:
        return repetition * 10 + 1
    if dataset in {"PubMed", "Computers"}:
        return repetition
    raise ValueError(f"Unsupported dataset: {dataset}")


def reset(module) -> None:
    """Recursively reset a module and its children."""
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
        return
    if hasattr(module, "children"):
        for child in module.children():
            reset(child)


def create_dirs(directories: Iterable[str]) -> None:
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def config_to_string(args) -> str:
    """Serialize an argparse namespace for concise experiment logging."""
    return "_".join(
        f"{name}_{value}" for name, value in vars(args).items() if value is not False
    )


# Backward-compatible name used by earlier versions of the repository.
config2string = config_to_string


def _split_metrics(predictions, labels, mask):
    split_predictions = predictions[mask]
    split_labels = labels[mask]

    predictions_cpu = split_predictions.detach().cpu().numpy()
    labels_cpu = split_labels.detach().cpu().numpy()

    accuracy = (split_predictions == split_labels).float().mean().item() * 100
    balanced_accuracy = balanced_accuracy_score(labels_cpu, predictions_cpu) * 100
    macro_f1 = f1_score(labels_cpu, predictions_cpu, average="macro") * 100
    return accuracy, balanced_accuracy, macro_f1


def compute_accuracy(predictions, labels, train_mask, val_mask, test_mask):
    """Compute accuracy, balanced accuracy, and macro-F1 for each split."""
    train_metrics = _split_metrics(predictions, labels, train_mask)
    validation_metrics = _split_metrics(predictions, labels, val_mask)
    test_metrics = _split_metrics(predictions, labels, test_mask)

    train_accuracy, train_balanced_accuracy, train_f1 = train_metrics
    validation_accuracy, validation_balanced_accuracy, validation_f1 = (
        validation_metrics
    )
    test_accuracy, test_balanced_accuracy, test_f1 = test_metrics

    return (
        train_accuracy,
        validation_accuracy,
        test_accuracy,
        train_balanced_accuracy,
        validation_balanced_accuracy,
        test_balanced_accuracy,
        train_f1,
        validation_f1,
        test_f1,
    )


def compute_representation(network, data, device):
    network.eval()
    with torch.no_grad():
        return network(data.to(device))
