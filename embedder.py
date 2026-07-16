import math
import statistics
from pathlib import Path

import torch

from src.utils import compute_accuracy, config_to_string


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

    device_index = device.index if device.index is not None else 0
    if device_index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {device_index} is unavailable; "
            f"found {torch.cuda.device_count()} GPU(s)."
        )
    return device


class Embedder:
    """Shared experiment bookkeeping and evaluation utilities."""

    def __init__(self, args) -> None:
        self.args = args
        self.device = _resolve_device(getattr(args, "device", "auto"))
        self.path = Path(__file__).resolve().parent / "data" / args.dataset
        self.hidden_layers = [args.dim] * args.layers

        print(f"\n[Config] {config_to_string(args)}\n")
        print(f"[Device] {self.device}\n")

        self.best_val = 0.0
        self.cnt = 0
        self.epoch_list = []

        self.train_accs = []
        self.valid_accs = []
        self.test_accs = []
        self.train_baccs = []
        self.valid_baccs = []
        self.test_baccs = []
        self.train_f1 = []
        self.valid_f1 = []
        self.test_f1 = []

        self._reset_running_metrics()

    def _reset_running_metrics(self) -> None:
        self.running_train_accs = []
        self.running_valid_accs = []
        self.running_test_accs = []
        self.running_train_baccs = []
        self.running_valid_baccs = []
        self.running_test_baccs = []
        self.running_train_f1 = []
        self.running_valid_f1 = []
        self.running_test_f1 = []

    def evaluate(self, batch_data, status: str) -> None:
        self.model.eval()
        with torch.no_grad():
            _, predictions = self.model.cls(batch_data)

        metrics = compute_accuracy(
            predictions,
            batch_data.y,
            self.train_mask,
            self.val_mask,
            self.test_mask,
        )
        (
            train_acc,
            val_acc,
            test_acc,
            train_bacc,
            val_bacc,
            test_bacc,
            train_f1,
            val_f1,
            test_f1,
        ) = metrics

        self.running_train_accs.append(train_acc)
        self.running_valid_accs.append(val_acc)
        self.running_test_accs.append(test_acc)
        self.running_train_baccs.append(train_bacc)
        self.running_valid_baccs.append(val_bacc)
        self.running_test_baccs.append(test_bacc)
        self.running_train_f1.append(train_f1)
        self.running_valid_f1.append(val_f1)
        self.running_test_f1.append(test_f1)

        if val_acc > self.best_val:
            self.best_val = val_acc
            self.cnt = 0
        else:
            self.cnt += 1

        print(
            f"{status}"
            f"| train_acc: {train_acc:.2f} "
            f"| valid_acc: {val_acc:.2f} "
            f"| test_acc: {test_acc:.2f} "
            f"| train_bacc: {train_bacc:.2f} "
            f"| valid_bacc: {val_bacc:.2f} "
            f"| test_bacc: {test_bacc:.2f} "
            f"| train_f1: {train_f1:.2f} "
            f"| valid_f1: {val_f1:.2f} "
            f"| test_f1: {test_f1:.2f} |"
        )

    def save_results(self, repetition: int) -> None:
        validation_accuracies = torch.tensor(self.running_valid_accs)
        selected_epoch = int(validation_accuracies.argmax().item())

        self.epoch_list.append(selected_epoch)
        self.train_accs.append(self.running_train_accs[selected_epoch])
        self.valid_accs.append(self.running_valid_accs[selected_epoch])
        self.test_accs.append(self.running_test_accs[selected_epoch])
        self.train_baccs.append(self.running_train_baccs[selected_epoch])
        self.valid_baccs.append(self.running_valid_baccs[selected_epoch])
        self.test_baccs.append(self.running_test_baccs[selected_epoch])
        self.train_f1.append(self.running_train_f1[selected_epoch])
        self.valid_f1.append(self.running_valid_f1[selected_epoch])
        self.test_f1.append(self.running_test_f1[selected_epoch])

        if repetition + 1 != self.args.repetitions:
            self._reset_running_metrics()
            self.cnt = 0
            self.best_val = 0.0

    @staticmethod
    def _standard_error(values, repetitions: int) -> float:
        if len(values) == 1:
            return 0.0
        return statistics.stdev(values) / math.sqrt(repetitions)

    def summary(self) -> None:
        train_acc_mean = statistics.mean(self.train_accs)
        val_acc_mean = statistics.mean(self.valid_accs)
        test_acc_mean = statistics.mean(self.test_accs)
        val_f1_mean = statistics.mean(self.valid_f1)
        test_f1_mean = statistics.mean(self.test_f1)
        test_bacc_mean = statistics.mean(self.test_baccs)

        acc_ci = self._standard_error(self.test_accs, self.args.repetitions)
        bacc_ci = self._standard_error(self.test_baccs, self.args.repetitions)
        f1_ci = self._standard_error(self.test_f1, self.args.repetitions)

        print(
            "** "
            f"| test acc: {test_acc_mean:.2f} +- {acc_ci:.2f} "
            f"| test bacc: {test_bacc_mean:.2f} +- {bacc_ci:.2f} "
            f"| test f1: {test_f1_mean:.2f} +- {f1_ci:.2f} "
            f"| val acc: {val_acc_mean:.2f} "
            f"| val f1: {val_f1_mean:.2f} "
            f"| train acc: {train_acc_mean:.2f} "
            "| **\n"
        )


# Backward-compatible class name used by earlier versions.
embedder = Embedder
