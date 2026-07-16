import torch
import torch.nn.functional as F


def log_softmax(values):
    """Original UNREAL log-softmax implementation."""
    return values - values.exp().sum(-1).log().unsqueeze(-1)


class My_loss:
    """Round-weighted loss retained for compatibility with prior experiments."""

    def __init__(
        self,
        input,
        target,
        round,
        data,
        running_train_mask,
        args,
    ) -> None:
        self.inputs = input
        self.targets = target
        self.round_index = round
        self.data = data
        self.running_train_mask = running_train_mask
        self.stride = args.stride

    def weight(self):
        confidence = torch.max(F.softmax(self.inputs, dim=1), dim=1)[0]
        round_weight = torch.sigmoid(
            torch.tensor(
                self.round_index * self.stride,
                device=self.inputs.device,
            )
        )
        train_confidence = confidence * round_weight
        train_confidence[torch.nonzero(self.data.train_mask)] = 1
        return train_confidence[self.running_train_mask]

    def loss(self):
        weights = self.weight()
        inputs = self.inputs[self.running_train_mask]
        targets = self.targets[self.running_train_mask]
        negative_log_likelihood = -log_softmax(inputs)[
            range(targets.shape[0]),
            targets,
        ]
        return (negative_log_likelihood * weights).mean()


class My_end_loss:
    """Confidence-weighted final loss retained for backward compatibility."""

    def __init__(
        self,
        input,
        target,
        data,
        running_train_mask,
        args,
    ) -> None:
        self.inputs = input
        self.targets = target
        self.data = data
        self.running_train_mask = running_train_mask
        self.stride = args.stride

    def weight(self):
        train_confidence = torch.max(
            F.softmax(self.inputs, dim=1),
            dim=1,
        )[0]
        train_confidence[torch.nonzero(self.data.train_mask)] = 1
        return train_confidence[self.running_train_mask]

    def loss(self):
        weights = self.weight()
        inputs = self.inputs[self.running_train_mask]
        targets = self.targets[self.running_train_mask]
        negative_log_likelihood = -log_softmax(inputs)[
            range(targets.shape[0]),
            targets,
        ]
        return (negative_log_likelihood * weights).mean()
