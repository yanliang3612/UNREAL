import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True) -> None:
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average

        if isinstance(alpha, (float, int)):
            alpha = torch.tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            alpha = torch.tensor(alpha)
        self.alpha = alpha

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
            inputs = inputs.transpose(1, 2)
            inputs = inputs.contiguous().view(-1, inputs.size(2))
        targets = targets.view(-1, 1)

        log_probabilities = F.log_softmax(inputs, dim=1)
        log_probabilities = log_probabilities.gather(1, targets).view(-1)
        probabilities = log_probabilities.detach().exp()

        if self.alpha is not None:
            self.alpha = self.alpha.type_as(inputs)
            alpha = self.alpha.gather(0, targets.view(-1))
            log_probabilities = log_probabilities * alpha

        loss = -((1 - probabilities) ** self.gamma) * log_probabilities
        return loss.mean() if self.size_average else loss.sum()
