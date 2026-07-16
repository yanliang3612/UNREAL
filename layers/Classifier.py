import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, hidden_size: int, num_class: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_class, bias=True)
        self.reset_parameters()

    def forward(self, embeddings):
        logits = self.linear(embeddings)
        predictions = torch.argmax(logits, dim=1)
        return logits, predictions

    def reset_parameters(self) -> None:
        self.linear.reset_parameters()
