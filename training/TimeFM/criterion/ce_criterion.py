import torch
from torch import nn


class CrossEntropyWrapper(nn.Module):
    def __init__(self, label_smoothing):
        super(CrossEntropyWrapper, self).__init__()
        self.label_smoothing = label_smoothing
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def forward(self, pred, batch):
        y = batch['label']
        loss = self.loss_fn(pred, y)
        return loss


