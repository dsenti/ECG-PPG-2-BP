import torch
from torch import nn


class BaselineCriterion(nn.Module):
    def __init__(self, name):
        super(BaselineCriterion, self).__init__()
        self.name = name
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, model, batch, return_predicts=False):
        inputs = batch[0]
        print(inputs.shape)
        output = model.forward(inputs)
        labels = torch.FloatTensor(batch[1])

        output = output.squeeze(-1)
        loss = self.loss_fn(output, labels)
        images = {"wav": batch["input"][0],
                  "wav_label": batch["labels"][0]}
        if return_predicts:
            predicts = self.sigmoid(output).squeeze().detach().cpu().numpy()
            logging_output = {"loss": loss.item(),
                              "predicts": predicts,
                              "images": images}
        else:
            logging_output = {"loss": loss.item(),
                              "images": images}
        return loss, logging_output

