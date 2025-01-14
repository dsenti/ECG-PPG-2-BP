import torch
from torch import nn

class ContentAwareCriterion(nn.Module):
    def __init__(self, name, alpha):
        super(ContentAwareCriterion, self).__init__()
        self.name = name
        self.alpha = alpha

    def forward(self, pred, batch):
        # output, pos_enc = model.forward(masked_input, pad_mask)
        X_masked = batch["masked_input"]
        mask = batch["mask_label"].bool()
        X = batch["target"]
        
        masked_only_pred = torch.zeros_like(X)
        masked_only_pred[mask==1] = pred[mask==1]
        
        true_activity = X.masked_select(mask)
        predicted = pred.masked_select(mask)

        # Select masked regions M        
        # given true_activity and predicted, calculate losses
        l1 = torch.mean(torch.abs(true_activity - predicted))  # masked l1 only
    
        non_zero_idxs = torch.abs(true_activity)>1  # anything larget than gamma (in this case 1)
        non_zero = torch.mean(torch.abs(true_activity[non_zero_idxs] - predicted[non_zero_idxs])) # l1 loss on elements with magnitudes larger than gamma
        content_l1 = non_zero
        
        content_aware_loss = self.alpha*non_zero # regularization term  
        loss = l1 + content_aware_loss # final loss        

        images = {
                "ground_truth_channel_0": X[0,:1].detach().cpu(),
                "masked_spectrogram_channel_0": X_masked[0,:1].detach().cpu(),
                "pred_spectrogram_channel_0": pred[0, :1].detach().cpu(),
                "mask_channel_0":mask[0,:1].detach().cpu(),
                "masked_only_input_0":batch["masked_only_input"][0,:1].detach().cpu(),
                "masked_only_pred_0":masked_only_pred[0,:1].detach().cpu()
                }
          
        logging_output = {"loss": loss.item(), 
                        "images": images,
                        "l1_loss": l1.item(),
                        "content_l1": content_l1.item(),
                        "content_aware_loss": content_aware_loss.item()}

        return loss, logging_output