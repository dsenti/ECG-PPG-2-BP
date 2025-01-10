import torch
from torch import nn
from models.modules.patching import unpatchify
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


class ReconstructionNorm(nn.Module):
    def __init__(self, patch_size, keep_chans, using_spectrogram, square_patches, alpha=0.1, loss_type="smooth_l1"):
        super(ReconstructionNorm, self).__init__()
        self.patch_size = patch_size
        self.keep_chans = keep_chans
        self.using_spectrogram = using_spectrogram
        self.alpha = alpha
        self.loss_type = loss_type
        self.square_patches = square_patches

    def loss_fn(self, pred, target, token_mask, alpha=0.1):
        #print(f"pred shape: {pred.shape}, target shape: {target.shape}, token mask shape: {token_mask.shape}")
        if self.loss_type == 'l1':
            loss = F.l1_loss(pred, target, reduction='none')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(pred, target, reduction='none')
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pred, target, reduction='none')
        else:
            raise ValueError("Invalid loss_type. Choose 'l1', 'l2', or 'smooth_l1'.")
            
        loss = loss.mean(dim=-1)  # [batch_size, num_patches], mean loss per patch

        # Loss for masked patches
        # In the token binary mask: 0 is visible token, 1 is masked token
        masked_loss = (loss * token_mask).sum() / token_mask.sum()

        if alpha == 0:
            return masked_loss
        else:
            # Loss for visible patches
            visible_loss = (loss * (1 - token_mask)).sum() / (1 - token_mask).sum()
            # Combined loss
            total_loss = masked_loss + (alpha * visible_loss)
            return total_loss
        
    def forward(self, pred, batch):
        # Extract necessary elements from the batch
        X = batch["input"]
        target = batch["target"]
        token_mask = batch["token_mask"] 
        # Compute loss

        loss = self.loss_fn(pred, target, token_mask, self.alpha)
        logging_output = {}
    
        # logging
        if self.using_spectrogram:
            B, C, H, W = X.shape
            # for logging
            pred_unpatchified = unpatchify(pred, patch_size=self.patch_size, height=H, width=W, num_channels=C, keep_chans=self.keep_chans, using_spectrogram=self.using_spectrogram, square_patches=self.square_patches)
            # pred_unpatchified = torch.einsum('nchw->nhwc', pred_unpatchified)
            target_unpatchified = unpatchify(target, patch_size=self.patch_size, height=H, width=W, num_channels=C, keep_chans=self.keep_chans, using_spectrogram=self.using_spectrogram, square_patches=self.square_patches)
            # target_unpatchified = torch.einsum('nchw->nhwc', target_unpatchified)
            if self.keep_chans and self.square_patches:
                token_mask = token_mask.unsqueeze(-1).repeat(1, 1, self.patch_size**2)
            elif self.keep_chans:
                token_mask = token_mask.unsqueeze(-1).repeat(1, 1, self.patch_size*H)
            else:                
                token_mask = token_mask.unsqueeze(-1).repeat(1, 1, self.patch_size**2*C)
            token_mask = unpatchify(token_mask, patch_size=self.patch_size, height=H, width=W, num_channels=C, keep_chans=self.keep_chans, using_spectrogram=self.using_spectrogram, square_patches=self.square_patches)  # 1 is removing, 0 is keeping
            masked_image = X * (1 - token_mask)
            recon_with_visible_patches = X * (1 - token_mask) + pred_unpatchified * token_mask
            # For visualization purposes
            token_mask = token_mask.bool()
            # Logging images and other relevant data
            images = {
                "target_channel_0": target_unpatchified[0, :1, :, :].detach().cpu(),
                "pred_channel_0": pred_unpatchified[0, :1, :, :].detach().cpu(),  # Add channel dimension: [1, 64, 64]
                "mask_channel_0": ~token_mask[0, :1, :, :].detach().cpu(),  # Add channel dimension: [1, 64, 64]
                "masked_image_channel_0": masked_image[0, :1, :, :].cpu(),  # Add channel dimension: [1, 64, 64]
                "reconstruction_with_visible_channel_0": recon_with_visible_patches[0, :1, :, :].detach().cpu(),  # Add channel dimension: [1, 64, 64]
                "target_channel_5": target_unpatchified[0, 4:5, :, :].detach().cpu(),
                "pred_channel_5": pred_unpatchified[0, 4:5, :, :].detach().cpu(),  # Add channel dimension: [1, 64, 64]
                "mask_channel_5": ~token_mask[0, 4:5, :, :].detach().cpu(),  # Add channel dimension: [1, 64, 64]
                "masked_image_channel_5": masked_image[0, 4:5, :, :].cpu(),  # Add channel dimension: [1, 64, 64]
                "reconstruction_with_visible_channel_5": recon_with_visible_patches[0, 4:5, :, :].detach().cpu(),
            }
            
            logging_output = {"l2_loss": loss.item(), "images": images}
        
        else:
            logging_output = {"l2_loss": loss.item()}#, "images": images}

        return loss, logging_output
