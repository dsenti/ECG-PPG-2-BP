import torch
import pytorch_lightning as pl
import hydra
import torch_optimizer as torch_optim
from models.modules.augmentations import SpecAugment, WhiteNoiseAugment
from torchmetrics import MeanSquaredError, MeanAbsoluteError
def get_params_from_checkpoint(checkpoint, head=False):
    model_weights = {}
    for k, v in checkpoint["state_dict"].items(): 
        head_cond = ('_head' not in k) if not head else True
        if k.startswith("model.") and head_cond:
            weight_key = k.replace('model.', '')            
            model_weights[weight_key] = v
    return model_weights

class FinetunePretrainedModel(pl.LightningModule):
    def __init__(self, hparams, transform=None, freeze_backbone=False, 
                 layerwise_lr_decay=0.9, freq_mask_param=0, time_mask_param=0, noise_level=0.15, augment_prob=0.5):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.layerwise_lr_decay = layerwise_lr_decay
        self.model = hydra.utils.instantiate(self.hparams.model)
        self.model_head = hydra.utils.instantiate(self.hparams.model_head)
        self.num_outputs = 2  # Regression outputs
        self.freeze_backbone = freeze_backbone
        self.criterion = torch.nn.MSELoss()  # Using Mean Squared Error for regression
        self.transform = transform
        self.using_spectrogram = self.model.using_spectrogram
        self.augment_prob = augment_prob
        self.noise_level = noise_level

        # Augmentation
        self.white_noise_augment = WhiteNoiseAugment(noise_level=self.noise_level, augment_prob=self.augment_prob)
        if self.using_spectrogram:
            self.freq_mask_param = freq_mask_param
            self.time_mask_param = time_mask_param
            self.spec_augment = SpecAugment(freq_mask_param=self.freq_mask_param,
                                            time_mask_param=self.time_mask_param,
                                            augment_prob=self.augment_prob)

        # Regression Metrics
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        if self.freeze_backbone:
            print('Freezing encoder params when training from scratch')
            for param in self.model.parameters():
                param.requires_grad = False

    def training_step(self, batch, batch_idx):
        if self.freeze_backbone:
            self.model.eval()
        self.model_head.train()

        X = batch['input']
        y = batch['label']  # Shape: (batch_size, 2)

        encoder_output = self.model(X, mask_tokens=False)

        # Compute predictions
        y_preds = self.model_head(encoder_output)

        # Compute loss
        loss = self.criterion(y_preds, y)

        # Update metrics
        self.train_mse(y_preds, y)
        self.train_mae(y_preds, y)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mse', self.train_mse, on_step=False, on_epoch=True, logger=True)
        self.log('train_mae', self.train_mae, on_step=False, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X = batch['input']
        y = batch['label']  # Shape: (batch_size, 2)

        encoder_output = self.model(X, mask_tokens=False)

        # Compute predictions
        y_preds = self.model_head(encoder_output)

        # Compute loss
        loss = self.criterion(y_preds, y)

        # Update metrics
        self.val_mse(y_preds, y)
        self.val_mae(y_preds, y)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mse', self.val_mse, on_step=False, on_epoch=True, logger=True)
        self.log('val_mae', self.val_mae, on_step=False, on_epoch=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        X = batch['input']
        y = batch['label']  # Shape: (batch_size, 2)

        encoder_output = self.model(X, mask_tokens=False)

        # Compute predictions
        y_preds = self.model_head(encoder_output)

        # Compute loss
        loss = self.criterion(y_preds, y)

        # Update metrics
        self.test_mse(y_preds, y)
        self.test_mae(y_preds, y)

        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('test_mse', self.test_mse, on_step=False, on_epoch=True, logger=True)
        self.log('test_mae', self.test_mae, on_step=False, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        """
        Define optimizers and learning-rate schedulers to use in your optimization.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
        #scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer=optimizer)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.layerwise_lr_decay)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def load_from_checkpoint(self, checkpoint_path, map_location= None, hparams_file = None, strict= None, **kwargs):
        print('\n\nOverriding load_from_checkpoint method')
    
        ckp = torch.load(checkpoint_path, map_location=map_location)
        state_dict_no_head = get_params_from_checkpoint(ckp, head=False)
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict_no_head, strict=False)

        if missing_keys:
            print(f"Missing keys (kept in their initialized state): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {unexpected_keys}")

        print("Weights loaded successfully!")
              
        if self.freeze_backbone:
            print('Freezing encoder params from loaded checkpoint')
            for param in self.model.parameters():
                param.requires_grad = False      
        return self