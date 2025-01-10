import torch
import pytorch_lightning as pl
import hydra
import torch_optimizer as torch_optim
from models.modules.augmentations import SpecAugment, WhiteNoiseAugment
import matplotlib.pyplot as plt
from torchmetrics.classification import Accuracy, Recall, AUROC, AveragePrecision

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
        self.img_logging_step = 0
        self.save_hyperparameters(hparams)
        self.layerwise_lr_decay = layerwise_lr_decay
        self.model = hydra.utils.instantiate(self.hparams.model)
        self.model_head = hydra.utils.instantiate(self.hparams.model_head)
        self.num_classes = self.hparams.model_head.num_classes
        self.freeze_backbone = freeze_backbone
        self.softmax = torch.nn.Softmax(dim=1)
        self.criterion = hydra.utils.instantiate(self.hparams.criterion)
        self.transform = transform
        self.using_spectrogram = self.model.using_spectrogram
        self.augment_prob = augment_prob
        self.noise_level = noise_level
        

        # White noise augmentation for both waveforms and spectrograms
        self.white_noise_augment = WhiteNoiseAugment(noise_level=self.noise_level, augment_prob=self.augment_prob)
        
        if self.using_spectrogram:
            self.freq_mask_param = freq_mask_param
            self.time_mask_param = time_mask_param
            self.spec_augment = SpecAugment(freq_mask_param=self.freq_mask_param,
                                            time_mask_param=self.time_mask_param,
                                            augment_prob=self.augment_prob)
    
        if not isinstance(self.num_classes, int):
            raise TypeError("Number of classes must be an integer.")
        elif self.num_classes < 2:
            raise ValueError("Number of classes must be at least 2 for a valid classification task.")
        elif self.num_classes == 2:
            self.classification_task = "binary"
        else:
            self.classification_task = "multiclass"

        # PERFORMANCE METRICS
        # 1) Regular classification accuracy
        self.train_acc = Accuracy(task=self.classification_task, num_classes=self.num_classes, average="macro")
        self.val_acc = Accuracy(task=self.classification_task, num_classes=self.num_classes, average="macro")
        self.test_acc= Accuracy(task=self.classification_task, num_classes=self.num_classes, average="macro")

        # 2) Balanced classification accuracy = macro average of recall scores per class
        # Source: https://neptune.ai/blog/balanced-accuracy
        # For Recall, we have to always have task = "multiclass" because of a bug in the PyTorch Lightning source code 
        self.train_balanced_acc = Recall(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_balanced_acc = Recall(task="multiclass", num_classes=self.num_classes, average="macro")
        self.test_balanced_acc = Recall(task="multiclass", num_classes=self.num_classes, average="macro")

        # 3) Area Under ROC
        self.train_auroc = AUROC(task=self.classification_task, num_classes=self.num_classes, average="macro", thresholds=None)
        self.val_auroc = AUROC(task=self.classification_task, num_classes=self.num_classes, average="macro", thresholds=None)
        self.test_auroc = AUROC(task=self.classification_task, num_classes=self.num_classes, average="macro", thresholds=None)

        # 4) Area Under Precision-Recall Curve = Average Precision
        # Source: https://lightning.ai/docs/torchmetrics/stable/classification/average_precision.html
        self.train_aupr = AveragePrecision(task=self.classification_task, num_classes=self.num_classes, average="macro")
        self.val_aupr = AveragePrecision(task=self.classification_task, num_classes=self.num_classes, average="macro")
        self.test_aupr = AveragePrecision(task=self.classification_task, num_classes=self.num_classes, average="macro")


        if self.freeze_backbone:
            print('Freezing encoder params when training from scratch')
            for param in self.model.parameters():
                param.requires_grad = False


    def training_step(self, batch, batch_idx):
        if self.freeze_backbone:
            self.model.eval()
        self.model_head.train()

        X = batch['input']
        y = batch['label']
        
        encoder_output = self.model(X, mask_tokens=False)

        # Compute logits from the classifier head
        y_preds_logits = self.model_head(encoder_output)
        y_preds_labels = torch.argmax(y_preds_logits, dim=1)
        y_preds_probs = self.softmax(y_preds_logits)
        y_preds_probs_positive_class = y_preds_probs[:, 1].squeeze()
        
        loss = self.criterion(y_preds_logits, batch)
        self.train_acc(y_preds_labels, y)
        self.train_balanced_acc(y_preds_labels, y)
        if self.num_classes == 2:
            self.train_auroc(y_preds_probs_positive_class, y)
            self.train_aupr(y_preds_probs_positive_class, y)
        elif self.num_classes > 2:
            self.train_auroc(y_preds_probs, y)
            self.train_aupr(y_preds_probs, y)

        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
        
        
    def validation_step(self, batch, batch_idx):
        X = batch['input']
        y = batch['label']
    
        # Obtain encoder representation
        encoder_output = self.model(X, mask_tokens=False)
        
        # Compute logits from the classifier head
        y_preds_logits = self.model_head(encoder_output)
        y_preds_labels = torch.argmax(y_preds_logits, dim=1)
        y_preds_probs = self.softmax(y_preds_logits)
        y_preds_probs_positive_class = y_preds_probs[:, 1].squeeze()

        # Compute metrics
        loss = self.criterion(y_preds_logits, batch)
        self.val_acc(y_preds_labels, y)
        self.val_balanced_acc(y_preds_labels, y)
        if self.num_classes == 2:
            self.val_auroc(y_preds_probs_positive_class, y)
            self.val_aupr(y_preds_probs_positive_class, y)
        elif self.num_classes > 2:
            self.val_auroc(y_preds_probs, y)
            self.val_aupr(y_preds_probs, y)

        # Log performance metrics in Tensorboard
        self.log('val_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        X = batch['input']
        y = batch['label']
        
        # Obtain encoder representation
        encoder_output = self.model(X, mask_tokens=False)
        
        # Compute logits from the classifier head
        y_preds_logits = self.model_head(encoder_output)
        y_preds_labels = torch.argmax(y_preds_logits, dim=1)
        y_preds_probs = self.softmax(y_preds_logits) # convert logits to probabilities 
        y_preds_probs_positive_class = y_preds_probs[:, 1].squeeze() # take probability of class 1 

        # Compute metrics
        loss = self.criterion(y_preds_logits, batch)
        self.test_acc(y_preds_labels, y)
        self.test_balanced_acc(y_preds_labels, y)
        
        if self.num_classes == 2:
            self.test_auroc(y_preds_probs_positive_class, y)
            self.test_aupr(y_preds_probs_positive_class, y)
        elif self.num_classes > 2:
            self.test_auroc(y_preds_probs, y)
            self.test_aupr(y_preds_probs, y)
            
        # Log performance metrics in Tensorboard
        self.log('test_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.training:
            # Apply white noise augmentation to the input (waveform)
            batch['input'] = self.white_noise_augment(batch['input'])
        
        if self.using_spectrogram:
            # Compute STFT Representation
            batch['input'] = self.transform(batch['input'])
            # Apply SpecAugment
            batch['input'] = self.spec_augment(batch['input'])
        
        return batch

    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('train_balanced_acc', self.train_balanced_acc, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('train_auroc', self.train_auroc, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('train_aupr', self.train_aupr, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
    
    
    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val_balanced_acc', self.val_balanced_acc, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val_auroc', self.val_auroc, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val_aupr', self.val_aupr, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
    
    
    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('test_balanced_acc', self.test_balanced_acc, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('test_auroc', self.test_auroc, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('test_aupr', self.test_aupr, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        """
        Define optimizers and learning-rate schedulers to use in your optimization.

        Returns:
            [optimizer],[scheduler] - The first list contains optimizers, the
            second contains LR schedulers (or lr_dict).
        """
        # Separate parameters for the encoder and the head
        model_params = list(self.model.named_parameters())
        model_head_params = list(self.model_head.named_parameters())

        # Calculate the number of Transformer blocks in the encoder
        num_blocks = self.hparams.model.depth

        # Group parameters with their layer-wise learning rates
        params_to_pass = []

        # Apply layer-wise decay to encoder parameters
        base_lr = self.hparams.optimizer.lr
        decay_factor = self.layerwise_lr_decay

        for name, param in model_params:
            lr = base_lr
            if name.startswith('blocks.'):
                block_nr = int(name.split('.')[1])
                lr *= decay_factor ** (num_blocks - block_nr)
            params_to_pass.append({"params": param, "lr": lr})


        # Add head parameters with the base learning rate
        params_to_pass.extend([{"params": params} for name, params in model_head_params])

        print("\nLearning rates for encoder blocks:")
        for name, param in self.model.named_parameters():
            if name.startswith('blocks.'):
                block_nr = int(name.split('.')[1])
                lr = base_lr * (decay_factor ** (num_blocks - block_nr))
                print(f"Block {block_nr}: {lr}")


        if self.hparams.optimizer.optim == "SGD":
            optimizer = torch.optim.SGD(params_to_pass, lr=self.hparams.optimizer.lr, momentum=self.hparams.optimizer.momentum)
        elif self.hparams.optimizer.optim == 'Adam':
            optimizer = torch.optim.Adam(params_to_pass, lr=self.hparams.optimizer.lr, weight_decay=self.hparams.optimizer.weight_decay)
        elif self.hparams.optimizer.optim == 'AdamW':
            optimizer = torch.optim.AdamW(params_to_pass, lr=self.hparams.optimizer.lr, weight_decay=self.hparams.optimizer.weight_decay, betas=self.hparams.optimizer.betas)
        elif self.hparams.optimizer.optim == 'LAMB':
            optimizer = torch_optim.Lamb(params_to_pass, lr=self.hparams.optimizer.lr)
        else:
            raise NotImplementedError("No valid optimizer name")

        print('OPTIMIZER', optimizer)
        print(f"ESTIMATED TRAINING BATCHES: {self.trainer.num_training_batches}")
        print(f"ESTIMATED GRAD ACCUM: {self.trainer.accumulate_grad_batches}")
        print(f"ESTIMATED STEPPING BATCHES FOR ENTIRE TRAINING: {self.trainer.estimated_stepping_batches}")
        print(f"MAX EPOCHS: {self.trainer.max_epochs}")
        scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer=optimizer, 
                                            total_training_opt_steps=self.trainer.estimated_stepping_batches)
        print('SCHEDULER', scheduler)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step_update(num_updates=self.global_step)

    def load_from_checkpoint(self, checkpoint_path, map_location= None, hparams_file = None, strict= None, **kwargs):
        print('\n\nOverriding load_from_checkpoint method')
    
        ckp = torch.load(checkpoint_path, map_location=map_location)
        state_dict_no_head = get_params_from_checkpoint(ckp, head=False)
        self.model.load_state_dict(state_dict_no_head, strict=False)
              
        if self.freeze_backbone:
            print('Freezing encoder params from loaded checkpoint')
            for param in self.model.parameters():
                param.requires_grad = False      
        return self