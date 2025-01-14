from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# unchanged DataModule from TimeFM pretraining
class EEGDataModule(pl.LightningDataModule):

    def __init__(self, train, val, test=None, cfg=None, name="", **kwargs):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.name = name
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = self.train
            self.val_dataset = self.val

        # Assign test dataset for use in dataloader(s)
        elif stage == "validate":
            self.val_dataset = self.val
        elif stage == "test":
            self.test_dataset = self.test

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            pin_memory=True,
        )