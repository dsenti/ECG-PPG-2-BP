import torch
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import os
from models import simMiM
import inspect
import our_model as our_model
import datasets.ecg_ppg_dataset as dataset
from torch.utils.data import DataLoader
import our_model as our_model
import random
import numpy as np
import timm
import wandb
import argparse

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a machine learning model.")

    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=50, 
        help="Number of training epochs (default: 50)"
    )

    parser.add_argument(
        "--patience", 
        type=int, 
        default=64, 
        help="Early stopping patience (default: 64)"
    )

    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-4, 
        help="Learning rate for optimization (default: 1e-4)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="no_model_specified",
        help="Model to use (default: no_model_specified)"
    )

    parser.add_argument(
        "--wandb_name",
        type=str,
        default="no_name_specified",
        help="Name for the wandb run (default: no_name_specified)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="Vital",
        help="Dataset to use Vital or MIMIC(default: Vital)"
    )

    parser.add_argument(
        "--pretrained",
        type=bool,
        default=False,
        help="Use pretrained model or not"
    )

    return parser.parse_args()


args = parse_args()

print("Parsed arguments:")
print(f"Seed: {args.seed}")
print(f"Number of epochs: {args.num_epochs}")
print(f"Patience: {args.patience}")
print(f"Learning rate: {args.learning_rate}")
print(f"Model: {args.model}")
print(f"Wandb name: {args.wandb_name}")
print(f"Batch size: {args.batch_size}")
print(f"Dataset: {args.dataset}")

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # multi-GPU setups
random.seed(seed)
np.random.seed(seed)

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# our_dataset = dataset.ECGPPGDataset(csv_folder="/users/<username>/dummy_dataset", finetune=True, minmax=True, cache_size=1000)
train_dataset = dataset.ECGPPGDataset(csv_folder=f"/capstor/scratch/cscs/<username>/dataset/{args.dataset}", finetune=True, minmax=True, cache_size=1000)
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

# for batch in data_loader:
#     print("Input Signal Shape:", batch['input'].shape)
#     print("Input Signal:", batch['input'])
#     print("Label Shape:", batch['label'].shape)
#     print("Label:", batch['label'])
#     break

model = our_model.FinetunePretrainedModel(quantize=False)
print(model)
print("Model base classes:", [cls.__name__ for cls in inspect.getmro(type(model))])
model = model.to(torch.float32)
model.eval()

model.to(device)
from torch.utils.data import random_split, DataLoader


dl_dataset = train_data_loader.dataset  # Extract dataset from dl

# Define the split sizes
total_size = len(dl_dataset)
train_size = int(0.85 * total_size)  # 85% for training
val_size = total_size - train_size    # 15% for validation

# Perform the split
train_dataset, val_dataset = random_split(dl_dataset, [train_size, val_size])

# Create DataLoaders for each split
batch_size = train_data_loader.batch_size  # Use the same batch size as the original DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = dataset.ECGPPGDataset(csv_folder=f"/capstor/scratch/cscs/<username>/dataset/{args.dataset}_test", finetune=True, minmax=True, cache_size=1000)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print the sizes of each split
print(f"Total Dataset Size: {total_size}")
print(f"Training Set Size: {len(train_dataset)}")
print(f"Validation Set Size: {len(val_dataset)}")
print(f"Test Set Size: {len(test_dataset)}")
print(f"Number of samples in the dataset: {len(train_loader.dataset)}")

# Fetch first
first_sample = train_loader.dataset[0]
print("First Sample Input Shape:", first_sample['input'].shape)
print("First Sample Label:", first_sample['label'].shape)

#check dataset
def check_for_nans(data_loader):
    for batch in data_loader:
        inputs, labels = batch['input'], batch['label']
        if torch.isnan(inputs).any() or torch.isnan(labels).any():
            return True
    return False

train_has_nans = check_for_nans(train_loader)
val_has_nans = check_for_nans(val_loader)
test_has_nans = check_for_nans(test_loader)

print(f"Training set contains NaNs: {train_has_nans}")
print(f"Validation set contains NaNs: {val_has_nans}")
print(f"Test set contains NaNs: {test_has_nans}")