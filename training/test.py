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
from distutils.util import strtobool

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
        type=lambda x: bool(strtobool(x)),  # Converts "True"/"False" to boolean
        default=False,
        help="Use pretrained model or not"
    )

    parser.add_argument(
        "--freeze_backbone",
        type=lambda x: bool(strtobool(x)),  # Converts "True"/"False" to boolean
        default=False,
        help="Freeze the backbone of the model (default: False)"
    )

    parser.add_argument(
        "--evaluate",
        type=lambda x: bool(strtobool(x)),  # Converts "True"/"False" to boolean
        default=False,
        help="Evaluate the model (default: False)"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        help="Model size (options: [small, base, large] default: small)"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="checkpoint_Vital_small_fitu_wh_LR1e-4.pt",
        help="Model checkpoint, default= checkpoint_Vital_small_fitu_wh_LR1e-4.pt"
    )

    return parser.parse_args()


args = parse_args()

print(f'=========================\nTESTING {args.model_checkpoint}\n=========================\n')

seed = args.seed  # Any fixed number
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
random.seed(seed)
np.random.seed(seed)

# Check if CUDA is available and list all GPUs
if torch.cuda.is_available():
    device = torch.device("cuda")
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs: {', '.join([torch.cuda.get_device_name(i) for i in range(num_gpus)])}")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

from torch.utils.data import random_split, DataLoader

# Create DataLoaders for each split
batch_size = args.batch_size  # Use the same batch size as the original DataLoader

our_dataset = dataset.ECGPPGDataset(csv_folder=f"/capstor/scratch/cscs/dsenti/dataset/{args.dataset}_all", finetune=True, minmax=True, cache_size=1000, dataset=args.dataset)
# our_dataset = dataset.ECGPPGDataset(csv_folder=f"/users/dsenti/dummy_dataset", finetune=True, minmax=True, cache_size=1000, dataset=args.dataset)

# data_loader = DataLoader(our_dataset, batch_size=args.batch_size, shuffle=False)

# Assuming `dataset` is the dataset used in the DataLoader
# dataset = data_loader.dataset  # Extract the dataset from the DataLoader

# Define the split sizes
total_size = len(our_dataset)
train_size = int(0.8 * total_size)  # 80% for training
val_size = int(0.1 * total_size)    # 10% for validation
test_size = total_size - train_size - val_size  # Remaining for testing

# Perform the split
train_dataset, val_dataset, test_dataset = random_split(our_dataset, [train_size, val_size, test_size])

# Create DataLoaders for each split
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print the sizes of each split
print(f"Total Dataset Size: {total_size}")
print(f"Training Set Size: {len(train_dataset)}")
print(f"Validation Set Size: {len(val_dataset)}")
print(f"Test Set Size: {len(test_dataset)}")


# Check the model size argument and load the appropriate model
if args.model_size == "small":
    model = our_model.FinetunePretrainedModelSmall(quantize=False)
elif args.model_size == "base":
    model = our_model.FinetunePretrainedModelBase(quantize=False)
elif args.model_size == "large":
    model = our_model.FinetunePretrainedModelLarge(quantize=False)
else:
    raise ValueError(f"Unknown model size: {args.model_size}")

print(model)
print("Model base classes:", [cls.__name__ for cls in inspect.getmro(type(model))])
model = model.to(torch.float32)
model.eval()

model.to(device)


# Wrap the model with DataParallel for multi-GPU training
if torch.cuda.is_available() and num_gpus > 1:
    model = nn.DataParallel(model)  # Wrap your model to use all GPUs
model.to(device)


# Print the sizes of each split
# print(f"Total Dataset Size: {total_size}")
print(f"Training Set Size: {len(train_dataset)}")
print(f"Validation Set Size: {len(val_dataset)}")
print(f"Test Set Size: {len(test_dataset)}")
print(f"Number of samples in the dataset: {len(train_loader.dataset)}")

# Fetch the first sample directly
first_sample = train_loader.dataset[0]
print("First Sample Input Shape:", first_sample['input'].shape)
print("First Sample Label:", first_sample['label'].shape)

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    model.to(device)
    predictions = []
    true_values = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            true_values.append(labels.cpu().numpy())

    predictions = np.vstack(predictions)
    true_values = np.vstack(true_values)
    mse = mean_squared_error(true_values, predictions)

    print(f"Test MSE: {mse:.4f}")
    return predictions, true_values


# trained_model, train_losses, val_losses = train_model_with_loss_plot(model, train_loader, val_loader, num_epochs=args.num_epochs, patience=args.patience, learning_rate=args.learning_rate, device=device)

model.load_state_dict(torch.load(f'/users/dsenti/code/TimeFM/{args.model_checkpoint}'))
trained_model = model

predictions, true_values = evaluate_model(trained_model, test_loader, device=device)

# plot_losses(train_losses, val_losses)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torchaudio

# Split predictions and true values into DBP and SBP
true_dbp, true_sbp = true_values[:, 0], true_values[:, 1]
pred_dbp, pred_sbp = predictions[:, 0], predictions[:, 1]

# Calculate metrics for DBP
mse_dbp = mean_squared_error(true_dbp, pred_dbp)
mae_dbp = mean_absolute_error(true_dbp, pred_dbp)
r2_dbp = r2_score(true_dbp, pred_dbp)
var_dbp = np.var(true_dbp - pred_dbp)
std_dbp = np.std(true_dbp - pred_dbp)

# Calculate metrics for SBP
mse_sbp = mean_squared_error(true_sbp, pred_sbp)
mae_sbp = mean_absolute_error(true_sbp, pred_sbp)
r2_sbp = r2_score(true_sbp, pred_sbp)
var_sbp = np.var(true_sbp - pred_sbp)
std_sbp = np.std(true_sbp - pred_sbp)

# Print the results
print("Metrics for DBP (Diastolic Blood Pressure):")
print(f"  Test MSE: {mse_dbp:.4f}")
print(f"  Test MAE: {mae_dbp:.4f}")
print(f"  Test R^2: {r2_dbp:.4f}")
print(f"  Error Variance: {var_dbp:.4f}")
print(f"  Error Standard Deviation: {std_dbp:.4f}")

print("\nMetrics for SBP (Systolic Blood Pressure):")
print(f"  Test MSE: {mse_sbp:.4f}")
print(f"  Test MAE: {mae_sbp:.4f}")
print(f"  Test R^2: {r2_sbp:.4f}")
print(f"  Error Variance: {var_sbp:.4f}")
print(f"  Error Standard Deviation: {std_sbp:.4f}")

# Calculate errors for DBP and SBP
error_dbp = np.abs(true_dbp - pred_dbp)
error_sbp = np.abs(true_sbp - pred_sbp)

# Calculate BHS grading for DBP
within_5_dbp = np.mean(error_dbp <= 5) * 100
within_10_dbp = np.mean(error_dbp <= 10) * 100
within_15_dbp = np.mean(error_dbp <= 15) * 100

bhs_grade_dbp = 'D'
if within_5_dbp >= 60 and within_10_dbp >= 85 and within_15_dbp >= 95:
    bhs_grade_dbp = 'A'
elif within_5_dbp >= 50 and within_10_dbp >= 75 and within_15_dbp >= 90:
    bhs_grade_dbp = 'B'
elif within_5_dbp >= 40 and within_10_dbp >= 65 and within_15_dbp >= 85:
    bhs_grade_dbp = 'C'

# Calculate BHS grading for SBP
within_5_sbp = np.mean(error_sbp <= 5) * 100
within_10_sbp = np.mean(error_sbp <= 10) * 100
within_15_sbp = np.mean(error_sbp <= 15) * 100

bhs_grade_sbp = 'D'
if within_5_sbp >= 60 and within_10_sbp >= 85 and within_15_sbp >= 95:
    bhs_grade_sbp = 'A'
elif within_5_sbp >= 50 and within_10_sbp >= 75 and within_15_sbp >= 90:
    bhs_grade_sbp = 'B'
elif within_5_sbp >= 40 and within_10_sbp >= 65 and within_15_sbp >= 85:
    bhs_grade_dbp = 'C'

# Print the results
print("BHS Grades for DBP (Diastolic Blood Pressure):")
print(f"  <=5mmHg: {within_5_dbp:.2f}%, <=10mmHg: {within_10_dbp:.2f}%, <=15mmHg: {within_15_dbp:.2f}% -> Grade {bhs_grade_dbp}")

print("\nBHS Grades for SBP (Systolic Blood Pressure):")
print(f"  <=5mmHg: {within_5_sbp:.2f}%, <=10mmHg: {within_10_sbp:.2f}%, <=15mmHg: {within_15_sbp:.2f}% -> Grade {bhs_grade_sbp}")

# Calculate errors for DBP and SBP
error_dbp = true_dbp - pred_dbp
error_sbp = true_sbp - pred_sbp

# AAMI validation for DBP
mean_error_dbp = np.abs(np.mean(error_dbp))
std_error_dbp = np.std(error_dbp)
aami_valid_dbp = mean_error_dbp < 5 and std_error_dbp < 8

# AAMI validation for SBP
mean_error_sbp = np.abs(np.mean(error_sbp))
std_error_sbp = np.std(error_sbp)
aami_valid_sbp = mean_error_sbp < 5 and std_error_sbp < 8

# Print the results
print("AAMI Validation for DBP (Diastolic Blood Pressure):")
print(f"  Mean Error = {mean_error_dbp:.2f}, STD Error = {std_error_dbp:.2f}, Valid = {aami_valid_dbp}")

print("\nAAMI Validation for SBP (Systolic Blood Pressure):")
print(f"  Mean Error = {mean_error_sbp:.2f}, STD Error = {std_error_sbp:.2f}, Valid = {aami_valid_sbp}")
