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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import numpy as np
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tasks.finetune_regression import get_params_from_checkpoint

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
        type=lambda x: bool(strtobool(x)), 
        default=False,
        help="Freeze the backbone of the model (default: False)"
    )

    parser.add_argument(
        "--evaluate",
        type=lambda x: bool(strtobool(x)), 
        default=False,
        help="Evaluate the model (default: False)"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        help="Model size (options: [small, base, large] default: small)"
    )

    return parser.parse_args()


args = parse_args()

print("Parsed arguments:")
print(f"Seed: {args.seed}")
print(f"Number of epochs: {args.num_epochs}")
print(f"Patience: {args.patience}")
print(f"Learning rate: {args.learning_rate}")
print(f"Model: {args.model_size}")
print(f"Wandb name: {args.wandb_name}")
print(f"Batch size: {args.batch_size}")
print(f"Dataset: {args.dataset}")
print(f"Pretrained: {args.pretrained}")
print(f"Freeze backbone: {args.freeze_backbone}")


wandb.init(
    project = "ECG_PPG_BPE",
    name = args.wandb_name,
    config={
    "learning_rate": args.learning_rate,
    "epochs": args.num_epochs,
    "patience": args.patience,
    "seed": args.seed,
    "model": args.model_size,
    "batch_size": args.batch_size,
    "dataset": args.dataset,
    "pretrained": args.pretrained,
    "freeze_backbone": args.freeze_backbone
    }
)

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


batch_size = args.batch_size 

our_dataset = dataset.ECGPPGDataset(csv_folder=f"/capstor/scratch/cscs/<username>/dataset/{args.dataset}_all", finetune=True, minmax=True, cache_size=1000, dataset=args.dataset)

# split sizes
total_size = len(our_dataset)
train_size = int(0.8 * total_size)  # 80% for training
val_size = int(0.1 * total_size)    # 10% for validation
test_size = total_size - train_size - val_size  # Remaining for testing

train_dataset, val_dataset, test_dataset = random_split(our_dataset, [train_size, val_size, test_size])

# DataLoaders for each split
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# sizes of each split
print(f"Total Dataset Size: {total_size}")
print(f"Training Set Size: {len(train_dataset)}")
print(f"Validation Set Size: {len(val_dataset)}")
print(f"Test Set Size: {len(test_dataset)}")

# load the appropriate model
if args.model_size == "small":
    model = our_model.FinetunePretrainedModelSmall(quantize=False)
    checkpoint_path = "/users/<username>/pretrained_models/cerebro_alternating_small.ckpt"
elif args.model_size == "base":
    model = our_model.FinetunePretrainedModelBase(quantize=False)
    checkpoint_path = "/users/<username>/pretrained_models/cerebro_alternating_base_updated.ckpt"
elif args.model_size == "large":
    model = our_model.FinetunePretrainedModelLarge(quantize=False)
    checkpoint_path = "/users/<username>/pretrained_models/cerebro_alternating_large.ckpt"
else:
    raise ValueError(f"Unknown model size: {args.model_size}")

print(model)
print("Model base classes:", [cls.__name__ for cls in inspect.getmro(type(model))])
model = model.to(torch.float32)
model.eval()

model.to(device)

if args.pretrained:
    # checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(checkpoint.keys())

    state_dict = checkpoint['state_dict']
    print("Layers and their sizes in the state_dict:")
    for name, param in state_dict.items():
        print(f"{name}: {param.size()}")

    state_dict_no_head = get_params_from_checkpoint(checkpoint, head=False)

    updated_state_dict = {}
    for key, value in state_dict.items():
        # Map model_head.decoder_pred.* to model_head1.mlp.0.fc.*
        if key.startswith("model_head.decoder_pred.weight"):
            if value.shape == model.state_dict()["model_head.mlp.0.fc.weight"].shape:
                updated_state_dict["model_head.mlp.0.fc.weight"] = value
            else:
                print(f"Skipping key {key} due to size mismatch.")
        elif key.startswith("model_head.decoder_pred.bias"):
            if value.shape == model.state_dict()["model_head.mlp.0.fc.bias"].shape:
                updated_state_dict["model_head.mlp.0.fc.bias"] = value
            else:
                print(f"Skipping key {key} due to size mismatch.")
        # Retain other keys as they are
        else:
            updated_state_dict[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(updated_state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys (kept in their initialized state): {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys (ignored): {unexpected_keys}")

    print("Weights loaded successfully!")

# Wrap the model with DataParallel for multi-GPU training
if torch.cuda.is_available() and num_gpus > 1:
    model = nn.DataParallel(model)
model.to(device)


# Print the sizes of each split
# print(f"Total Dataset Size: {total_size}")
print(f"Training Set Size: {len(train_dataset)}")
print(f"Validation Set Size: {len(val_dataset)}")
print(f"Test Set Size: {len(test_dataset)}")
print(f"Number of samples in the dataset: {len(train_loader.dataset)}")

# Fetch first sample
first_sample = train_loader.dataset[0]
print("First Sample Input Shape:", first_sample['input'].shape)
print("First Sample Label:", first_sample['label'].shape)

def freeze_attention_blocks(model):
    # local variable to access unwrapped model
    unwrapped_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    for param in unwrapped_model.parameters():
        param.requires_grad = False
    for param in unwrapped_model.model.patch_embed.parameters():
        param.requires_grad = True
    for param in unwrapped_model.model_head.parameters():
        param.requires_grad = True


if args.freeze_backbone == True:
    freeze_attention_blocks(model)

# Define EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...")
        self.best_loss = val_loss
        torch.save(model.state_dict(), f'checkpoint_{args.wandb_name}.pt')

# Training function with loss tracking
def train_model_with_loss_plot(model, train_loader, val_loader, num_epochs=50, patience=5, learning_rate=1e-3, device=device):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch['input'].to(device), batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Train Batch {batch_idx+1}/{len(train_loader)}: Loss = {loss.item():.4f}")

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                inputs, labels = batch['input'].to(device), batch['label'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == len(val_loader):
                    print(f"  Val Batch {batch_idx+1}/{len(val_loader)}: Loss = {loss.item():.4f}")

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load the best model
    model.load_state_dict(torch.load(f'checkpoint_{args.wandb_name}.pt'))

    # Return the model and the loss values
    return model, train_losses, val_losses

# Plotting function
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_plot_{args.wandb_name}.png')

# Evaluate function
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


trained_model, train_losses, val_losses = train_model_with_loss_plot(model, train_loader, val_loader, num_epochs=args.num_epochs, patience=args.patience, learning_rate=args.learning_rate, device=device)

model.load_state_dict(torch.load(f'checkpoint_{args.wandb_name}.pt'))
trained_model = model

predictions, true_values = evaluate_model(trained_model, test_loader, device=device)

plot_losses(train_losses, val_losses)



# Split predictions and true values into DBP and SBP
true_dbp, true_sbp = true_values[:, 0], true_values[:, 1]
pred_dbp, pred_sbp = predictions[:, 0], predictions[:, 1]

# DBP
mse_dbp = mean_squared_error(true_dbp, pred_dbp)
mae_dbp = mean_absolute_error(true_dbp, pred_dbp)
r2_dbp = r2_score(true_dbp, pred_dbp)
var_dbp = np.var(true_dbp - pred_dbp)
std_dbp = np.std(true_dbp - pred_dbp)

# SBP
mse_sbp = mean_squared_error(true_sbp, pred_sbp)
mae_sbp = mean_absolute_error(true_sbp, pred_sbp)
r2_sbp = r2_score(true_sbp, pred_sbp)
var_sbp = np.var(true_sbp - pred_sbp)
std_sbp = np.std(true_sbp - pred_sbp)

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

# errors for DBP and SBP
error_dbp = np.abs(true_dbp - pred_dbp)
error_sbp = np.abs(true_sbp - pred_sbp)

# BHS grading for DBP
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

# BHS grading for SBP
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

print("BHS Grades for DBP (Diastolic Blood Pressure):")
print(f"  <=5mmHg: {within_5_dbp:.2f}%, <=10mmHg: {within_10_dbp:.2f}%, <=15mmHg: {within_15_dbp:.2f}% -> Grade {bhs_grade_dbp}")

print("\nBHS Grades for SBP (Systolic Blood Pressure):")
print(f"  <=5mmHg: {within_5_sbp:.2f}%, <=10mmHg: {within_10_sbp:.2f}%, <=15mmHg: {within_15_sbp:.2f}% -> Grade {bhs_grade_sbp}")

# errors for DBP and SBP
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

print("AAMI Validation for DBP (Diastolic Blood Pressure):")
print(f"  Mean Error = {mean_error_dbp:.2f}, STD Error = {std_error_dbp:.2f}, Valid = {aami_valid_dbp}")

print("\nAAMI Validation for SBP (Systolic Blood Pressure):")
print(f"  Mean Error = {mean_error_sbp:.2f}, STD Error = {std_error_sbp:.2f}, Valid = {aami_valid_sbp}")

path = f"final_model_{args.wandb_name}.ckpt"

# checkpoint dictionary
checkpoint = {
    'model_state_dict': trained_model.state_dict()
}

# Save model
torch.save(checkpoint, path)
print(f"Model saved to {path}")
wandb.finish()