import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class ECGPPGDataset(Dataset):
    def __init__(self, csv_folder, finetune, minmax, cache_size, transform=None):
        """
        Initialize the dataset by preloading all data into memory.
        Args:
            csv_folder (str): Path to the folder containing CSV files.
            transform (callable, optional): A function/transform to apply to the samples.
        """
        self.transform = transform
        self.inputs = []  # To store input signals
        self.targets = []  # To store target values

        # Preload data from all CSV files
        for file in os.listdir(csv_folder):
            if file.endswith('.csv'):  # Adjust extension if needed
                file_path = os.path.join(csv_folder, file)
                df = pd.read_csv(file_path)

                # Extract and combine ECG and PPG signals
                ecg_signals = df[[f"ECG_F_{i}" for i in range(1, 1251)]].values.astype(np.float32)
                ppg_signals = df[[f"PPG_F_{i}" for i in range(1, 1251)]].values.astype(np.float32)
                combined_signals = np.stack((ecg_signals, ppg_signals), axis=1)  # Shape: (num_samples, 2, 1250)

                # Store inputs and targets
                self.inputs.append(combined_signals)
                self.targets.append(df[["SegDBP_AVG", "SegSBP_AVG"]].values.astype(np.float32))

        # Concatenate all data from multiple files
        self.inputs = np.concatenate(self.inputs, axis=0)  # Shape: (total_samples, 2, 1250)
        self.targets = np.concatenate(self.targets, axis=0)  # Shape: (total_samples, 2)

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieve a sample and its label by index.
        Args:
            idx (int): Index of the sample.
        Returns:
            dict: A dictionary containing the input signals and target values.
        """
        input_signals = self.inputs[idx]
        target = self.targets[idx]

        # Apply transformations, if any
        if self.transform:
            input_signals = self.transform(input_signals)

        # Convert to PyTorch tensors
        input_signals = torch.tensor(input_signals, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return {'input': input_signals, 'label': target}

