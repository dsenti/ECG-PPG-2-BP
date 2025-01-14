import numpy as np
import torch.nn as nn
import torch
from torchvision import transforms


class STFTPreprocessor(nn.Module):
    def __init__(self, cfg):
        super(STFTPreprocessor, self).__init__()
        self.cfg = cfg
        # Standard Hann window used in STFT computations
        self.register_buffer('window', torch.hann_window(self.cfg.window_length))
        

    def get_stft(self, x, n_fft: int, hop_length: int, window_length: int, window: torch.Tensor, normalizing: bool):
        # x shape: [batch_size, num_channels, timeseries_length]
        batch_size, num_channels, _ = x.shape
        
        # Reshape to combine batch and channel dimensions
        x_reshaped = x.reshape(-1, x.shape[-1])  # Shape: [batch_size * num_channels, timeseries_length]

        # Compute the spectrogram representations using STFT
        Sx = torch.stft(
            input=x_reshaped,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window_length,
            window=window,
            return_complex=True,
            onesided=True)
        
        # Take moduli of the spectrograms
        magnitude = torch.abs(Sx)
        magnitude = torch.log(torch.abs(Sx).pow(2) + 1e-10)
        
        # Reshape to separate batch and channel dimensions
        # Shape: [batch_size, num_channels, freq_bins, time_bins]
        magnitude = magnitude.reshape(batch_size, num_channels, magnitude.shape[-2], magnitude.shape[-1])
        
        # Trim spectrograms to 64 x 64
        magnitude = magnitude[:, :, :64, :64]
        
        # Z-score the spectrograms globally per channel
        if normalizing:
            mean = magnitude.mean(dim=(2, 3), keepdim=True)
            std = magnitude.std(dim=(2, 3), keepdim=True)
            magnitude = (magnitude - mean) / std.clamp(min=1e-10)
        
        return magnitude

    # Override the forward method for the nn.Module
    def forward(self, x):
        Sx = self.get_stft(x, n_fft=self.cfg.n_fft,
                                     hop_length=self.cfg.hop_length,
                                     window_length=self.cfg.window_length,
                                     window=self.window,
                                     normalizing=self.cfg.normalizing)
        return Sx