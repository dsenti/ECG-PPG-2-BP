import torch
import torch.nn as nn
from torchaudio.transforms import TimeMasking, FrequencyMasking

class SpecAugment(nn.Module):
    """
    Applies SpecAugment, a data augmentation method for spectrograms.
    
    This class implements frequency and time masking as described in the SpecAugment paper:
    https://arxiv.org/abs/1904.08779
    
    Attributes:
        freq_mask (FrequencyMasking): Frequency masking transform.
        time_mask (TimeMasking): Time masking transform.
        augment_prob (float): Probability of applying each augmentation.
    """

    def __init__(self, freq_mask_param=5, time_mask_param=5, augment_prob=0.5):
        """
        Initializes the SpecAugment class.

        Args:
            freq_mask_param (int): Maximum frequency mask length.
            time_mask_param (int): Maximum time mask length.
            augment_prob (float): Probability of applying each augmentation.
        """
        super(SpecAugment, self).__init__()
        self.freq_mask = FrequencyMasking(freq_mask_param, iid_masks=True)
        self.time_mask = TimeMasking(time_mask_param, iid_masks=True)
        self.augment_prob = augment_prob

    def forward(self, x):
        """
        Applies SpecAugment to the input tensor.

        Args:
            x (torch.Tensor): Input spectrogram tensor.

        Returns:
            torch.Tensor: Augmented spectrogram tensor.
        """
        if torch.rand(1).item() < self.augment_prob:
            x = self.freq_mask(x)
        if torch.rand(1).item() < self.augment_prob:
            x = self.time_mask(x)
        return x


class WaveformAugment(nn.Module):
    """
    Applies waveform augmentation by masking random portions of the input signal.

    This class implements a vectorized version of waveform masking, where consecutive parts of the signal 
    are randomly set to zero.

    Attributes:
        signal_mask_param (int): Maximum length of the masking window.
        augment_prob (float): Probability of applying augmentation to each channel.
    """

    def __init__(self, signal_mask_param=200, augment_prob=0.5):
        """
        Initializes the WaveformAugment class.

        Args:
            signal_mask_param (int): Maximum length of the masking window.
            augment_prob (float): Probability of applying augmentation to each channel.
        """
        super(WaveformAugment, self).__init__()
        self.signal_mask_param = signal_mask_param
        self.augment_prob = augment_prob

    def forward(self, x):
        """
        Applies waveform augmentation to the input tensor.

        Args:
            x (torch.Tensor): Input waveform tensor of shape [batch_size, num_channels, signal_length].

        Returns:
            torch.Tensor: Augmented waveform tensor.
        """
        batch_size, num_channels, signal_length = x.shape

        # Create a mask for whether to apply augmentation to each channel
        augment_mask = torch.rand(batch_size, num_channels, 1) < self.augment_prob

        # Generate random mask lengths
        mask_lengths = torch.randint(1, self.signal_mask_param + 1, (batch_size, num_channels, 1))

        # Generate random start positions
        start_positions = torch.randint(0, signal_length + 1 - self.signal_mask_param, (batch_size, num_channels, 1))

        # Create a range tensor
        range_tensor = torch.arange(signal_length).expand(batch_size, num_channels, signal_length)

        # Create the mask
        mask = (range_tensor < start_positions + mask_lengths) & (range_tensor >= start_positions)

        # Invert and cast the mask to float
        mask = (~mask).float()

        # Apply the augmentation mask
        mask = torch.where(augment_mask, mask, torch.ones_like(mask))

        # Apply the mask to the input tensor
        return x * mask

class WhiteNoiseAugment(nn.Module):
    """
    Applies white noise augmentation to the input time series signal.
    This class injects white noise into the signal, with a specified noise level.
    Each channel is independently affected.

    Attributes:
        noise_level (float): Controls the intensity of the white noise.
        augment_prob (float): Probability of applying augmentation to each channel.
    """

    def __init__(self, noise_level=0.1, augment_prob=0.5):
        """
        Initializes the WhiteNoiseAugment class.

        Args:
            noise_level (float): Controls the intensity of the white noise.
            augment_prob (float): Probability of applying augmentation to each channel.
        """
        super(WhiteNoiseAugment, self).__init__()
        self.noise_level = noise_level
        self.augment_prob = augment_prob

    def forward(self, x):
        """
        Applies white noise augmentation to the input tensor.

        Args:
            x (torch.Tensor): Input time series tensor of shape [batch_size, num_channels, signal_length].

        Returns:
            torch.Tensor: Augmented time series tensor of shape [batch_size, num_channels, signal_length].
        """
        batch_size, num_channels, signal_length = x.shape

        # Generate white noise
        white_noise = torch.randn_like(x) * self.noise_level

        # Create a mask for whether to apply augmentation to each channel
        augment_mask = (torch.rand(batch_size, num_channels, 1) < self.augment_prob).to(x.device)

        # Apply the augmentation
        augmented = x + white_noise * augment_mask

        return augmented