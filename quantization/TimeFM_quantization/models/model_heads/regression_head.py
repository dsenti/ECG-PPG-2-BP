import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    def __init__(
        self,
        output_dim: int = 2,  # Number of regression outputs
        embed_dim: int = 768,  # Input embedding dimension
    ):
        super().__init__()
        self.output_dim = output_dim
        self.embed_dim = embed_dim

        # Projection from embed_dim to output_dim
        self.regressor = nn.Linear(self.embed_dim, self.output_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, num_tokens + 1, embed_dim] - patch tokens + CLS token 
        """
        x = x[:, :1, :]  # Keep only the CLS token
        x = x.squeeze(1)  # [B, embed_dim]
        x = self.regressor(x)  # [B, output_dim]

        return x
