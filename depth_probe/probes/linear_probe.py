import torch
import torch.nn as nn

class LinearDepthProbe(nn.Module):
    """Linear probe mapping tokens to 16x16 depth maps."""
    def __init__(self, input_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)
        
    def forward(self, tokens):
        # tokens shape: [B, 256, 768]
        # output shape: [B, 256, 1]
        out = self.proj(tokens)
        # Reshape to [B, 1, 16, 16]
        out = out.transpose(1, 2)  # [B, 1, 256]
        out = out.reshape(-1, 1, 16, 16)  # [B, 1, 16, 16]
        return out
