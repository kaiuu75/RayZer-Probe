import torch
import torch.nn as nn

class MLPDepthProbe(nn.Module):
    """3-layer MLP probe with GELU activations, mapping tokens to 16x16 depth maps."""
    def __init__(self, input_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.GELU(),
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, 1),
        )
        
    def forward(self, tokens):
        # tokens shape: [B, 256, 768]
        # output shape: [B, 256, 1]
        out = self.mlp(tokens)
        # Reshape to [B, 1, 16, 16]
        out = out.transpose(1, 2)  # [B, 1, 256]
        out = out.reshape(-1, 1, 16, 16)  # [B, 1, 16, 16]
        return out
