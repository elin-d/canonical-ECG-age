"""Age prediction head for the global deformation axis."""

import torch
import torch.nn as nn


class AgeScalarHead(nn.Module):
    """Calibrate age from scalar axis coordinate g_hat."""

    def __init__(self, hidden: int = 32):
        """Build a small calibrator mapping g_hat -> age."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, g_hat: torch.Tensor) -> torch.Tensor:
        """Estimate age from scalar projection on the aging axis."""
        if g_hat.dim() == 1:
            g_hat = g_hat.unsqueeze(1)
        elif g_hat.dim() != 2 or g_hat.size(1) != 1:
            raise ValueError(f"Expected g_hat shape [B] or [B,1], got {tuple(g_hat.shape)}.")
        return self.net(g_hat).squeeze(-1)
