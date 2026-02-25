"""Age conditioning embeddings for deformation decoding."""

import math

import torch
import torch.nn as nn


class SinusoidalAgeEmbedding(nn.Module):
    """Map scalar age to a multi-scale sinusoidal embedding."""

    def __init__(self, n: int = 8, tau_min: float = 1.0, tau_max: float = 100.0):
        """Create log-spaced temporal scales for sinusoidal projection."""
        super().__init__()
        taus = torch.exp(torch.linspace(math.log(tau_min), math.log(tau_max), n))
        self.register_buffer("taus", taus)
        self.embed_dim = 2 * n

    def forward(self, age: torch.Tensor) -> torch.Tensor:
        """Embed age values as concatenated sine and cosine features."""
        age = age.view(-1, 1).float()
        scaled = age / self.taus.unsqueeze(0)
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)
