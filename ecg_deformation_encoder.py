"""Morphology encoder for age-invariant ECG representation learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphologyEncoder(nn.Module):
    """Encode ECG beats into a compact morphology code."""

    def __init__(self, input_channels: int = 12, embedding_dim: int = 256):
        """Build derivative front-end and temporal convolutional backbone."""
        super().__init__()
        d_weight = torch.zeros(input_channels * 3, input_channels, 3)
        for i in range(input_channels):
            d_weight[i, i, 1] = 1.0
            d_weight[input_channels + i, i, 1] = 1.0
            d_weight[input_channels + i, i, 0] = -1.0
            d_weight[2 * input_channels + i, i, 1] = 1.0
            d_weight[2 * input_channels + i, i, 0] = -2.0
            d_weight[2 * input_channels + i, i, 2] = 1.0

        self.register_buffer("deriv_weight", d_weight)

        in_ch = input_channels * 3
        self.conv1 = nn.Conv1d(in_ch, 64, 25, padding=12, bias=False)
        self.norm1 = nn.InstanceNorm1d(64, affine=True)
        self.conv2 = nn.Conv1d(64, 128, 25, dilation=2, padding=24, bias=False)
        self.norm2 = nn.InstanceNorm1d(128, affine=True)
        self.conv3 = nn.Conv1d(128, 128, 21, dilation=4, padding=40, bias=False)
        self.norm3 = nn.InstanceNorm1d(128, affine=True)
        self.conv4 = nn.Conv1d(128, embedding_dim, 15, dilation=8, padding=56, bias=False)
        self.norm4 = nn.InstanceNorm1d(embedding_dim, affine=True)

    def _fused_derivatives(self, x: torch.Tensor) -> torch.Tensor:
        """Compute signal, first derivative, and second derivative in one convolution."""
        return F.conv1d(x, self.deriv_weight, bias=None, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a per-beat embedding normalized in feature space."""
        h = self._fused_derivatives(x)
        h = F.silu(self.norm1(self.conv1(h)))
        h = F.silu(self.norm2(self.conv2(h)))
        h = F.silu(self.norm3(self.conv3(h)))
        h = F.silu(self.norm4(self.conv4(h)))
        z = h.mean(dim=2)
        z = F.layer_norm(z, z.shape[1:])
        return z
