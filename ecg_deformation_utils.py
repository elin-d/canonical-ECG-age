"""Utility functions for ECG beat preprocessing."""

import torch


def _prepare_mask(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast a beat mask to match tensor layout [B, C, T]."""
    if mask.dim() == 2:
        m = mask.unsqueeze(1)  # [B, 1, T]
    elif mask.dim() == 3:
        if mask.size(1) not in (1, x.size(1)):
            raise ValueError(
                f"Mask channel dimension must be 1 or {x.size(1)}, got {mask.size(1)}."
            )
        m = mask
    else:
        raise ValueError(f"Mask must have shape [B, T] or [B, 1/C, T], got {tuple(mask.shape)}.")
    return m.to(device=x.device, dtype=x.dtype)


def normalize_beat(x: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each beat to zero mean/unit variance, optionally within a beat mask."""
    if mask is None:
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True).clamp(min=eps)
        return (x - mean) / std

    m = _prepare_mask(mask, x)
    denom = m.sum(dim=(1, 2), keepdim=True)
    if m.size(1) == 1:
        denom = denom * x.size(1)
    denom = denom.clamp(min=1.0)

    mean = (x * m).sum(dim=(1, 2), keepdim=True) / denom
    centered = (x - mean) * m
    var = centered.pow(2).sum(dim=(1, 2), keepdim=True) / denom
    std = var.sqrt().clamp(min=eps)
    return (centered / std) * m
