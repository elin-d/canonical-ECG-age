"""Decoders for canonical morphology and factorized age deformation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphologyDecoder(nn.Module):
    """Decode latent morphology into canonical ECG waveform."""

    def __init__(
        self,
        z_dim: int = 256,
        out_leads: int = 12,
        T: int = 500,
        h: int = 128,
        seed: int = 32,
        num_ups: int | None = None,
    ):
        """Build transposed-convolution decoder for fixed-length ECG output.

        Notes:
            The decoder grows the temporal length as: seed * 2**num_ups.
            If num_ups is None, it is chosen as the smallest integer such that
            seed * 2**num_ups >= T. This makes the module automatically scale
            to shorter beats (e.g., T=100 will use 2 upsampling blocks: 32→64→128).
        """
        super().__init__()
        self.T = int(T)
        self.seed = int(seed)
        if self.T <= 0 or self.seed <= 0:
            raise ValueError("T and seed must be positive integers.")

        if num_ups is None:
            n = 0
            L = self.seed
            while L < self.T:
                L *= 2
                n += 1
            num_ups = n
        self.num_ups = int(num_ups)
        if self.num_ups < 0:
            raise ValueError("num_ups must be >= 0.")

        self.proj = nn.Linear(z_dim, h * self.seed)

        def up_block() -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose1d(h, h, kernel_size=4, stride=2, padding=1, bias=False),
                nn.SiLU(),
            )

        self.ups = nn.ModuleList([up_block() for _ in range(self.num_ups)])

        self.refine = nn.Sequential(
            nn.Conv1d(h, h, 15, padding=7, bias=False),
            nn.InstanceNorm1d(h, affine=True),
            nn.SiLU(),
            nn.Conv1d(h, h // 2, 9, padding=4, bias=False),
            nn.InstanceNorm1d(h // 2, affine=True),
            nn.SiLU(),
        )
        self.out = nn.Conv1d(h // 2, out_leads, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate canonical morphology from latent code."""
        B = z.size(0)
        h = F.silu(self.proj(z)).view(B, -1, self.seed)
        for up in self.ups:
            h = up(h)

        # If we slightly overshoot (common), crop to exact T.
        h = h[:, :, : self.T]
        h = self.refine(h)
        return self.out(h)


class MonotonicSplineAgeScalar(nn.Module):
    """Monotonic piecewise-linear g(age) spline with g(a0) = 0.

    Architecture
    ------------
    K=16 knots are fixed and evenly spaced over [AGE_MIN, AGE_MAX].
    We learn K-1 raw increments; softplus maps them to strictly positive
    values, and a cumulative sum builds a strictly increasing g_raw.
    Linear interpolation evaluates g_raw at arbitrary ages.

    g(age) = g_raw(age) - g_raw(a0)

    where a0 is the learnable neutral age passed in at call time.
    This guarantees g(a0) = 0 exactly for any value of a0.
    Monotonicity is global: because all increments are positive, g_raw
    (and therefore g) is strictly increasing with age.
    """

    K: int = 16
    AGE_MIN: float = 10.0
    AGE_MAX: float = 90.0

    def __init__(self) -> None:
        super().__init__()
        # K-1 raw increment scalars; softplus forces all increments > 0.
        self.delta_raw = nn.Parameter(torch.zeros(self.K - 1))

    # Internal helpers
    def _knot_values(self) -> torch.Tensor:
        """Return g_raw at the K knots: [0, c1, c1+c2, …]."""
        increments = F.softplus(self.delta_raw)          # [K-1], all > 0
        return torch.cat(
            [increments.new_zeros(1), torch.cumsum(increments, dim=0)],
            dim=0,
        )                                                # shape [K]

    def _linear_interp(
        self,
        x: torch.Tensor,
        vals: torch.Tensor,
    ) -> torch.Tensor:
        """Piecewise-linear interpolation at query points x.

        Knots are evenly spaced in [AGE_MIN, AGE_MAX], so the segment index
        is computed with direct arithmetic instead of searchsorted.
        This is fully compatible with torch.compile / torch.inductor.
        """
        x = x.clamp(self.AGE_MIN, self.AGE_MAX)
        spacing = (self.AGE_MAX - self.AGE_MIN) / (self.K - 1)
        # Integer segment index in [0, K-2]
        idx = ((x - self.AGE_MIN) / spacing).long().clamp(0, self.K - 2)
        t_l = self.AGE_MIN + idx.float() * spacing
        v_l = vals[idx]
        v_r = vals[idx + 1]
        alpha = (x - t_l) / spacing
        return v_l + alpha * (v_r - v_l)


    # Public API
    def forward(self, age: torch.Tensor, neutral_age: torch.Tensor) -> torch.Tensor:
        """Return g(age) = g_raw(age) - g_raw(neutral_age).

        Parameters
        ----------
        age:         [B] ages in years.
        neutral_age: scalar or [1] tensor — the learnable neutral age a0.
                     Must be a live tensor so gradients flow to a0.
        """
        age = age.float().view(-1)
        neutral_age = neutral_age.float().view(1)
        g_vals = self._knot_values().to(device=age.device, dtype=age.dtype)
        g_age = self._linear_interp(age, g_vals)
        g_a0  = self._linear_interp(neutral_age, g_vals)
        return g_age - g_a0

