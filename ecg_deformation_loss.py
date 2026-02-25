"""Training loss for ECG deformation factorization."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformationLoss(nn.Module):
    """Objective for global-axis canonical-age deformation with monotonic g(age).

    Term overview
    -------------
    L_rec       : waveform reconstruction fidelity.
    L_age       : age prediction from axis projection.
    L_anchor    : latent consistency between x and canonical branch.
    L_def0      : deformation at neutral age a0 should be zero (sanity).
    L_transport : B_dir invariant to amplitude-only augmentations.
    L_center    : keep neutral age a0 near 50 (prevents drift).
    L_smooth    : bounded second finite differences of g(age) (smoothness).
    L_axis      : g_hat ≈ g(age) consistency.
    L_rank      : pairwise rank alignment — g_hat(i) > g_hat(j) whenever age(i) > age(j).
    L_div       : B diversity (zero in global-axis mode).
    L_repel     : B repulsion (zero in global-axis mode).
    """

    # Age grid for smoothness penalty (integer years, avoids boundary effects).
    _SMOOTH_AGE_MIN: float = 11.0
    _SMOOTH_AGE_MAX: float = 89.0
    _SMOOTH_N: int = 79   # 11..89 gives 79 points, 77 second differences

    def __init__(
        self,
        w_rec: float = 1.0,
        w_age: float = 1.0,
        w_anchor: float = 0.3,
        w_def0: float = 0.3,
        w_transport: float = 0.005,
        w_center: float = 1e-3,
        w_smooth: float = 1e-3,
        w_div: float = 0.01,
        w_repel: float = 0.05,
        w_axis: float = 1.0,
        w_rank: float = 0.0,
    ):
        super().__init__()
        self.w_rec = w_rec
        self.w_age = w_age
        self.w_anchor = w_anchor
        self.w_def0 = w_def0
        self.w_transport = w_transport
        self.w_center = w_center
        self.w_smooth = w_smooth
        self.w_div = w_div
        self.w_repel = w_repel
        self.w_axis = w_axis
        self.w_rank = w_rank

    @staticmethod
    def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return F.l1_loss(pred, target)

        if mask.dim() == 2:
            m = mask.unsqueeze(1)
        elif mask.dim() == 3:
            if mask.size(1) not in (1, pred.size(1)):
                raise ValueError(
                    f"Mask channel dimension must be 1 or {pred.size(1)}, got {mask.size(1)}."
                )
            m = mask
        else:
            raise ValueError(f"Mask must have shape [B, T] or [B, 1/C, T], got {tuple(mask.shape)}.")

        m = m.to(device=pred.device, dtype=pred.dtype)
        abs_err = (pred - target).abs() * m
        denom = m.sum()
        if m.size(1) == 1:
            denom = denom * pred.size(1)
        denom = denom.clamp(min=1.0)
        return abs_err.sum() / denom

    @staticmethod
    def _g_smooth_loss(
        model: nn.Module,
        device: torch.device,
        age_min: float,
        age_max: float,
        n: int,
    ) -> torch.Tensor:
        """Penalize second finite differences of g(age) on an integer age grid."""
        ages = torch.linspace(age_min, age_max, n, device=device, dtype=torch.float32)
        # Retrieve neutral_age from model (must be live tensor for gradient).
        a0 = model.neutral_age                     # property, clamped [20,80]
        g  = model.age_scalar_net(ages, a0)        # [n], monotone
        d2 = g[2:] - 2.0 * g[1:-1] + g[:-2]      # second differences [n-2]
        return (d2 ** 2).mean()

    @staticmethod
    def _rank_loss(g_hat: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        """Pairwise softplus ranking: penalize g_hat(i) <= g_hat(j) when age(i) > age(j)."""
        n = g_hat.size(0)
        if n < 2:
            return g_hat.new_zeros(())
        diff_age = age.float().view(n, 1) - age.float().view(1, n)   # [n, n]
        diff_g   = g_hat.view(n, 1) - g_hat.view(1, n)               # [n, n]
        mask = diff_age > 0          # pairs where age[i] > age[j] → want g_hat[i] > g_hat[j]
        if not mask.any():
            return g_hat.new_zeros(())
        return F.softplus(-diff_g[mask]).mean()

    def forward(
        self,
        o: dict,
        x: torch.Tensor,
        age: torch.Tensor,
        mask: torch.Tensor | None = None,
        model: nn.Module | None = None,
    ):
        """Compute weighted training objective and detached diagnostics.

        Parameters
        ----------
        o     : output dict from model._forward_impl.
        x     : input ECG batch [B, C, T].
        age   : true ages [B].
        mask  : optional validity mask.
        model : model reference needed for L_smooth; if None, L_smooth = 0.
        """

        # Core losses
        # ----------------------------------------------------------------
        L_rec = self._masked_l1(o["x_recon"], x, mask)

        age_target = o.get("age_norm", age.float().view(-1))
        L_age  = F.l1_loss(o["age_pred"], age_target)

        z  = o["z"]
        z2 = o.get("z2", None)
        if z2 is None:
            z2 = z.detach()
        L_anchor = F.mse_loss(z, z2)

        # Deformation at neutral age should be ≈ 0 (always true by construction,
        # kept as a sanity / gradient-health check).
        delta_a0 = o.get("delta_50", None)   # "delta_50" key now holds delta_a0
        if delta_a0 is None:
            L_def0 = x.new_zeros(())
        else:
            L_def0 = delta_a0.flatten(1).norm(dim=1).mean()

        # Transport invariance: B should not change under amplitude augmentation.
        B_dir_aug = o.get("B_dir_aug", None)
        if B_dir_aug is None:
            L_transport = x.new_zeros(())
        else:
            b1 = F.normalize(o["B"].flatten(1), p=2, dim=1, eps=1e-8)
            b2 = F.normalize(B_dir_aug.flatten(1), p=2, dim=1, eps=1e-8)
            L_transport = (1.0 - (b1 * b2).sum(dim=1)).mean()

        # ----------------------------------------------------------------
        # Neutral-age center regularizer: prevent a0 from drifting far from 50.
        # Gradient flows directly to neutral_age_param.
        # ----------------------------------------------------------------
        neutral_age_param = o.get("neutral_age_param", None)
        if neutral_age_param is not None:
            L_center = (neutral_age_param - 50.0) ** 2
        else:
            L_center = x.new_zeros(())

        # ----------------------------------------------------------------
        # Spline smoothness: penalize large second differences of g(age).
        # ----------------------------------------------------------------
        if model is not None and self.w_smooth > 0.0:
            L_smooth = self._g_smooth_loss(
                model,
                device=x.device,
                age_min=self._SMOOTH_AGE_MIN,
                age_max=self._SMOOTH_AGE_MAX,
                n=self._SMOOTH_N,
            )
        else:
            L_smooth = x.new_zeros(())

        # Axis consistency: g_hat ≈ g(age)
        g_hat = o.get("g_hat", None)
        g_age = o.get("g_age", None)
        if g_hat is None or g_age is None:
            L_axis = x.new_zeros(())
        else:
            L_axis = F.l1_loss(g_hat.detach(), g_age)

        # ----------------------------------------------------------------
        # Pairwise rank alignment: g_hat(i) > g_hat(j) when age(i) > age(j).
        # Gradient flows through g_hat → B_dir → sensitivity_dir.
        # ----------------------------------------------------------------
        if g_hat is None:
            L_rank = x.new_zeros(())
        else:
            L_rank = self._rank_loss(g_hat, age)

        # Diversity / repulsion (zero in global-axis mode; weights set to 0 by schedule).
        L_div   = x.new_zeros(())
        L_repel = x.new_zeros(())

        loss = (
            self.w_rec       * L_rec
            + self.w_age     * L_age
            + self.w_anchor  * L_anchor
            + self.w_def0    * L_def0
            + self.w_transport * L_transport
            + self.w_center  * L_center
            + self.w_smooth  * L_smooth
            + self.w_div     * L_div
            + self.w_repel   * L_repel
            + self.w_axis    * L_axis
            + self.w_rank    * L_rank
        )

        return {
            "loss":        loss,
            "L_rec":       L_rec.detach(),
            "L_age":       L_age.detach(),
            "L_anchor":    L_anchor.detach(),
            "L_def0":      L_def0.detach(),
            "L_transport": L_transport.detach(),
            "L_center":    L_center.detach(),
            "L_smooth":    L_smooth.detach(),
            "L_repel":     L_repel.detach(),
            "L_div":       L_div.detach(),
            "L_axis":      L_axis.detach(),
            "L_rank":      L_rank.detach(),
        }