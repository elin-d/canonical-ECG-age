"""Main ECG deformation model with strict 1D global aging axis."""

import torch
import torch.nn as nn

from ecg_deformation_age_heads import AgeScalarHead
from ecg_deformation_decoders import MonotonicSplineAgeScalar, MorphologyDecoder
from ecg_deformation_encoder import MorphologyEncoder


class ECGDeformationModel(nn.Module):
    """Factorize ECG deformation as Delta(z, a) = g(a) * B(z).

    Key design decisions
    --------------------
    * g(age) is a monotonic piecewise-linear spline (K=16 knots, 10–90 y).
    * The neutral age a0 is a learnable scalar parameter, clamped to [20, 80].
    * g(a0) = 0 by construction, so Delta(a0) = 0 exactly.
    * No antisymmetry constraint: aging can be asymmetric and nonlinear.
    """

    AGE_SCALE = 60.0
    EPS = 1e-6
    PRETRAIN_G_SCALE = 0.5
    DRIFT_CTRL_POINTS = 8
    DRIFT_REL_AMP = 0.10

    def __init__(self, beat_length: int = 100):
        """Initialize strict canonical-age factorization components."""
        super().__init__()
        self.beat_length = int(beat_length)
        if self.beat_length <= 0:
            raise ValueError("beat_length must be a positive integer.")

        self.encoder = MorphologyEncoder()
        self.morph_decoder = MorphologyDecoder(T=self.beat_length)

        # Learnable neutral age  (clamped to [20, 80] via property)
        self.neutral_age_param = nn.Parameter(torch.tensor(50.0))

        # ------------------------------------------------------------------
        # Global aging axis
        # ------------------------------------------------------------------
        # Single shared sensitivity direction across all samples.
        self.sensitivity_dir = nn.Parameter(0.01 * torch.randn(1, 12, self.beat_length))

        # Flag used by training/validation to skip per-sample B-collapse checks.
        self.is_global_axis = True

        # ------------------------------------------------------------------
        # Monotonic spline g(age)
        # ------------------------------------------------------------------
        self.age_scalar_net = MonotonicSplineAgeScalar()

        # Age head: calibrates scalar axis coordinate g_hat → normalized age.
        self.age_head = AgeScalarHead()


    # Neutral-age helpers
    # ------------------------------------------------------------------
    @property
    def neutral_age(self) -> torch.Tensor:
        """Learnable neutral age, clamped to [20, 80]."""
        return self.neutral_age_param.clamp(20.0, 80.0)

    # ------------------------------------------------------------------
    # Age normalization
    # age_to_norm: a0 detached — L_age targets must not train a0
    # norm_to_age: a0 live — gradients from L_age/MAE reach a0 via age_pred_years
    # ------------------------------------------------------------------

    def age_to_norm(self, age_years: torch.Tensor) -> torch.Tensor:
        """Normalize age in years: u = (a - a0) / AGE_SCALE."""
        a0 = self.neutral_age.detach()
        return (age_years.float().view(-1) - a0) / self.AGE_SCALE

    def norm_to_age(self, age_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized age back to years: a = u * AGE_SCALE + a0."""
        a0 = self.neutral_age          # live — gradient flows to neutral_age_param
        return age_norm.float().view(-1) * self.AGE_SCALE + a0

    # ------------------------------------------------------------------
    # g(age): monotonic nonlinear scalar
    # ------------------------------------------------------------------

    def age_scalar(self, a: torch.Tensor) -> torch.Tensor:
        """Map age in years to scalar axis coordinate g(a).

        g is strictly increasing; g(neutral_age) = 0.
        Gradient flows through both the spline parameters (delta_raw) and
        neutral_age_param (via the g_raw(a0) subtraction term).
        """
        return self.age_scalar_net(a.float().view(-1), self.neutral_age)

    # ------------------------------------------------------------------
    # Encoder / decoder helpers
    # ------------------------------------------------------------------

    def _encode_impl(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.compile
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode_impl(x)

    def canonical(self, z: torch.Tensor) -> torch.Tensor:
        """Decode canonical morphology x_canon(z)."""
        return self.morph_decoder(z)

    def sensitivity(self, z: torch.Tensor) -> torch.Tensor:
        """Return global sensitivity field B (independent of z)."""
        b = self.sensitivity_dir
        if b.size(0) != z.size(0):
            b = b.expand(z.size(0), -1, -1)
        return b

    @staticmethod
    def _normalize_field(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        n = v.flatten(1).norm(dim=1, keepdim=True).clamp(min=eps)
        return v / n.view(-1, 1, 1)

    def _sensitivity_dir_field(self, z: torch.Tensor) -> torch.Tensor:
        return self._normalize_field(self.sensitivity(z), eps=self.EPS)

    @staticmethod
    def _axis_projection(delta: torch.Tensor, b_dir: torch.Tensor) -> torch.Tensor:
        return (delta * b_dir).sum(dim=(1, 2))

    # ------------------------------------------------------------------
    # Deformation: Delta(z, a) = g(a) * B_dir(z)
    # ------------------------------------------------------------------

    def _deformation_impl(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        B = self._sensitivity_dir_field(z)
        g = self.age_scalar(a).view(-1, 1, 1)
        return g * B

    @torch.compile
    def deformation(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self._deformation_impl(z, a)

    def _decode_impl(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.canonical(z) + self._deformation_impl(z, a)

    @torch.compile
    def decode(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self._decode_impl(z, a)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_impl(self, x: torch.Tensor, age: torch.Tensor, mode: str = "axis"):
        if mode not in {"pretrain", "axis"}:
            raise ValueError(f"Unknown forward mode '{mode}'. Expected 'pretrain' or 'axis'.")

        z = self._encode_impl(x)
        age_norm = self.age_to_norm(age)

        x_morph = self.canonical(z)
        B_raw = self.sensitivity(z)
        B_dir = self._normalize_field(B_raw, eps=self.EPS)
        B_dir_aug = None
        z_delta_norm = x.new_tensor(float("nan"))

        if mode == "axis":
            b, c, t = x.shape
            lead_std  = x.std(dim=2, keepdim=True).clamp(min=1e-6)
            lead_scale = x.new_empty(b, c, 1).uniform_(0.70, 1.30)
            lead_shift = x.new_empty(b, c, 1).uniform_(-0.10, 0.10)
            x_aug = x * lead_scale + lead_shift * lead_std
            max_k = int(self.DRIFT_CTRL_POINTS)
            k = torch.randint(4, max_k + 1, (), device=x.device, dtype=torch.int64)  # 0-dim int64 tensor
            ctrl_v = x.new_empty(b, c, max_k).uniform_(-1.0, 1.0) * (self.DRIFT_REL_AMP * lead_std)
            time = torch.arange(t, device=x.device, dtype=x.dtype)
            spacing = (time.new_tensor(t - 1.0)) / (k.to(time.dtype) - 1.0)
            seg_idx = torch.floor(time / spacing).to(torch.int64)
            seg_idx = torch.minimum(seg_idx, k - 2)
            seg_idx = torch.clamp(seg_idx, min=0)
            seg_l = seg_idx.view(1, 1, t).expand(b, c, t)
            seg_r = (seg_idx + 1).view(1, 1, t).expand(b, c, t)
            t_l = (seg_idx.to(time.dtype) * spacing).view(1, 1, t)
            t_r = ((seg_idx + 1).to(time.dtype) * spacing).view(1, 1, t)

            alpha = (time.view(1, 1, t) - t_l) / (t_r - t_l).clamp(min=1e-6)

            v_l = ctrl_v.gather(dim=2, index=seg_l)
            v_r = ctrl_v.gather(dim=2, index=seg_r)

            x_aug = x_aug + (1.0 - alpha) * v_l + alpha * v_r

            ctrl_t = torch.linspace(0, t - 1, steps=k, device=x.device, dtype=x.dtype)
            ctrl_v = x.new_empty(b, c, k).uniform_(-1.0, 1.0) * (self.DRIFT_REL_AMP * lead_std)
            time   = torch.arange(t, device=x.device, dtype=x.dtype)
            # ctrl_t is evenly spaced (linspace), so the segment index is direct
            # arithmetic — avoids searchsorted(ctrl_t[1:], ...) which produces a
            # SliceView that TorchInductor cannot stride.
            spacing_ctrl = ctrl_t[1] - ctrl_t[0]   # scalar tensor, integer index
            seg_idx = (time / spacing_ctrl).long().clamp(0, k - 2)
            seg_l = seg_idx.view(1, 1, t).expand(b, c, t)
            seg_r = (seg_idx + 1).view(1, 1, t).expand(b, c, t)
            t_l   = ctrl_t[seg_idx].view(1, 1, t)
            t_r   = ctrl_t[seg_idx + 1].view(1, 1, t)
            alpha = (time.view(1, 1, t) - t_l) / (t_r - t_l).clamp(min=1e-6)
            v_l   = ctrl_v.gather(dim=2, index=seg_l)
            v_r   = ctrl_v.gather(dim=2, index=seg_r)
            x_aug = x_aug + (1.0 - alpha) * v_l + alpha * v_r

            z_aug    = self._encode_impl(x_aug)
            B_raw_aug = self.sensitivity(z_aug)
            B_dir_aug = self._normalize_field(B_raw_aug, eps=self.EPS)
            with torch.no_grad():
                z_delta_norm = (z_aug - z).flatten(1).norm(dim=1).mean()

        # ----------------------------------------------------------------
        # Monotonic scalar coordinate g(age)
        # ----------------------------------------------------------------
        g_age     = self.age_scalar(age)        # [B], gradient to delta_raw + a0
        g_age_det = g_age.detach()

        # ----------------------------------------------------------------
        # Reconstruction path
        # ----------------------------------------------------------------
        if mode == "pretrain":
            g_pre   = torch.empty_like(g_age_det).uniform_(
                -self.PRETRAIN_G_SCALE, self.PRETRAIN_G_SCALE
            ).detach()
            delta_age = g_pre.view(-1, 1, 1) * B_dir
        else:
            delta_age = g_age_det.view(-1, 1, 1) * B_dir
        x_recon = x_morph + delta_age

        # ----------------------------------------------------------------
        # Deformation at neutral age (should be zero by construction)
        # a0_full = age * 0.0 + a0 preserves gradient to neutral_age_param
        # ----------------------------------------------------------------
        a0       = self.neutral_age                    # live tensor [scalar]
        a0_full  = age * 0.0 + a0                     # [B], gradient-preserving
        g_a0     = self.age_scalar(a0_full)            # ≡ 0 by spline construction
        delta_a0 = g_a0.view(-1, 1, 1) * B_dir        # L_def0 sanity check

        # ----------------------------------------------------------------
        # Predictive path: infer age from observed deformation
        # x_morph detached — no gradient into canonical branch.
        # B_dir NOT detached — L_age and L_rank train sensitivity_dir via g_hat.
        # ----------------------------------------------------------------
        delta_obs = x - x_morph.detach()
        g_hat     = self._axis_projection(delta_obs, B_dir)
        age_pred  = self.age_head(g_hat)
        age_pred_years = self.norm_to_age(age_pred)

        # ----------------------------------------------------------------
        # Latent anchor
        # ----------------------------------------------------------------
        z2 = self._encode_impl(x_morph.detach())

        return dict(
            z=z,
            B=B_dir,
            B_dir_aug=B_dir_aug,
            B_raw=B_raw,
            g_age=g_age,
            g_age_det=g_age_det,
            x_morph=x_morph,
            x_recon=x_recon,
            delta=delta_age,
            delta_obs=delta_obs,
            delta_50=delta_a0,
            g_hat=g_hat,
            age_pred=age_pred,
            age_pred_years=age_pred_years,
            age_norm=age_norm,
            z_delta_norm=z_delta_norm,
            z2=z2,
            neutral_age_param=self.neutral_age_param,
        )

    @torch.compile(fullgraph=False, dynamic=True)
    def forward(self, x: torch.Tensor, age: torch.Tensor, mode: str = "axis"):
        return self._forward_impl(x, age, mode=mode)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _project_delta_onto_axis(self, x: torch.Tensor) -> torch.Tensor:
        z       = self.encode(x)
        x_morph = self.canonical(z)
        b       = self._sensitivity_dir_field(z)
        return self._axis_projection(x - x_morph, b)

    @torch.no_grad()
    def predict_age(self, x: torch.Tensor) -> torch.Tensor:
        g_hat        = self._project_delta_onto_axis(x)
        age_pred_norm = self.age_head(g_hat)
        return self.norm_to_age(age_pred_norm)

    def predict_age_center(self, g_hat: torch.Tensor) -> torch.Tensor:
        age_pred_norm = self.age_head(g_hat)
        return self.norm_to_age(age_pred_norm)