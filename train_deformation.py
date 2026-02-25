"""Training script for the ECG canonical-age deformation model."""

import os
import time
import random

import numpy as np
import numpy.random as nprand
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_float32_matmul_precision('high')


from ecg_deformation_model import (
    ECGDeformationModel,
    DeformationLoss,
    normalize_beat,
)

try:
    from clearml import Task
    _CLEARML_AVAILABLE = True
except ImportError:
    _CLEARML_AVAILABLE = False

from ECG_Dataset import ECG_Dataset
from ecg_beat_batcher import BeatBatch, ECGBeatBatcher
from deformation_validation import _is_global_axis_model, run_all_validation_tests



# Augmentation
# ---------------------------------------------------------------------------
class AmplitudeAugmentation:
    """
    Amplitude-only augmentations that preserve ECG morphology.
    Applied every training batch to prevent amplitude→age shortcuts.
    """

    @staticmethod
    def random_gain(x: torch.Tensor, min_gain: float = 0.3, max_gain: float = 3.0) -> torch.Tensor:
        gains = x.new_empty(x.size(0), 1, 1).uniform_(min_gain, max_gain)
        return x * gains

    @staticmethod
    def per_lead_scaling(x: torch.Tensor, min_scale: float = 0.5, max_scale: float = 2.0) -> torch.Tensor:
        scales = x.new_empty(x.size(0), x.size(1), 1).uniform_(min_scale, max_scale)
        return x * scales

    @staticmethod
    def baseline_shift(
        x: torch.Tensor,
        max_rel_amp: float = 0.10,
        min_ctrl_points: int = 4,
        max_ctrl_points: int = 8,
    ) -> torch.Tensor:
        b, c, t = x.shape
        if t <= 1:
            return x

        std = x.std(dim=2, keepdim=True).clamp(min=1e-6)
        k = int(torch.randint(min_ctrl_points, max_ctrl_points + 1, (1,), device=x.device).item())

        ctrl_t = torch.linspace(0, t - 1, steps=k, device=x.device, dtype=x.dtype)
        amp = max_rel_amp * std
        ctrl_v = x.new_empty(b, c, k).uniform_(-1.0, 1.0) * amp

        time = torch.arange(t, device=x.device, dtype=x.dtype)
        seg_idx = torch.searchsorted(ctrl_t[1:], time, right=True).clamp(max=k - 2)
        seg_l = seg_idx.view(1, 1, t).expand(b, c, t)
        seg_r = (seg_idx + 1).view(1, 1, t).expand(b, c, t)

        t_l = ctrl_t[seg_idx].view(1, 1, t)
        t_r = ctrl_t[seg_idx + 1].view(1, 1, t)
        alpha = (time.view(1, 1, t) - t_l) / (t_r - t_l).clamp(min=1e-6)

        v_l = ctrl_v.gather(dim=2, index=seg_l)
        v_r = ctrl_v.gather(dim=2, index=seg_r)
        drift = (1.0 - alpha) * v_l + alpha * v_r
        return x + drift

    @classmethod
    def apply_all(cls, x: torch.Tensor) -> torch.Tensor:
        x = cls.random_gain(x)
        x = cls.per_lead_scaling(x)
        x = cls.baseline_shift(x)
        return x


# Logging helper
# ---------------------------------------------------------------------------
def _report_scalar(logger, title: str, series: str, step: int, value: float) -> None:
    """
    Wrapper around ClearML report_scalar that guarantees step is a Python int.
    """
    if logger is not None:
        logger.report_scalar(title, series, iteration=int(step), value=float(value))


def _tensor_stats(t: torch.Tensor) -> dict[str, float]:
    """Return lightweight summary stats for a tensor."""
    x = t.detach().float().reshape(-1)
    if x.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def _safe_pearson_corr_torch_or_np(x, y, eps: float = 1e-12) -> float:
    """
    Pearson correlation that accepts either numpy arrays or torch tensors.
    Returns nan if not enough samples or degenerate variance.
    """
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    else:
        x = np.asarray(x, dtype=np.float64)
    if torch.is_tensor(y):
        y = y.detach().float().cpu().numpy()
    else:
        y = np.asarray(y, dtype=np.float64)

    x = x.reshape(-1).astype(np.float64, copy=False)
    y = y.reshape(-1).astype(np.float64, copy=False)
    if x.size < 2 or y.size < 2:
        return float("nan")

    x = x - x.mean()
    y = y - y.mean()
    den = (np.sqrt((x * x).mean() + eps) * np.sqrt((y * y).mean() + eps))
    if not np.isfinite(den) or den <= 0:
        return float("nan")
    return float((x * y).mean() / den)


def _corr_torch(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Differentiable Pearson correlation (returns scalar tensor).
    Both inputs are treated as 1D vectors.
    """
    a = a.float().view(-1)
    b = b.float().view(-1)
    if int(a.numel()) < 2 or int(b.numel()) < 2:
        return a.new_zeros(())
    a = (a - a.mean()) / (a.std(unbiased=False) + eps)
    b = (b - b.mean()) / (b.std(unbiased=False) + eps)
    return (a * b).mean()


def _module_grad_norm(module: nn.Module | None) -> float:
    """Compute L2 norm of gradients for all parameters in a module."""
    if module is None:
        return float("nan")
    sq_sum = 0.0
    has_grad = False
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        sq_sum += float(g.square().sum().item())
        has_grad = True
    if not has_grad:
        return float("nan")
    return float(np.sqrt(sq_sum))


def _parameter_grad_norm(p: torch.nn.Parameter | None) -> float:
    """L2 norm of a single parameter gradient."""
    if p is None or not isinstance(p, torch.nn.Parameter) or p.grad is None:
        return float("nan")
    return float(p.grad.detach().float().norm().item())


def _mean_pairwise_cosine(v: torch.Tensor) -> float:
    """Mean cosine similarity across random non-identical pairs."""
    n = int(v.size(0))
    if n < 2:
        return 1.0
    flat = v.detach().float().flatten(1)
    flat = F.normalize(flat, p=2, dim=1, eps=1e-8)
    idx = torch.randperm(n, device=flat.device)
    if bool((idx == torch.arange(n, device=flat.device)).any()):
        idx = torch.roll(torch.arange(n, device=flat.device), shifts=1)
    cos = F.cosine_similarity(flat, flat[idx], dim=1)
    return float(cos.mean().item())


def _flatten_parameters(param_spec) -> list[nn.Parameter]:
    """Flatten nested parameter iterables into a concrete parameter list."""
    if param_spec is None:
        return []
    if isinstance(param_spec, nn.Parameter):
        return [param_spec]
    if isinstance(param_spec, dict):
        return _flatten_parameters(param_spec.get("params", None))
    try:
        iterator = iter(param_spec)
    except TypeError:
        return []

    params: list[nn.Parameter] = []
    for item in iterator:
        if isinstance(item, nn.Parameter):
            params.append(item)
        else:
            params.extend(_flatten_parameters(item))
    return params


def check_grad_flow(
    model: nn.Module,
    loss: torch.Tensor,
    groups: dict[str, object],
    strict: bool = False,
    tol: float = 1e-12,
) -> dict[str, float]:
    """
    Measure per-group grad L2 norms using autograd.grad without touching .grad buffers.

    Group entries support:
    - iterable of parameters
    - {"params": <iterable>, "forbidden": bool}
    """
    _ = model
    norms: dict[str, float] = {}
    for group_name, group_spec in groups.items():
        forbidden = False
        params_spec = group_spec
        if isinstance(group_spec, dict):
            params_spec = group_spec.get("params", None)
            forbidden = bool(group_spec.get("forbidden", False))

        params = [p for p in _flatten_parameters(params_spec) if p.requires_grad]
        if not params:
            norms[group_name] = 0.0
            continue

        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        sq_sum = 0.0
        for g in grads:
            if g is None:
                continue
            gf = g.detach().float()
            sq_sum += float(gf.square().sum().item())
        norm = float(np.sqrt(max(sq_sum, 0.0)))
        norms[group_name] = norm

        if strict and forbidden and norm > tol:
            raise AssertionError(
                f"Grad leakage in forbidden group '{group_name}': "
                f"norm={norm:.3e} > tol={tol:.3e}"
            )
    return norms


def _warn_or_raise(message: str, strict: bool) -> None:
    if strict:
        raise AssertionError(message)
    print(f"[warning] {message}")


def _check_norm_expectations(
    tag: str,
    step: int,
    norms: dict[str, float],
    forbidden: list[str],
    allowed: list[str],
    strict: bool,
    tol: float,
) -> None:
    leaks = [name for name in forbidden if float(norms.get(name, 0.0)) > tol]
    dead = [name for name in allowed if float(norms.get(name, 0.0)) <= tol]
    if not leaks and not dead:
        return

    parts = []
    if leaks:
        parts.append(
            "forbidden>tol: " + ", ".join(f"{k}={norms[k]:.3e}" for k in leaks)
        )
    if dead:
        parts.append(
            "allowed<=tol: " + ", ".join(f"{k}={norms[k]:.3e}" for k in dead)
        )
    _warn_or_raise(
        f"[gradcheck step={step}] {tag} expectation mismatch | " + " | ".join(parts),
        strict=strict,
    )


def _log_gradcheck_norms(logger, step: int, scope: str, norms: dict[str, float]) -> None:
    for name, value in norms.items():
        _report_scalar(logger, f"gradcheck/{scope}/{name}", "train", step, value)


def _build_grad_groups(base_model: nn.Module, forbidden: set[str]) -> dict[str, dict[str, object]]:
    return {
        "encoder": {
            "params": base_model.encoder.parameters(),
            "forbidden": "encoder" in forbidden,
        },
        "canonical": {
            "params": base_model.morph_decoder.parameters(),
            "forbidden": "canonical" in forbidden,
        },
        "sensitivity": {
            "params": [getattr(base_model, "sensitivity_dir")],
            "forbidden": "sensitivity" in forbidden,
        },
        "age_scalar": {
            "params": base_model.age_scalar_net.parameters(),
            "forbidden": "age_scalar" in forbidden,
        },
        "neutral_age": {
            "params": [base_model.neutral_age_param],
            "forbidden": "neutral_age" in forbidden,
        },
        "age_head": {
            "params": base_model.age_head.parameters(),
            "forbidden": "age_head" in forbidden,
        },
    }


def _run_graph_wiring_asserts(
    model: nn.Module,
    out: dict[str, torch.Tensor | bool],
    x_sub: torch.Tensor,
    age_sub: torch.Tensor,
    step: int,
    strict: bool,
    age_pred_tol: float,
    forward_mode: str = "axis",
) -> dict[str, float]:
    """
    Cheap structural checks:
    1) generative age scalar must be detached.
    2) age_pred should be invariant to provided age input.
    """
    g_age_flag = bool(out["g_age"].requires_grad)
    g_age_det_flag = bool(out["g_age_det"].requires_grad)

    if g_age_det_flag:
        _warn_or_raise(
            f"[graphcheck step={step}] g_age_det requires_grad=True (must be False)",
            strict=strict,
        )

    train_state = bool(model.training)
    model.eval()
    with torch.no_grad():
        out_ref = model(x_sub, age_sub, mode=forward_mode)
        if int(age_sub.numel()) > 1:
            idx = torch.randperm(int(age_sub.numel()), device=age_sub.device)
            if bool((idx == torch.arange(int(age_sub.numel()), device=age_sub.device)).all()):
                idx = torch.roll(torch.arange(int(age_sub.numel()), device=age_sub.device), shifts=1)
            age_alt = age_sub[idx]
        else:
            age_alt = age_sub + 10.0
        out_alt = model(x_sub, age_alt, mode=forward_mode)
        age_pred_delta = float(torch.abs(out_ref["age_pred"] - out_alt["age_pred"]).mean().item())
    model.train(train_state)

    if age_pred_delta > age_pred_tol:
        _warn_or_raise(
            f"[graphcheck step={step}] age_pred changed with age input: "
            f"mean|Δ|={age_pred_delta:.3e} > tol={age_pred_tol:.3e}",
            strict=strict,
        )

    return {
        "g_age_requires_grad": float(g_age_flag),
        "g_age_det_requires_grad": float(g_age_det_flag),
        "age_pred_delta_vs_age_input": age_pred_delta,
    }


# Validation pass (beat-level)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _log_validation(
    model:     ECGDeformationModel,
    criterion: DeformationLoss,
    val_loader: ECGBeatBatcher,
    step:      int,
    logger=None,
    num_val_batches: int = 20,   # keeps runtime bounded; set None to full pass
    forward_mode: str = "axis",
) -> dict[str, float] | None:
    """
    Validation over multiple beat-batches.

    Returns:
        Dict with validation metrics (weighted by number of beats), or None.
    """
    model.eval()
    base_model = getattr(model, "_orig_mod", model)
    shortcut_warn_mae = 10.0
    age_scale = float(getattr(base_model, "AGE_SCALE", 60.0))

    loss_sum = 0.0
    L_rec_sum = 0.0
    L_age_sum = 0.0
    L_anchor_sum = 0.0
    L_def0_sum = 0.0
    L_transport_sum = 0.0
    L_repel_sum = 0.0
    L_center_sum = 0.0
    L_smooth_sum = 0.0
    L_div_sum = 0.0
    L_axis_sum = 0.0
    abs_err_sum = 0.0
    shortcut_mae_noise_sum = 0.0
    count = 0

    it = iter(val_loader)
    total = len(val_loader) if num_val_batches is None else min(len(val_loader), int(num_val_batches))

    for _ in range(total):
        batch = next(it)

        x_v = normalize_beat(batch.beats, batch.masks)
        a_v = batch.ages.float()

        out = model(x_v, a_v, mode=forward_mode)
        ld  = criterion(out, x_v, a_v, mask=batch.masks, model=base_model)

        # Shortcut detection: randomize age only in generative branch and check MAE.
        n = int(x_v.size(0))
        if n > 1:
            noise_idx = torch.randperm(n, device=a_v.device)
            if bool((noise_idx == torch.arange(n, device=a_v.device)).all()):
                noise_idx = torch.roll(torch.arange(n, device=a_v.device), shifts=1)
            a_noise = a_v[noise_idx]
        else:
            a_noise = a_v

        g_noise = base_model.age_scalar(a_noise)
        delta_noise = g_noise.view(-1, 1, 1) * out["B"]
        g_hat_noise = (delta_noise * out["B"]).sum(dim=(1, 2))
        age_pred_noise = base_model.age_head(g_hat_noise)
        age_pred_noise_years = base_model.norm_to_age(age_pred_noise)
        age_pred_years = out.get("age_pred_years", base_model.norm_to_age(out["age_pred"]))

        loss_sum += float(ld["loss"].item()) * n
        L_rec_sum += float(ld["L_rec"].item()) * n
        L_age_sum += float(ld["L_age"].item()) * n
        L_anchor_sum += float(ld["L_anchor"].item()) * n
        L_def0_sum += float(ld["L_def0"].item()) * n
        L_transport_sum += float(ld["L_transport"].item()) * n
        L_repel_sum += float(ld["L_repel"].item()) * n
        L_center_sum += float(ld["L_center"].item()) * n
        L_smooth_sum += float(ld["L_smooth"].item()) * n
        L_div_sum += float(ld["L_div"].item()) * n
        L_axis_sum += float(ld["L_axis"].item()) * n
        abs_err_sum += float(torch.abs(age_pred_years - a_v).sum().item())
        shortcut_mae_noise_sum += float(torch.abs(age_pred_noise_years - a_v).sum().item())
        count += n

    if count == 0:
        model.train()
        return None

    val_loss = loss_sum / count
    val_L_rec = L_rec_sum / count
    val_L_age = L_age_sum / count
    val_L_anchor = L_anchor_sum / count
    val_L_def0 = L_def0_sum / count
    val_L_transport = L_transport_sum / count
    val_L_repel = L_repel_sum / count
    val_L_center = L_center_sum / count
    val_L_smooth = L_smooth_sum / count
    val_L_div = L_div_sum / count
    val_L_axis = L_axis_sum / count
    val_mae = abs_err_sum / count
    val_shortcut_mae_noise = shortcut_mae_noise_sum / count

    _report_scalar(logger, "loss",        "val", step, val_loss)
    _report_scalar(logger, "L_rec",       "val", step, val_L_rec)
    _report_scalar(logger, "L_age",       "val", step, val_L_age)
    _report_scalar(logger, "MAE",         "val", step, val_mae)
    _report_scalar(logger, "L_anchor",    "val", step, val_L_anchor)
    _report_scalar(logger, "L_def0",      "val", step, val_L_def0)
    _report_scalar(logger, "L_transport", "val", step, val_L_transport)
    _report_scalar(logger, "L_repel",     "val", step, val_L_repel)
    _report_scalar(logger, "L_center",    "val", step, val_L_center)
    _report_scalar(logger, "L_smooth",    "val", step, val_L_smooth)
    _report_scalar(logger, "L_div",       "val", step, val_L_div)
    _report_scalar(logger, "L_axis",      "val", step, val_L_axis)
    _report_scalar(logger, "shortcut_mae_noise_age", "val", step, val_shortcut_mae_noise)
    # Log current neutral age
    _report_scalar(logger, "neutral_age", "val", step,
                   float(getattr(base_model, "neutral_age", torch.tensor(50.0)).item()))

    if val_shortcut_mae_noise < shortcut_warn_mae:
        print(
            f"[warning step={step}] shortcut-detection MAE with noisy age input is low "
            f"({val_shortcut_mae_noise:.3f}y < {shortcut_warn_mae:.1f}y). "
            "Potential age-input leakage shortcut."
        )

    model.train()
    return {
        "loss":                  float(val_loss),
        "L_rec":                 float(val_L_rec),
        "L_age":                 float(val_L_age),
        "MAE":                   float(val_mae),
        "L_anchor":              float(val_L_anchor),
        "L_def0":                float(val_L_def0),
        "L_transport":           float(val_L_transport),
        "L_repel":               float(val_L_repel),
        "L_center":              float(val_L_center),
        "L_smooth":              float(val_L_smooth),
        "L_div":                 float(val_L_div),
        "L_axis":                float(val_L_axis),
        "shortcut_mae_noise_age": float(val_shortcut_mae_noise),
    }


def train_one_epoch(
    model:             ECGDeformationModel,
    criterion:         DeformationLoss,
    optimizer:         optim.Optimizer,
    train_loader:      ECGBeatBatcher,
    val_loader:        ECGBeatBatcher,
    epoch:             int,
    logger=None,
    global_step_start: int = 0,
    sub_batch_size:    int = 256,
    total_steps:       int | None = None,
    grad_assert_every: int = 200,
    grad_assert_strict: bool = False,
    grad_assert_tol: float = 1e-10,
    graph_assert_age_pred_tol: float = 1e-6,
    strict_collapse_assert: bool = False,
    forward_mode: str = "axis",
) -> tuple[float, int, bool]:
    """
    Train for one full epoch with sub-batch gradient accumulation.

    Returns:
        mean_loss:   mean total loss over all batches in this epoch
        global_step: updated integer step counter
    """
    model.train()
    train_loader.seed = 1986 + epoch  # epoch-dependent shuffle, deterministic given seed

    num_batches = len(train_loader)
    epoch_losses: list[float] = []
    step: int = int(global_step_start)

    log_L_rec       = 0.0
    log_L_age       = 0.0
    log_mae         = 0.0
    log_L_anchor    = 0.0
    log_L_def0      = 0.0
    log_L_transport = 0.0
    log_L_repel     = 0.0
    log_L_center    = 0.0
    log_L_smooth    = 0.0
    log_L_div       = 0.0
    log_L_axis      = 0.0
    log_L_rank      = 0.0
    log_count       = 0

    debug_every = 50
    grad_debug_every = 200
    base_model = getattr(model, "_orig_mod", model)
    global_axis_mode = _is_global_axis_model(model)

    age_scale = float(getattr(base_model, "AGE_SCALE", 60.0))
    last_grad_age_scalar = float("nan")
    last_grad_age_head = float("nan")
    last_grad_sensitivity = float("nan")
    graph_wiring_checked = False
    collapse_detected_this_epoch = False

    # For non-global axis models, you might want a collapse detector.
    # For GLOBAL axis, "B collapse" is expected and should NOT trigger watchdog.
    b_collapse_cos_threshold = 0.999
    b_collapse_var_threshold = 1e-6

    pbar_total = int(total_steps) if total_steps is not None else num_batches
    pbar = tqdm(total=pbar_total, initial=global_step_start, desc="Training", leave=False)

    for batch in train_loader:
        x = batch.beats
        mask = batch.masks
        age = batch.ages.float()
        V = x.size(0)

        # 2. Augment → normalize
        x_aug = AmplitudeAugmentation.apply_all(x)
        x_aug = x_aug * mask.unsqueeze(1)
        x_norm = normalize_beat(x_aug, mask)

        # 3. Sub-batch gradient accumulation (main model)
        optimizer.zero_grad()

        total_loss_val        = 0.0
        total_L_rec_val       = 0.0
        total_L_age_val       = 0.0
        total_mae_val         = 0.0
        total_L_anchor_val    = 0.0
        total_L_def0_val      = 0.0
        total_L_transport_val = 0.0
        total_L_repel_val     = 0.0
        total_L_center_val    = 0.0
        total_L_smooth_val    = 0.0
        total_L_div_val       = 0.0
        total_L_axis_val      = 0.0
        total_L_rank_val      = 0.0
        debug_stats: dict[str, float] | None = None
        debug_this_step = (step % debug_every == 0)

        for start in range(0, V, sub_batch_size):
            end   = min(start + sub_batch_size, V)
            scale = (end - start) / V

            out_sub = model(x_norm[start:end], age[start:end], mode=forward_mode)
            ld_sub  = criterion(
                out_sub,
                x_norm[start:end],
                age[start:end],
                mask=mask[start:end],
                model=base_model,
            )

            # ------------------------------------------------------------------
            # Latent age-leak penalty on z (CRITICAL for probe tests)
            # Affects encoder only (loss is computed from z).
            # ------------------------------------------------------------------
            w_leak_z = 2.0

            z_sub = out_sub["z"]  # [B, D]
            age_sub = age[start:end].float()  # [B]
            age_norm = (age_sub - age_sub.mean()) / (age_sub.std(unbiased=False) + 1e-6)

            # normalize age
            b = age_norm.view(-1, 1)  # [B, 1]

            # normalize z per-dim
            z = z_sub.float()
            z = (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, unbiased=False, keepdim=True) + 1e-6)

            # corr per latent dim: mean over batch of z[:,k] * age_norm
            corr_per_dim = (z * b).mean(dim=0)  # [D]
            leak_z_loss = (corr_per_dim ** 2).mean()  # scalar

            # ------------------------------------------------------------------
            # Canonical leakage penalty on x_morph / x50 (CRITICAL)
            # Penalize correlation between simple amplitude stats of x_morph and age.
            # This must only affect encoder + morph_decoder (x_morph path).
            # ------------------------------------------------------------------
            w_leak_morph = 50.0

            x_morph_sub = out_sub["x_morph"]
            mean_abs = x_morph_sub.abs().mean(dim=(1, 2))
            std_val = x_morph_sub.std(dim=(1, 2), unbiased=False)

            age_sub = age[start:end].float()
            age_norm = (age_sub - age_sub.mean()) / (age_sub.std(unbiased=False) + 1e-6)

            leak_loss = _corr_torch(mean_abs, age_norm).square() + _corr_torch(std_val, age_norm).square()

            # Add to total training loss (do NOT route through age_head/age_scalar).
            loss_total_sub = ld_sub["loss"] + (w_leak_morph * leak_loss) + (w_leak_z * leak_z_loss)

            is_first_sub_batch = (start == 0)
            run_gradcheck = (
                is_first_sub_batch
                and grad_assert_every > 0
                and (step % grad_assert_every == 0)
            )

            if is_first_sub_batch and not graph_wiring_checked:
                graph_stats = _run_graph_wiring_asserts(
                    model=model,
                    out=out_sub,
                    x_sub=x_norm[start:end],
                    age_sub=age[start:end].float(),
                    step=step,
                    strict=grad_assert_strict,
                    age_pred_tol=graph_assert_age_pred_tol,
                    forward_mode=forward_mode,
                )
                for k, v in graph_stats.items():
                    _report_scalar(logger, f"graphcheck/{k}", "train", step, v)
                graph_wiring_checked = True

            if run_gradcheck:
                x_sub = x_norm[start:end]
                age_sub = age[start:end].float()
                mask_sub = mask[start:end]
                was_training = bool(base_model.training)
                base_model.eval()
                try:
                    if hasattr(base_model, "_forward_impl"):
                        out_gc = base_model._forward_impl(x_sub, age_sub, mode=forward_mode)
                    else:
                        out_gc = base_model(x_sub, age_sub, mode=forward_mode)

                    age_sub_norm = base_model.age_to_norm(age_sub)
                    L_age_only = F.l1_loss(out_gc["age_pred"], age_sub_norm)
                    L_rec_only = criterion._masked_l1(out_gc["x_recon"], x_sub, mask_sub)
                    L_axis_only = F.l1_loss(out_gc["g_hat"].detach(), out_gc["g_age"])

                    age_groups = _build_grad_groups(
                        base_model,
                        forbidden={"encoder", "canonical", "age_scalar"},
                    )
                    age_norms = check_grad_flow(
                        model=base_model,
                        loss=L_age_only,
                        groups=age_groups,
                        strict=grad_assert_strict,
                        tol=grad_assert_tol,
                    )
                    _check_norm_expectations(
                        tag="L_age",
                        step=step,
                        norms=age_norms,
                        forbidden=["encoder", "canonical", "age_scalar"],
                        allowed=["age_head", "sensitivity"],
                        strict=grad_assert_strict,
                        tol=grad_assert_tol,
                    )
                    _log_gradcheck_norms(logger, step, "age", age_norms)

                    rec_groups = _build_grad_groups(
                        base_model,
                        forbidden={"age_head", "age_scalar", "sensitivity"},
                    )
                    rec_norms = check_grad_flow(
                        model=base_model,
                        loss=L_rec_only,
                        groups=rec_groups,
                        strict=grad_assert_strict,
                        tol=grad_assert_tol,
                    )
                    _check_norm_expectations(
                        tag="L_rec",
                        step=step,
                        norms=rec_norms,
                        forbidden=["age_head", "age_scalar", "sensitivity"],
                        allowed=["encoder", "canonical"],
                        strict=grad_assert_strict,
                        tol=grad_assert_tol,
                    )
                    _log_gradcheck_norms(logger, step, "rec", rec_norms)

                    cheat_line = "L_cheat skipped (pretrain mode)"
                    if forward_mode == "axis":
                        g_hat_cheat = base_model._axis_projection(out_gc["delta"], out_gc["B"])
                        age_pred_cheat = base_model.age_head(g_hat_cheat)
                        L_age_cheat = F.l1_loss(age_pred_cheat, age_sub_norm)
                        cheat_groups = _build_grad_groups(
                            base_model,
                            forbidden={"age_scalar"},
                        )
                        cheat_norms = check_grad_flow(
                            model=base_model,
                            loss=L_age_cheat,
                            groups=cheat_groups,
                            strict=grad_assert_strict,
                            tol=grad_assert_tol,
                        )
                        _check_norm_expectations(
                            tag="L_age_cheat",
                            step=step,
                            norms=cheat_norms,
                            forbidden=["age_scalar"],
                            allowed=[],
                            strict=grad_assert_strict,
                            tol=grad_assert_tol,
                        )
                        _log_gradcheck_norms(logger, step, "cheat", cheat_norms)
                        cheat_line = f"L_cheat grads: g={cheat_norms['age_scalar']:.2e}"

                    axis_groups = _build_grad_groups(
                        base_model,
                        forbidden={"encoder", "canonical", "sensitivity", "age_head"},
                    )
                    axis_norms = check_grad_flow(
                        model=base_model,
                        loss=L_axis_only,
                        groups=axis_groups,
                        strict=grad_assert_strict,
                        tol=grad_assert_tol,
                    )
                    _check_norm_expectations(
                        tag="L_axis",
                        step=step,
                        norms=axis_norms,
                        forbidden=["encoder", "canonical", "sensitivity", "age_head"],
                        allowed=["age_scalar"],
                        strict=grad_assert_strict,
                        tol=grad_assert_tol,
                    )
                    _log_gradcheck_norms(logger, step, "axis", axis_norms)

                    if (
                        float(axis_norms.get("age_scalar", 0.0)) > grad_assert_tol
                        and float(axis_norms.get("sensitivity", 0.0)) > grad_assert_tol
                    ):
                        raise AssertionError(
                            "L_axis gradient routing invalid: both age_scalar and sensitivity received gradients."
                        )

                    print(
                        f"[gradcheck step={step}] "
                        f"L_age grads: enc={age_norms['encoder']:.2e} canon={age_norms['canonical']:.2e} "
                        f"sens={age_norms['sensitivity']:.2e} g={age_norms['age_scalar']:.2e} "
                        f"head={age_norms['age_head']:.2e}  |  "
                        f"L_rec grads: enc={rec_norms['encoder']:.2e} canon={rec_norms['canonical']:.2e} "
                        f"sens={rec_norms['sensitivity']:.2e} g={rec_norms['age_scalar']:.2e} "
                        f"head={rec_norms['age_head']:.2e}  |  "
                        f"{cheat_line}  |  "
                        f"L_axis grads: sens={axis_norms['sensitivity']:.2e} g={axis_norms['age_scalar']:.2e}"
                    )
                finally:
                    base_model.train(was_training)

            (loss_total_sub * scale).backward()

            total_loss_val     += float(loss_total_sub.detach().item())     * scale
            total_L_rec_val    += ld_sub["L_rec"].item()    * scale
            total_L_age_val    += ld_sub["L_age"].item()    * scale
            age_pred_years_sub = out_sub.get("age_pred_years", base_model.norm_to_age(out_sub["age_pred"]))
            total_mae_val      += F.l1_loss(age_pred_years_sub, age[start:end].float()).item() * scale
            total_L_anchor_val    += ld_sub["L_anchor"].item()    * scale
            total_L_def0_val      += ld_sub["L_def0"].item()      * scale
            total_L_transport_val += ld_sub["L_transport"].item() * scale
            total_L_repel_val     += ld_sub["L_repel"].item()     * scale
            total_L_center_val    += ld_sub["L_center"].item()    * scale
            total_L_smooth_val    += ld_sub["L_smooth"].item()    * scale
            total_L_div_val       += ld_sub["L_div"].item()       * scale
            total_L_axis_val      += ld_sub["L_axis"].item()      * scale
            total_L_rank_val      += ld_sub["L_rank"].item()      * scale

            if debug_this_step and start == 0:
                with torch.no_grad():
                    age_sub = age[start:end].float()
                    g_raw = out_sub.get("g_age")
                    if g_raw is None:
                        g_raw = base_model.age_scalar(age_sub)

                    g_hat = out_sub.get("g_hat")
                    if g_hat is None:
                        g_hat = (out_sub["delta"] * out_sub["B"]).sum(dim=(1, 2))

                    age_pred_norm = out_sub["age_pred"]
                    age_pred = out_sub.get("age_pred_years", base_model.norm_to_age(age_pred_norm))
                    b_dir = out_sub["B"]
                    b_dir_aug = out_sub.get("B_dir_aug")
                    b_raw = out_sub.get("B_raw", b_dir)
                    delta = out_sub["delta"]
                    x_morph = out_sub["x_morph"]

                    g_stats = _tensor_stats(g_raw)
                    g_abs_stats = _tensor_stats(g_raw.abs())
                    g_hat_stats = _tensor_stats(g_hat)
                    age_pred_stats = _tensor_stats(age_pred)
                    b_norm_stats = _tensor_stats(b_dir.flatten(1).norm(dim=1))
                    delta_norm_stats = _tensor_stats(delta.flatten(1).norm(dim=1))
                    delta_obs_canon = x_norm[start:end] - x_morph.detach()
                    x50_delta_norm = float(delta_obs_canon.flatten(1).norm(dim=1).mean().item())
                    x50_scalar = x_morph.flatten(1).mean(dim=1)
                    x50_age_corr = _safe_pearson_corr_torch_or_np(x50_scalar, age_sub)

                    transport_cos_aug = float("nan")
                    if b_dir_aug is not None:
                        b1_aug = F.normalize(b_dir.flatten(1), p=2, dim=1, eps=1e-8)
                        b2_aug = F.normalize(b_dir_aug.flatten(1), p=2, dim=1, eps=1e-8)
                        cos_aug = F.cosine_similarity(b1_aug, b2_aug, dim=1).mean()
                        transport_cos_aug = float(cos_aug.item())
                        msg = f"transport_cos_aug={transport_cos_aug:.4f}"
                        if logger is not None and hasattr(logger, "info"):
                            logger.info(msg)
                        else:
                            print(msg)

                    cur_neutral_age = float(
                        getattr(base_model, "neutral_age", torch.tensor(50.0)).item()
                    )
                    age_pred_std = float(age_pred.std(unbiased=False).item())

                    debug_stats = {
                        "g_mean": g_stats["mean"],
                        "g_std": g_stats["std"],
                        "g_min": g_stats["min"],
                        "g_max": g_stats["max"],
                        "g_abs_mean": g_abs_stats["mean"],
                        "g_abs_std": g_abs_stats["std"],
                        "g_hat_mean": g_hat_stats["mean"],
                        "g_hat_std": g_hat_stats["std"],
                        "g_hat_min": g_hat_stats["min"],
                        "g_hat_max": g_hat_stats["max"],
                        "g_hat_corr_age": _safe_pearson_corr_torch_or_np(g_hat, age_sub),
                        "g_hat_corr_g": _safe_pearson_corr_torch_or_np(g_hat, g_raw),
                        "age_pred_mean": age_pred_stats["mean"],
                        "age_pred_std": age_pred_stats["std"],
                        "age_pred_min": age_pred_stats["min"],
                        "age_pred_max": age_pred_stats["max"],
                        "age_pred_near_const": float(age_pred_std < 1.0),
                        "age_mae_sub": float(F.l1_loss(age_pred, age_sub).item()),
                        "b_norm_mean": b_norm_stats["mean"],
                        "b_norm_std": b_norm_stats["std"],
                        "delta_norm_mean": delta_norm_stats["mean"],
                        "delta_norm_std": delta_norm_stats["std"],
                        "x50_delta_norm": x50_delta_norm,
                        "x50_age_corr": x50_age_corr,
                        "leak_morph_sub": float(leak_loss.detach().item()),
                        "neutral_age": cur_neutral_age,
                        "L_center_sub": float(ld_sub["L_center"].item()),
                        "L_smooth_sub": float(ld_sub["L_smooth"].item()),
                        "b_pairwise_cos": _mean_pairwise_cosine(b_dir),
                        "b_feature_var": float(b_dir.flatten(1).var(dim=0, unbiased=False).mean().item()),
                        "b_raw_pairwise_cos": _mean_pairwise_cosine(b_raw),
                        "b_raw_feature_var": float(b_raw.flatten(1).var(dim=0, unbiased=False).mean().item()),
                        "L_repel_sub": float(ld_sub["L_repel"].item()),
                        "L_transport_sub": float(ld_sub["L_transport"].item()),
                        "L_div_sub": float(ld_sub["L_div"].item()),
                        "L_axis_sub": float(ld_sub["L_axis"].item()),
                        "L_rank_sub": float(ld_sub["L_rank"].item()),
                        "transport_cos_aug": transport_cos_aug,
                    }

        # 4. Clip + step
        need_grad_debug = (
            (step % grad_debug_every == 0)
            or (
                debug_this_step
                and (
                    not np.isfinite(last_grad_age_scalar)
                    or not np.isfinite(last_grad_age_head)
                    or not np.isfinite(last_grad_sensitivity)
                )
            )
        )
        if need_grad_debug:
            last_grad_age_scalar = _module_grad_norm(getattr(base_model, "age_scalar_net", None))
            last_grad_age_head = _module_grad_norm(getattr(base_model, "age_head", None))
            last_grad_sensitivity = _parameter_grad_norm(getattr(base_model, "sensitivity_dir", None))

        pre_clip_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        global_grad_norm = float(pre_clip_grad_norm.item() if torch.is_tensor(pre_clip_grad_norm) else pre_clip_grad_norm)
        optimizer.step()

        epoch_losses.append(total_loss_val)

        # 5. Accumulate for smoothed logging
        log_L_rec       += total_L_rec_val
        log_L_age       += total_L_age_val
        log_mae         += total_mae_val
        log_L_anchor    += total_L_anchor_val
        log_L_def0      += total_L_def0_val
        log_L_transport += total_L_transport_val
        log_L_repel     += total_L_repel_val
        log_L_center    += total_L_center_val
        log_L_smooth    += total_L_smooth_val
        log_L_div       += total_L_div_val
        log_L_axis      += total_L_axis_val
        log_L_rank      += total_L_rank_val
        log_count       += 1

        if debug_this_step and debug_stats is not None:
            debug_stats["grad_global_preclip"] = global_grad_norm
            debug_stats["grad_age_scalar"] = last_grad_age_scalar
            debug_stats["grad_age_head"] = last_grad_age_head
            debug_stats["grad_sensitivity_dir"] = last_grad_sensitivity

            o = out_sub
            z_delta = o.get("z_delta_norm", None)
            if z_delta is not None:
                z_delta = float(z_delta)
            z_delta_str = f"  z_dnorm={z_delta:.4f}" if z_delta is not None else ""

            b_pairwise_cos = float(debug_stats["b_pairwise_cos"])
            b_feature_var = float(debug_stats["b_feature_var"])
            b_raw_pairwise_cos = float(debug_stats["b_raw_pairwise_cos"])
            b_raw_feature_var = float(debug_stats["b_raw_feature_var"])
            b_collapse = (
                (
                    (b_pairwise_cos > b_collapse_cos_threshold)
                    and (b_feature_var < b_collapse_var_threshold)
                )
                or (
                    (b_raw_pairwise_cos > b_collapse_cos_threshold)
                    and (b_raw_feature_var < b_collapse_var_threshold)
                )
            )
            debug_stats["b_collapse_flag"] = float(b_collapse)

            if global_axis_mode:
                b_collapse = False
                debug_stats["b_collapse_flag"] = 0.0

            if b_collapse:
                collapse_detected_this_epoch = True
                print(
                    f"WARNING: B collapse detected | step={step} "
                    f"dir(pairwise_cos={b_pairwise_cos:.6f}, feature_var={b_feature_var:.3e}) "
                    f"raw(pairwise_cos={b_raw_pairwise_cos:.6f}, feature_var={b_raw_feature_var:.3e})"
                )
                if strict_collapse_assert:
                    raise AssertionError("B collapse detected (strict_collapse_assert=True).")

            print(
                f"[debug step={step}] "
                f"loss={total_loss_val:.4f}  L_rec={total_L_rec_val:.4f}  L_age={total_L_age_val:.4f}  "
                f"mae={total_mae_val:.4f}  L_anchor={total_L_anchor_val:.4f}  "
                f"L_def0={total_L_def0_val:.4f}  "
                f"L_trans={total_L_transport_val:.4f}  L_repel={total_L_repel_val:.4f}  "
                f"L_center={total_L_center_val:.5f}  L_smooth={total_L_smooth_val:.5f}  "
                f"L_div={total_L_div_val:.4f}  L_axis={total_L_axis_val:.4f}  "
                f"a0={debug_stats['neutral_age']:.2f}  "
                f"grad(global)={debug_stats['grad_global_preclip']:.4f}  "
                f"grad(age_scalar)={debug_stats['grad_age_scalar']:.4f}  "
                f"grad(age_head)={debug_stats['grad_age_head']:.4f}  "
                f"grad(sensitivity_dir)={debug_stats['grad_sensitivity_dir']:.4f}"
                f"{z_delta_str}"
            )

            for k, v in debug_stats.items():
                _report_scalar(logger, f"debug/{k}", "train", step, v)

        # 6. Every 10 steps: log + validate
        if step % 10 == 0:
            avg_L_rec       = log_L_rec       / log_count
            avg_L_age       = log_L_age       / log_count
            avg_mae         = log_mae         / log_count
            avg_L_anchor    = log_L_anchor    / log_count
            avg_L_def0      = log_L_def0      / log_count
            avg_L_transport = log_L_transport / log_count
            avg_L_repel     = log_L_repel     / log_count
            avg_L_center    = log_L_center    / log_count
            avg_L_smooth    = log_L_smooth    / log_count
            avg_L_div       = log_L_div       / log_count
            avg_L_axis      = log_L_axis      / log_count
            avg_L_rank      = log_L_rank      / log_count
            log_L_rec = log_L_age = log_mae = log_L_anchor = log_L_def0 = 0.0
            log_L_transport = log_L_repel = 0.0
            log_L_center = log_L_smooth = 0.0
            log_L_div = log_L_axis = log_L_rank = 0.0
            log_count = 0

            val_metrics = _log_validation(
                model,
                criterion,
                val_loader,
                step,
                logger,
                forward_mode=forward_mode,
            )
            val_mae = val_metrics["MAE"] if val_metrics is not None else None

            cur_a0 = float(getattr(base_model, "neutral_age", torch.tensor(50.0)).item())
            pbar.set_postfix(
                l_age   = f"{avg_L_age:.3f}",
                rec     = f"{avg_L_rec:.3f}",
                mae     = f"{avg_mae:.3f}/{val_mae:.3f}" if val_mae is not None else f"{avg_mae:.3f}/n/a",
                anchor  = f"{avg_L_anchor:.3f}",
                def0    = f"{avg_L_def0:.3f}",
                trans   = f"{avg_L_transport:.3f}",
                center  = f"{avg_L_center:.4f}",
                smooth  = f"{avg_L_smooth:.4f}",
                a0      = f"{cur_a0:.2f}",
                axis    = f"{avg_L_axis:.3f}",
                rank    = f"{avg_L_rank:.4f}",
            )

            _report_scalar(logger, "L_rec",       "train", step, avg_L_rec)
            _report_scalar(logger, "L_age",       "train", step, avg_L_age)
            _report_scalar(logger, "MAE",         "train", step, avg_mae)
            _report_scalar(logger, "L_anchor",    "train", step, avg_L_anchor)
            _report_scalar(logger, "L_def0",      "train", step, avg_L_def0)
            _report_scalar(logger, "L_transport", "train", step, avg_L_transport)
            _report_scalar(logger, "L_repel",     "train", step, avg_L_repel)
            _report_scalar(logger, "L_center",    "train", step, avg_L_center)
            _report_scalar(logger, "L_smooth",    "train", step, avg_L_smooth)
            _report_scalar(logger, "L_div",       "train", step, avg_L_div)
            _report_scalar(logger, "L_axis",      "train", step, avg_L_axis)
            _report_scalar(logger, "L_rank",      "train", step, avg_L_rank)
            _report_scalar(logger, "loss",        "train", step, total_loss_val)
            _report_scalar(logger, "neutral_age", "train", step, cur_a0)

        step += 1
        pbar.update(1)

    pbar.close()
    return float(np.mean(epoch_losses)), step, collapse_detected_this_epoch


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def train(
    model:       ECGDeformationModel,
    criterion:   DeformationLoss,
    ds:          ECG_Dataset,
    batch_size:  int,
    num_epochs:  int             = 20,
    learning_rate: float         = 5e-4,
    device:      torch.device    = torch.device("cuda"),
    logger=None,
    max_beats_per_record: int | None = None,
    grad_assert_every: int = 200,
    grad_assert_strict: bool = False,
    grad_assert_tol: float = 1e-10,
    graph_assert_age_pred_tol: float = 1e-6,
    strict_collapse_assert: bool = False,
    n_pretrain_epochs: int = 4,
    phase2_ramp_epochs: int = 3,
    n_align_epochs: int = 2,
    align_lr_factor: float = 0.1,
) -> None:
    """Three-phase training: canonical pretrain → axis learning → alignment (sensitivity_dir only)."""
    print("\n" + "=" * 60)
    print("DEFORMATION MODEL – THREE-PHASE TRAINING")
    print("=" * 60 + "\n")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_loader = ECGBeatBatcher(
        ds=ds,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        seed=1986,
        device=device,
        pin_memory=True,
        max_beats_per_record=max_beats_per_record,
    )
    val_loader = ECGBeatBatcher(
        ds=ds,
        split="val",
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        device=device,
        pin_memory=True,
        max_beats_per_record=max_beats_per_record,
    )
    gib = 1024 ** 3
    print(
        f"Train split cache: {'GPU' if train_loader.cached_on_device else 'CPU'} "
        f"({train_loader.cache_bytes / gib:.2f} GiB)"
    )
    print(
        f"Val split cache: {'GPU' if val_loader.cached_on_device else 'CPU'} "
        f"({val_loader.cache_bytes / gib:.2f} GiB)"
    )

    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    global_step: int = 0

    base_w = {
        "w_rec":       float(getattr(criterion, "w_rec",       1.0)),
        "w_age":       float(getattr(criterion, "w_age",       1.0)),
        "w_anchor":    float(getattr(criterion, "w_anchor",    0.3)),
        "w_def0":      float(getattr(criterion, "w_def0",      0.3)),
        "w_transport": float(getattr(criterion, "w_transport", 0.005)),
        "w_center":    float(getattr(criterion, "w_center",    1e-3)),
        "w_smooth":    float(getattr(criterion, "w_smooth",    1e-3)),
        "w_div":       float(getattr(criterion, "w_div",       0.01)),
        "w_repel":     float(getattr(criterion, "w_repel",     0.05)),
        "w_axis":      float(getattr(criterion, "w_axis",      1.0)),
        "w_rank":      float(getattr(criterion, "w_rank",      0.0)),
    }

    # Global age axis: diversity/repulsion losses not needed with a single shared B.
    base_w["w_repel"] = 0.0
    base_w["w_div"]   = 0.0

    boost_epochs = 2
    boost_repel_mult = 5.0
    pretrain_div_mult = 0.25
    pretrain_repel_mult = 0.25
    collapse_watchdog_epochs = 2
    collapse_watchdog_div_mult = 2.0
    collapse_watchdog_repel_mult = 2.0
    collapse_watchdog_transport_mult = 0.25
    collapse_watchdog_left = 0

    n_pretrain_epochs = max(0, int(n_pretrain_epochs))
    phase2_ramp_epochs = max(1, int(phase2_ramp_epochs))

    def _set_phase_weights(epoch_idx: int) -> tuple[str, float, bool, bool]:
        nonlocal collapse_watchdog_left
        watchdog_active = collapse_watchdog_left > 0

        if epoch_idx < n_pretrain_epochs:
            phase = "pretrain"
            ramp = 0.0
            boosted = False
            pre_ramp = float(epoch_idx + 1) / float(max(1, n_pretrain_epochs))

            criterion.w_rec       = base_w["w_rec"]
            criterion.w_anchor    = base_w["w_anchor"]
            criterion.w_def0      = 0.0
            criterion.w_age       = 0.0
            criterion.w_transport = 0.0
            criterion.w_repel     = base_w["w_repel"] * pretrain_repel_mult * pre_ramp
            criterion.w_center    = base_w["w_center"]
            criterion.w_smooth    = base_w["w_smooth"]
            criterion.w_div       = base_w["w_div"] * pretrain_div_mult * pre_ramp
            criterion.w_axis      = 0.0
            criterion.w_rank      = 0.0

        else:
            phase = "axis"
            epoch_since_phase2_zero_based = epoch_idx - n_pretrain_epochs
            epoch_since_phase2 = epoch_since_phase2_zero_based + 1
            ramp = min(1.0, float(epoch_since_phase2) / float(phase2_ramp_epochs))
            boosted = epoch_since_phase2_zero_based < boost_epochs

            criterion.w_rec       = base_w["w_rec"]
            criterion.w_anchor    = base_w["w_anchor"]
            criterion.w_def0      = base_w["w_def0"] * ramp
            criterion.w_age       = base_w["w_age"] * ramp
            criterion.w_transport = base_w["w_transport"] * ramp
            criterion.w_repel     = base_w["w_repel"] * ramp
            criterion.w_center    = base_w["w_center"]
            criterion.w_smooth    = base_w["w_smooth"]
            criterion.w_div       = base_w["w_div"] * ramp
            criterion.w_axis      = base_w["w_axis"] * ramp
            criterion.w_rank      = base_w["w_rank"] * ramp

            if boosted:
                criterion.w_repel = criterion.w_repel * boost_repel_mult

        if watchdog_active:
            criterion.w_repel = criterion.w_repel * collapse_watchdog_repel_mult
            criterion.w_div = criterion.w_div * collapse_watchdog_div_mult
            criterion.w_transport = criterion.w_transport * collapse_watchdog_transport_mult

        return phase, ramp, boosted, watchdog_active

    for epoch in range(num_epochs):
        phase, ramp, boosted, watchdog_active = _set_phase_weights(epoch)

        forward_mode = "pretrain" if phase == "pretrain" else "axis"

        print(
            f"[schedule epoch={epoch + 1}] phase={phase} ramp={ramp:.3f} | "
            f"w_rec={criterion.w_rec:.3f} w_anchor={criterion.w_anchor:.3f} w_def0={criterion.w_def0:.3f} "
            f"w_age={criterion.w_age:.3f} w_trans={criterion.w_transport:.4f} "
            f"w_center={criterion.w_center:.5f} w_smooth={criterion.w_smooth:.5f} "
            f"w_div={criterion.w_div:.4f} w_axis={criterion.w_axis:.4f} "
            f"w_rank={criterion.w_rank:.4f} "
            f"boosted={boosted} watchdog_active={watchdog_active} watchdog_left={collapse_watchdog_left}"
        )

        _report_scalar(logger, "schedule/phase_is_axis", "train", epoch, float(phase == "axis"))
        _report_scalar(logger, "schedule/ramp", "train", epoch, ramp)
        _report_scalar(logger, "schedule/boosted", "train", epoch, float(boosted))
        _report_scalar(logger, "schedule/watchdog_active", "train", epoch, float(watchdog_active))
        _report_scalar(logger, "schedule/watchdog_left", "train", epoch, float(collapse_watchdog_left))
        _report_scalar(logger, "schedule/w_transport", "train", epoch, criterion.w_transport)
        _report_scalar(logger, "schedule/w_repel", "train", epoch, criterion.w_repel)
        _report_scalar(logger, "schedule/w_div", "train", epoch, criterion.w_div)
        _report_scalar(logger, "schedule/w_axis", "train", epoch, criterion.w_axis)
        _report_scalar(logger, "schedule/w_rank", "train", epoch, criterion.w_rank)

        mean_loss, global_step, collapse_detected = train_one_epoch(
            model, criterion, optimizer, train_loader, val_loader,
            epoch, logger, global_step, total_steps=total_steps,
            grad_assert_every=grad_assert_every,
            grad_assert_strict=grad_assert_strict,
            grad_assert_tol=grad_assert_tol,
            graph_assert_age_pred_tol=graph_assert_age_pred_tol,
            strict_collapse_assert=strict_collapse_assert,
            forward_mode=forward_mode,
        )

        if (not _is_global_axis_model(model)) and collapse_detected:
            collapse_watchdog_left = max(collapse_watchdog_left, collapse_watchdog_epochs)
            print(
                f"[watchdog epoch={epoch + 1}] collapse detected; enabling mitigation for "
                f"next {collapse_watchdog_left} epoch(s)."
            )
            _report_scalar(logger, "schedule/watchdog_triggered", "train", epoch, 1.0)
        else:
            _report_scalar(logger, "schedule/watchdog_triggered", "train", epoch, 0.0)
            if collapse_watchdog_left > 0:
                collapse_watchdog_left -= 1

        scheduler.step()
        print(f"Epoch {epoch + 1}/{num_epochs}  mean_loss={mean_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

    print("\n✓ Main training complete.\n")

    # ------------------------------------------------------------------
    # Align phase: freeze encoder + canonical, train sensitivity_dir only.
    # Purpose: sharpen g_hat monotonicity without corrupting canonical branch.
    # ------------------------------------------------------------------
    n_align_epochs = max(0, int(n_align_epochs))
    if n_align_epochs > 0:
        print(f"{'=' * 60}")
        print(f"ALIGN PHASE ({n_align_epochs} epoch(s)) — sensitivity_dir + age_head only")
        print(f"{'=' * 60}\n")

        base_model_ref = getattr(model, "_orig_mod", model)
        align_params = list(base_model_ref.age_head.parameters()) + [base_model_ref.sensitivity_dir]
        align_optimizer = optim.Adam(align_params, lr=learning_rate * align_lr_factor)

        # Freeze encoder + canonical decoder for the duration of the align phase.
        frozen_params = (
            list(base_model_ref.encoder.parameters())
            + list(base_model_ref.morph_decoder.parameters())
        )
        for p in frozen_params:
            p.requires_grad_(False)

        try:
            # Use full w_rank; keep w_age active; zero out reconstruction terms
            # that would need encoder gradients.
            criterion.w_rank      = base_w["w_rank"]
            criterion.w_age       = base_w["w_age"]
            criterion.w_axis      = 0.0   # L_axis trains spline, skip here
            criterion.w_rec       = 0.0
            criterion.w_anchor    = 0.0
            criterion.w_transport = 0.0
            criterion.w_def0      = 0.0
            criterion.w_leak      = 0.0

            print(
                f"[align schedule] w_rank={criterion.w_rank:.4f} "
                f"w_age={criterion.w_age:.3f} lr_factor={align_lr_factor}"
            )

            for align_epoch in range(n_align_epochs):
                model.train()
                train_loader.seed = 9999 + align_epoch
                epoch_align_losses: list[float] = []

                for batch in train_loader:
                    x_a    = batch.beats
                    mask_a = batch.masks
                    age_a  = batch.ages.float()

                    x_aug_a = AmplitudeAugmentation.apply_all(x_a)
                    x_aug_a = x_aug_a * mask_a.unsqueeze(1)
                    x_norm_a = normalize_beat(x_aug_a, mask_a)

                    align_optimizer.zero_grad()
                    out_a = model(x_norm_a, age_a, mode="axis")
                    ld_a  = criterion(out_a, x_norm_a, age_a, mask=mask_a, model=base_model_ref)
                    ld_a["loss"].backward()
                    nn.utils.clip_grad_norm_(align_params, max_norm=1.0)
                    align_optimizer.step()
                    epoch_align_losses.append(float(ld_a["loss"].detach().item()))

                mean_align = float(np.mean(epoch_align_losses))
                global_step += len(train_loader)
                print(
                    f"Align epoch {align_epoch + 1}/{n_align_epochs}  "
                    f"mean_loss={mean_align:.4f}  "
                    f"L_rank={float(ld_a['L_rank'].item()):.4f}  "
                    f"L_age={float(ld_a['L_age'].item()):.4f}"
                )
                _report_scalar(logger, "align/mean_loss", "train", align_epoch, mean_align)
                _report_scalar(logger, "align/L_rank",    "train", align_epoch, float(ld_a["L_rank"].item()))

        finally:
            # Always restore requires_grad, even if an exception occurs.
            for p in frozen_params:
                p.requires_grad_(True)

        print("\n✓ Align phase complete.\n")

    print("\n✓ Training complete.\n")


# ---------------------------------------------------------------------------
# Evaluation (beat-level)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:     ECGDeformationModel,
    ds:        ECG_Dataset,
    batch_size: int,
    device:    torch.device = torch.device("cuda"),
    max_beats_per_record: int | None = None,
) -> dict:
    """
    Evaluate on the test split at beat level.
    """
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60 + "\n")

    model.to(device).eval()

    test_loader = ECGBeatBatcher(
        ds=ds,
        split="test",
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        device=device,
        pin_memory=True,
        max_beats_per_record=max_beats_per_record,
    )

    age_true_all: list[float] = []
    age_pred_all: list[float] = []
    for batch in tqdm(test_loader, total=len(test_loader), desc="Evaluating"):
        x_norm = normalize_beat(batch.beats, batch.masks)
        age_pred = model.predict_age(x_norm)
        age_true = batch.ages.float()

        age_true_all.extend(age_true.detach().cpu().numpy().tolist())
        age_pred_all.extend(age_pred.detach().cpu().numpy().tolist())

    age_true_arr = np.array(age_true_all)
    age_pred_arr = np.array(age_pred_all)
    mae          = metrics.mean_absolute_error(age_true_arr, age_pred_arr)
    print(f"  Test beat-MAE: {mae:.4f} years\n")
    return {"MAE": float(mae), "age_true": age_true_arr, "age_pred": age_pred_arr}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _export_state_dict_for_inference(model: nn.Module) -> dict:
    """
    Export weights without torch.compile wrapper prefixes (e.g. "_orig_mod.").
    """
    base_model = getattr(model, "_orig_mod", model)
    state_dict = base_model.state_dict()

    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state_dict.items()
        }
    return state_dict


def main():
    FS              = 100
    NUM_EPOCHS      = 10
    BATCH_SIZE      = 500
    MAX_BEATS_PER_RECORD = 10
    LR              = 1e-3
    RANDOM_SEED     = 1986

    W_REC       = 1.5
    W_AGE       = 1.0
    W_ANCHOR    = 0.7
    W_DEF0      = 0.3
    W_TRANSPORT = 1.0
    W_CENTER    = 1e-3   # neutral-age center regularizer
    W_SMOOTH    = 1e-3   # g(age) smoothness regularizer
    W_DIV       = 0.5
    W_REPEL     = 0.15
    W_AXIS      = 0.1
    W_RANK      = 0.05

    N_PRETRAIN_EPOCHS  = 2
    PHASE2_RAMP_EPOCHS = 2
    N_ALIGN_EPOCHS     = 2

    GRAD_ASSERT_EVERY = 200
    GRAD_ASSERT_STRICT = False
    GRAD_ASSERT_TOL = 1e-10
    GRAPH_ASSERT_AGE_PRED_TOL = 1e-6
    STRICT_COLLAPSE_ASSERT = False

    random.seed(RANDOM_SEED)
    nprand.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    logger = None
    if _CLEARML_AVAILABLE:
        task = Task.init(
            project_name=f"Age-Def",
            task_name="canonical_age_deformation",
        )
        logger = task.get_logger()

    ecg_path = f"./data/ptbxl_{FS}_T.pkl"
    print(f"Loading ECG_Dataset from {ecg_path} ...")
    ds = ECG_Dataset.load(ecg_path)

    train_records = ds.get_data("train")
    if len(train_records) == 0:
        raise ValueError("Empty train split in dataset.")
    beat_length = int(train_records[0].beat_representations.shape[1])
    print(f"Detected beat length: {beat_length}")

    model = ECGDeformationModel(
        beat_length=beat_length,
    ).to(device)

    criterion = DeformationLoss(
        w_rec       = W_REC,
        w_age       = W_AGE,
        w_anchor    = W_ANCHOR,
        w_def0      = W_DEF0,
        w_transport = W_TRANSPORT,
        w_center    = W_CENTER,
        w_smooth    = W_SMOOTH,
        w_div       = W_DIV,
        w_repel     = W_REPEL,
        w_axis      = W_AXIS,
        w_rank      = W_RANK,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    start = time.time()
    train(
        model, criterion, ds, BATCH_SIZE,
        num_epochs    = NUM_EPOCHS,
        learning_rate = LR,
        device        = device,
        logger        = logger,
        max_beats_per_record = MAX_BEATS_PER_RECORD,
        grad_assert_every = GRAD_ASSERT_EVERY,
        grad_assert_strict = GRAD_ASSERT_STRICT,
        grad_assert_tol = GRAD_ASSERT_TOL,
        graph_assert_age_pred_tol = GRAPH_ASSERT_AGE_PRED_TOL,
        strict_collapse_assert = STRICT_COLLAPSE_ASSERT,
        n_pretrain_epochs  = N_PRETRAIN_EPOCHS,
        phase2_ramp_epochs = PHASE2_RAMP_EPOCHS,
        n_align_epochs     = N_ALIGN_EPOCHS,
    )

    os.makedirs("models", exist_ok=True)
    name = f"ecg_def_mod{FS}"
    torch.save(_export_state_dict_for_inference(model), f"models/{name}.pth")
    print(f"Checkpoint saved: models/{name}.pth\n")

    results     = evaluate(
        model, ds, BATCH_SIZE, device,
        max_beats_per_record=MAX_BEATS_PER_RECORD,
    )
    val_results = run_all_validation_tests(
        model, ds, BATCH_SIZE, device, logger,
        max_beats_per_record=MAX_BEATS_PER_RECORD,
    )

    if logger is not None:
        logger.report_single_value("test/MAE", results["MAE"])
        task.close()

    elapsed = time.time() - start
    print(f"{'=' * 60}")
    print(f"Done in {elapsed / 60:.1f} min | Test MAE: {results['MAE']:.4f} years")
    print(f"{'=' * 60}\n")
    return results, val_results


if __name__ == "__main__":
    main()