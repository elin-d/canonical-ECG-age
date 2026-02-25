"""Post-training validation tests for the ECG age-deformation model.

All test functions are self-contained and import only from the model
package and standard libraries — no import from train_deformation.py.

Standalone usage::

    python deformation_validation.py models/ecg_def_mod100.pth
    python deformation_validation.py models/ecg_def_mod100.pth \\
        --data ../data/ptbxl_100_T.pkl --batch-size 256 --device cuda
"""
from __future__ import annotations

import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ecg_deformation_model import (
    ECGDeformationModel,
    normalize_beat,
)
from ECG_Dataset import ECG_Dataset
from ecg_beat_batcher import BeatBatch, ECGBeatBatcher


# Correlation helpers (no scipy dependency)
def _safe_pearson_corr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    den = (np.sqrt((x * x).mean() + eps) * np.sqrt((y * y).mean() + eps))
    if not np.isfinite(den) or den <= 0:
        return float("nan")
    return float((x * y).mean() / den)


def _spearman_corr(x, y) -> float:
    """Spearman rank correlation without scipy. Works with numpy or torch."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    else:
        y = np.asarray(y)

    x = x.reshape(-1)
    y = y.reshape(-1)
    if x.size < 2 or y.size < 2:
        return float("nan")

    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return _safe_pearson_corr(rx, ry)



# Model-type detection
def _is_global_axis_model(model: nn.Module) -> bool:
    """
    Global-axis mode detection.

    Preferred: model.is_global_axis flag.
    Fallback: global-axis model exposes a single shared sensitivity_dir Parameter
    instead of a per-sample sensitivity network.
    """
    base = getattr(model, "_orig_mod", model)
    flag = getattr(base, "is_global_axis", None)
    if flag is not None:
        return bool(flag)

    sd = getattr(base, "sensitivity_dir", None)
    return isinstance(sd, torch.nn.Parameter)



# Data-sampling helpers
@torch.no_grad()
def _sample_split_beats(
    ds: ECG_Dataset,
    split: str,
    batch_size: int,
    device: torch.device,
    num_samples: int,
    max_beats_per_record: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect up to num_samples normalized beats and ages from a split."""
    loader = ECGBeatBatcher(
        ds=ds, split=split, batch_size=batch_size,
        shuffle=False, seed=0, device=device, pin_memory=True,
        max_beats_per_record=max_beats_per_record,
    )
    x_chunks: list[torch.Tensor] = []
    a_chunks: list[torch.Tensor] = []
    m_chunks: list[torch.Tensor] = []
    remaining = int(num_samples)

    for batch in loader:
        if remaining <= 0:
            break
        take = min(remaining, int(batch.beats.size(0)))
        x_chunks.append(batch.beats[:take])
        a_chunks.append(batch.ages[:take].float())
        m_chunks.append(batch.masks[:take])
        remaining -= take

    if not x_chunks:
        raise ValueError("No beats available in test split.")

    m = torch.cat(m_chunks, dim=0)
    x = normalize_beat(torch.cat(x_chunks, dim=0), m)
    a = torch.cat(a_chunks, dim=0)
    return x, a


@torch.no_grad()
def _collect_latents_and_age(
    model: ECGDeformationModel,
    ds: ECG_Dataset,
    split: str,
    batch_size: int,
    device: torch.device,
    max_samples: int,
    max_beats_per_record: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect latent embeddings z and ages from a split."""
    loader = ECGBeatBatcher(
        ds=ds,
        split=split,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        device=device,
        pin_memory=True,
        max_beats_per_record=max_beats_per_record,
    )
    z_chunks: list[np.ndarray] = []
    a_chunks: list[np.ndarray] = []
    remaining = int(max_samples)
    for batch in loader:
        if remaining <= 0:
            break
        take = min(remaining, int(batch.beats.size(0)))
        x = normalize_beat(batch.beats[:take], batch.masks[:take])
        z = model.encode(x).detach().cpu().numpy()
        a = batch.ages[:take].detach().cpu().numpy()
        z_chunks.append(z)
        a_chunks.append(a)
        remaining -= take

    if not z_chunks:
        return np.empty((0, 256), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.concatenate(z_chunks, axis=0), np.concatenate(a_chunks, axis=0)


# Validation tests
@torch.no_grad()
def test_direction_similarity(
    model:               ECGDeformationModel,
    ds:                  ECG_Dataset,
    batch_size:          int,
    device:              torch.device,
    num_samples:         int   = 128,
    threshold_cosine:    float = 0.90,
    max_beats_per_record: int | None = None,
) -> dict:
    """
    Test 1 – Direction similarity:
        cos(Delta_z1(70), Delta_z2(70)) > 0.9
    """
    print("\n[Validation] Direction Similarity Test")
    model.eval()

    x_sample, age_sample = _sample_split_beats(
        ds, "test", batch_size, device, num_samples, max_beats_per_record=max_beats_per_record
    )
    z = model.encode(x_sample)
    age70 = torch.full_like(age_sample, 70.0)
    delta70 = model.deformation(z, age70).flatten(1)
    if int(delta70.size(0)) < 2:
        mean_cos = 0.0
        test_pass = False
    else:
        perm = torch.randperm(delta70.size(0), device=delta70.device)
        if bool((perm == torch.arange(delta70.size(0), device=delta70.device)).any()):
            perm = torch.roll(torch.arange(delta70.size(0), device=delta70.device), shifts=1)
        cos = F.cosine_similarity(delta70, delta70[perm], dim=1)
        mean_cos = float(cos.mean().item())
        test_pass = mean_cos > threshold_cosine

    status = "PASS" if test_pass else "FAIL"
    print(f"  mean cosine similarity: {mean_cos:.6f}  | threshold>{threshold_cosine:.2f} | {status}")
    return {"mean_cosine_similarity": mean_cos, "pass": test_pass}


@torch.no_grad()
def test_neutral_point(
    model:               ECGDeformationModel,
    ds:                  ECG_Dataset,
    batch_size:          int,
    device:              torch.device,
    num_samples:         int   = 128,
    threshold_norm:      float = 1e-5,
    max_beats_per_record: int | None = None,
) -> dict:
    """
    Test 2 – Neutral point:
        ||Delta(a0)|| ≈ 0  (guaranteed by spline construction).

    g(a0) = 0 by design, so this is primarily a sanity check.
    """
    print("\n[Validation] Neutral Point Test")
    model.eval()
    base_model = getattr(model, "_orig_mod", model)

    x_sample, age_sample = _sample_split_beats(
        ds, "test", batch_size, device, num_samples,
        max_beats_per_record=max_beats_per_record,
    )
    z   = model.encode(x_sample)
    a0  = base_model.neutral_age           # live tensor
    a0_full = age_sample * 0.0 + a0
    delta_a0 = model.deformation(z, a0_full)
    mean_norm = float(delta_a0.flatten(1).norm(dim=1).mean().item())
    test_pass = mean_norm < threshold_norm
    status = "PASS" if test_pass else "FAIL"
    print(
        f"  neutral_age={float(a0.item()):.2f}y  "
        f"mean ||Delta(a0)||={mean_norm:.3e} | threshold<{threshold_norm:.1e} | {status}"
    )
    return {"mean_delta_a0_norm": mean_norm, "neutral_age": float(a0.item()), "pass": test_pass}


@torch.no_grad()
def test_g_monotonicity(
    model:               ECGDeformationModel,
    ds:                  ECG_Dataset,
    batch_size:          int,
    device:              torch.device,
    num_samples:         int   = 128,
    min_spearman:        float = 0.95,
    max_beats_per_record: int | None = None,
) -> dict:
    """
    Test 3 – Monotonicity of g(age):
        Spearman rho(g(age), age) > 0.95

    g is monotone by construction (cumsum of softplus increments), so this
    tests whether the learned mapping is non-degenerate (non-constant).
    """
    print("\n[Validation] g(age) Monotonicity Test")
    model.eval()
    base_model = getattr(model, "_orig_mod", model)

    ages = torch.linspace(10.0, 90.0, 81, device=device)   # 10, 11, ..., 90
    a0   = base_model.neutral_age
    g    = base_model.age_scalar_net(ages, a0).detach().cpu().numpy()
    ages_np = ages.cpu().numpy()

    rho = _spearman_corr(ages_np, g)
    ok  = (not np.isnan(rho)) and (rho > float(min_spearman))
    status = "PASS" if ok else "FAIL"
    print(f"  Spearman rho(g(age), age)={rho:.4f} | threshold>{min_spearman:.2f} | {status}")
    return {"spearman_rho": float(rho), "pass": bool(ok)}


@torch.no_grad()
def test_g_smoothness(
    model:               ECGDeformationModel,
    ds:                  ECG_Dataset,
    batch_size:          int,
    device:              torch.device,
    max_d2:              float = 0.5,
    max_beats_per_record: int | None = None,
) -> dict:
    """
    Test 3b – Smoothness of g(age):
        Mean squared second finite differences of g(age) on integer grid.

    Second differences approximate the discrete second derivative.  Very
    large values indicate sharp kinks in the aging curve.
    """
    print("\n[Validation] g(age) Smoothness Test")
    model.eval()
    base_model = getattr(model, "_orig_mod", model)

    ages = torch.linspace(11.0, 89.0, 79, device=device)
    a0   = base_model.neutral_age
    g    = base_model.age_scalar_net(ages, a0)
    d2   = g[2:] - 2.0 * g[1:-1] + g[:-2]
    mean_d2_sq = float((d2 ** 2).mean().item())
    ok = mean_d2_sq < float(max_d2)
    status = "PASS" if ok else "FAIL"
    print(f"  mean(d²g)²={mean_d2_sq:.6f} | threshold<{max_d2:.3f} | {status}")
    return {"mean_d2_sq": mean_d2_sq, "pass": bool(ok)}


@torch.no_grad()
def test_age_leakage(
    model:               ECGDeformationModel,
    ds:                  ECG_Dataset,
    batch_size:          int,
    device:              torch.device,
    max_train_samples:   int = 4096,
    max_test_samples:    int = 2048,
    threshold_abs_r2:    float = 0.05,
    max_beats_per_record: int | None = None,
) -> dict:
    """
    Test 4 – Age leakage:
        Linear probe on z should not predict age (R^2 near zero).
    """
    print("\n[Validation] Age Leakage Test")
    model.eval()

    z_train, a_train = _collect_latents_and_age(
        model, ds, "train", batch_size, device, max_train_samples,
        max_beats_per_record=max_beats_per_record,
    )
    z_test, a_test = _collect_latents_and_age(
        model, ds, "test", batch_size, device, max_test_samples,
        max_beats_per_record=max_beats_per_record,
    )

    if z_train.shape[0] < 2 or z_test.shape[0] < 2:
        r2 = float("nan")
        test_pass = False
    else:
        x_train = np.concatenate([z_train, np.ones((z_train.shape[0], 1), dtype=z_train.dtype)], axis=1)
        x_test = np.concatenate([z_test, np.ones((z_test.shape[0], 1), dtype=z_test.dtype)], axis=1)
        coef, *_ = np.linalg.lstsq(x_train, a_train, rcond=None)
        a_pred = x_test @ coef
        r2 = float(metrics.r2_score(a_test, a_pred))
        test_pass = max(0.0, r2) < threshold_abs_r2

    status = "PASS" if test_pass else "FAIL"
    print(f"  linear probe R^2(z->age): {r2:.6f} | max(0,R^2)<{threshold_abs_r2:.2f} | {status}")
    return {"r2": r2, "pass": test_pass}


def test_age_leakage_mlp(
    model: ECGDeformationModel,
    ds: ECG_Dataset,
    batch_size: int,
    device: torch.device,
    max_train_samples: int = 8192,
    max_test_samples: int = 4096,
    hidden: int = 128,
    train_steps: int = 400,
    lr: float = 2e-3,
    threshold_abs_r2: float = 0.05,
    max_beats_per_record: int | None = None,
) -> dict:
    """
    Extra Test – Age leakage (nonlinear):
        Tiny MLP probe on z should not predict age (R^2 near zero).
    """
    print("\n[Validation] Age Leakage Test (MLP probe)")
    model.eval()

    z_train_np, a_train_np = _collect_latents_and_age(
        model, ds, "train", batch_size, device, max_train_samples,
        max_beats_per_record=max_beats_per_record,
    )
    z_test_np, a_test_np = _collect_latents_and_age(
        model, ds, "test", batch_size, device, max_test_samples,
        max_beats_per_record=max_beats_per_record,
    )

    if z_train_np.shape[0] < 64 or z_test_np.shape[0] < 64:
        r2 = float("nan")
        test_pass = False
        print("  Not enough samples for MLP probe → FAIL")
        return {"r2": r2, "pass": test_pass}

    z_train = torch.from_numpy(z_train_np).to(device=device, dtype=torch.float32)
    a_train = torch.from_numpy(a_train_np).to(device=device, dtype=torch.float32).view(-1, 1)
    z_test = torch.from_numpy(z_test_np).to(device=device, dtype=torch.float32)
    a_test = torch.from_numpy(a_test_np).to(device=device, dtype=torch.float32).view(-1, 1)

    d = int(z_train.size(1))
    probe = nn.Sequential(
        nn.Linear(d, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, 1),
    ).to(device)

    opt = optim.Adam(probe.parameters(), lr=lr)
    probe.train()
    n = int(z_train.size(0))
    bs = min(512, n)
    for _ in range(int(train_steps)):
        idx = torch.randint(0, n, (bs,), device=device)
        pred = probe(z_train[idx])
        loss = F.mse_loss(pred, a_train[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    probe.eval()
    pred = probe(z_test).view(-1).detach().cpu().numpy()
    true = a_test.view(-1).detach().cpu().numpy()
    r2 = float(metrics.r2_score(true, pred))
    test_pass = max(0.0, r2) < threshold_abs_r2

    status = "PASS" if test_pass else "FAIL"
    print(f"  MLP probe R^2(z->age): {r2:.6f} | max(0,R^2)<{threshold_abs_r2:.2f} | {status}")
    return {"r2": r2, "pass": test_pass}


@torch.no_grad()
def test_canonical_leakage(
    model: ECGDeformationModel,
    ds: ECG_Dataset,
    batch_size: int,
    device: torch.device,
    num_samples: int = 2048,
    corr_tol: float = 0.05,
    max_beats_per_record: int | None = None,
) -> dict:
    """
    Extra Test – Canonical leakage:
        Amplitude/energy stats of x50 should not correlate with age.
    """
    print("\n[Validation] Canonical Leakage Test (x50 stats vs age)")
    model.eval()

    x, a = _sample_split_beats(
        ds, "test", batch_size, device, num_samples,
        max_beats_per_record=max_beats_per_record,
    )

    z   = model.encode(x)
    base_model = getattr(model, "_orig_mod", model)
    a0  = float(getattr(base_model, "neutral_age", torch.tensor(50.0)).item())
    a0_full = a * 0.0 + a0
    x50 = model.decode(z, a0_full)

    mean_abs = x50.abs().mean(dim=(1, 2)).detach().cpu().numpy()
    std = x50.std(dim=2).mean(dim=1).detach().cpu().numpy()
    age_np = a.detach().cpu().numpy()

    corr_mean_abs = _safe_pearson_corr(mean_abs, age_np)
    corr_std = _safe_pearson_corr(std, age_np)

    ok = (abs(corr_mean_abs) < corr_tol) and (abs(corr_std) < corr_tol)
    status = "PASS" if ok else "FAIL"
    print(f"  corr(mean|x50|, age)={corr_mean_abs:.4f} | tol<{corr_tol:.2f}")
    print(f"  corr(std(x50),  age)={corr_std:.4f} | tol<{corr_tol:.2f} | {status}")
    return {
        "corr_mean_abs": float(corr_mean_abs),
        "corr_std": float(corr_std),
        "pass": bool(ok),
    }


@torch.no_grad()
def test_b_collapse_by_lead(
    model: ECGDeformationModel,
    ds: ECG_Dataset,
    batch_size: int,
    device: torch.device,
    num_samples: int = 1024,
    cos_collapse: float = 0.999,
    max_beats_per_record: int | None = None,
) -> dict:
    """
    Extra Test – Partial collapse of B by lead:
        For each lead, mean pairwise cosine of B_dir should not be ~1.

    NOTE: In GLOBAL-AXIS mode this "collapse" is expected and this test is not meaningful.
    """
    print("\n[Validation] B Collapse Test (per-lead pairwise cosine)")
    model.eval()

    x, a = _sample_split_beats(
        ds, "test", batch_size, device, num_samples,
        max_beats_per_record=max_beats_per_record,
    )

    out = model(x, a, mode="axis")
    B = out["B"]  # [B,C,T]
    Bn = F.normalize(B, p=2, dim=2, eps=1e-8).detach()

    b, c, _t = Bn.shape
    if b < 4:
        print("  Not enough samples → FAIL")
        return {"max_mean_cos": float("nan"), "pass": False}

    m_pairs = min(2048, b * 4)
    i = torch.randint(0, b, (m_pairs,), device=device)
    j = torch.randint(0, b, (m_pairs,), device=device)
    same = (i == j)
    if bool(same.any()):
        j = j.clone()
        j[same] = (j[same] + 1) % b

    cos_per_lead = []
    for lead in range(c):
        v1 = Bn[i, lead, :]
        v2 = Bn[j, lead, :]
        cos = (v1 * v2).sum(dim=1)
        cos_per_lead.append(float(cos.mean().item()))

    max_mean_cos = float(np.max(cos_per_lead))
    ok = max_mean_cos < cos_collapse
    status = "PASS" if ok else "FAIL"
    print(f"  max over leads(mean cos)={max_mean_cos:.6f} | threshold<{cos_collapse:.3f} | {status}")
    return {"max_mean_cos": max_mean_cos, "mean_cos_per_lead": cos_per_lead, "pass": bool(ok)}


@torch.no_grad()
def test_g_hat_spearman_monotonicity(
    model: ECGDeformationModel,
    ds: ECG_Dataset,
    batch_size: int,
    device: torch.device,
    num_samples: int = 2048,
    min_spearman: float = 0.60,
    max_beats_per_record: int | None = None,
) -> dict:
    """
    Extra Test – g_hat monotonicity:
        Spearman corr(g_hat, age) should be reasonably positive.
    """
    print("\n[Validation] g_hat Spearman Monotonicity Test")
    model.eval()

    x, a = _sample_split_beats(
        ds, "test", batch_size, device, num_samples,
        max_beats_per_record=max_beats_per_record,
    )
    out = model(x, a, mode="axis")
    g_hat = out["g_hat"].detach().cpu().numpy().reshape(-1)
    age = a.detach().cpu().numpy().reshape(-1)

    rho = _spearman_corr(g_hat, age)
    ok = (not np.isnan(rho)) and (rho > float(min_spearman))
    status = "PASS" if ok else "FAIL"
    print(f"  Spearman rho(g_hat, age)={rho:.4f} | threshold>{min_spearman:.2f} | {status}")
    return {"spearman_rho": float(rho), "pass": bool(ok)}


@torch.no_grad()
def test_lead_permutation_effect(
    model: ECGDeformationModel,
    ds: ECG_Dataset,
    batch_size: int,
    device: torch.device,
    num_samples: int = 2048,
    min_mae_increase: float = 0.50,
    max_beats_per_record: int | None = None,
) -> dict:
    """
    Extra Test – Lead permutation effect:
        Permuting leads should worsen MAE by at least min_mae_increase.
        If not, model may rely on lead-invariant shortcuts.
    """
    print("\n[Validation] Lead Permutation Effect Test")
    model.eval()

    x, a = _sample_split_beats(
        ds, "test", batch_size, device, num_samples,
        max_beats_per_record=max_beats_per_record,
    )
    a_true = a.detach().cpu().numpy()

    pred = model.predict_age(x).detach().cpu().numpy()
    mae0 = float(metrics.mean_absolute_error(a_true, pred))

    c = int(x.size(1))
    perm = torch.randperm(c, device=device)
    x_perm = x[:, perm, :]
    pred_perm = model.predict_age(x_perm).detach().cpu().numpy()
    mae1 = float(metrics.mean_absolute_error(a_true, pred_perm))

    inc = mae1 - mae0
    ok = inc >= float(min_mae_increase)
    status = "PASS" if ok else "FAIL"
    print(f"  MAE original={mae0:.4f}  MAE permuted={mae1:.4f}  Δ={inc:+.4f} | minΔ>={min_mae_increase:.2f} | {status}")
    return {"mae_orig": mae0, "mae_perm": mae1, "mae_increase": float(inc), "pass": bool(ok)}


def test_transport_invariance(
    model:               ECGDeformationModel,
    ds:                  ECG_Dataset,
    batch_size:          int,
    device:              torch.device,
    num_samples:         int   = 128,
    threshold_ratio:     float = 0.20,
    max_beats_per_record: int | None = None,
) -> dict:
    """
    Test 5 – Transport invariance:
        (x_70(z1)-x_50(z1)) vs (x_70(z2)-x_50(z2))
    """
    print("\n[Validation] Transport Invariance Test")
    model.eval()

    x_sample, age_sample = _sample_split_beats(
        ds, "test", batch_size, device, num_samples, max_beats_per_record=max_beats_per_record
    )
    z = model.encode(x_sample)
    age70 = torch.full_like(age_sample, 70.0)
    age50 = torch.full_like(age_sample, 50.0)
    delta = (model.decode(z, age70) - model.decode(z, age50)).flatten(1)

    if int(delta.size(0)) < 2:
        mean_l2 = float("nan")
        ratio = float("nan")
        test_pass = False
    else:
        perm = torch.randperm(delta.size(0), device=delta.device)
        if bool((perm == torch.arange(delta.size(0), device=delta.device)).any()):
            perm = torch.roll(torch.arange(delta.size(0), device=delta.device), shifts=1)
        diff = delta - delta[perm]
        mean_l2 = float(diff.norm(dim=1).mean().item())
        mean_norm = float(delta.norm(dim=1).mean().item())
        ratio = mean_l2 / (mean_norm + 1e-8)
        test_pass = ratio < threshold_ratio

    status = "PASS" if test_pass else "FAIL"
    print(f"  mean L2 difference: {mean_l2:.6f}  ratio={ratio:.6f} | ratio<{threshold_ratio:.2f} | {status}")
    return {"mean_l2_difference": mean_l2, "relative_ratio": ratio, "pass": test_pass}


# Test runner
def _run_test(name: str, fn, *args, fail_result: dict | None = None, **kwargs) -> dict:
    """Run a single validation test, catching exceptions so the suite continues."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"  ERROR in {name}: {exc}")
        return fail_result if fail_result is not None else {"pass": False, "error": str(exc)}


def run_all_validation_tests(
    model:     ECGDeformationModel,
    ds:        ECG_Dataset,
    batch_size: int,
    device:    torch.device,
    logger=None,
    max_beats_per_record: int | None = None,
) -> dict:
    """Run required post-training axis-identifiability tests.

    Each test is wrapped in a try/except so a single crash does not abort
    the remaining tests.  Results are printed after all tests complete.
    """
    kw = dict(max_beats_per_record=max_beats_per_record)

    # 1. Direction similarity.
    r1 = _run_test("direction_similarity", test_direction_similarity,
                   model, ds, batch_size, device,
                   fail_result={"mean_cosine_similarity": float("nan"), "pass": False}, **kw)

    # 2. Neutral point: ||Delta(a0)|| ≈ 0.
    r2 = _run_test("neutral_point", test_neutral_point,
                   model, ds, batch_size, device,
                   fail_result={"mean_delta_a0_norm": float("nan"), "neutral_age": float("nan"), "pass": False}, **kw)

    # 3. g(age) monotonicity.
    r3 = _run_test("g_monotonicity", test_g_monotonicity,
                   model, ds, batch_size, device,
                   fail_result={"spearman_rho": float("nan"), "pass": False}, **kw)

    # 3b. g(age) smoothness.
    r3b = _run_test("g_smoothness", test_g_smoothness,
                    model, ds, batch_size, device,
                    fail_result={"mean_d2_sq": float("nan"), "pass": False}, **kw)

    # 4. Latent age leakage (linear probe).
    r4 = _run_test("age_leakage_linear", test_age_leakage,
                   model, ds, batch_size, device,
                   fail_result={"r2": float("nan"), "pass": False}, **kw)

    # 4b. Latent age leakage (MLP probe).
    r4b = _run_test("age_leakage_mlp", test_age_leakage_mlp,
                    model, ds, batch_size, device,
                    fail_result={"r2": float("nan"), "pass": False}, **kw)

    # 5. Transport invariance.
    r5 = _run_test("transport_invariance", test_transport_invariance,
                   model, ds, batch_size, device,
                   fail_result={"mean_l2_difference": float("nan"), "relative_ratio": float("nan"), "pass": False}, **kw)

    # 6. Canonical leakage.
    r6 = _run_test("canonical_leakage", test_canonical_leakage,
                   model, ds, batch_size, device,
                   fail_result={"corr_mean_abs": float("nan"), "corr_std": float("nan"), "pass": False}, **kw)

    # 7. B-collapse (skip in global-axis mode).
    if _is_global_axis_model(model):
        print("\n[Validation] B Collapse Test (per-lead pairwise cosine)")
        print("  GLOBAL axis mode detected: skipping (expected shared B). → PASS")
        r7 = {"max_mean_cos": float("nan"), "pass": True}
    else:
        r7 = _run_test("b_collapse", test_b_collapse_by_lead,
                       model, ds, batch_size, device,
                       fail_result={"max_mean_cos": float("nan"), "pass": False}, **kw)

    # 8. g_hat Spearman monotonicity.
    r8 = _run_test("g_hat_spearman", test_g_hat_spearman_monotonicity,
                   model, ds, batch_size, device,
                   fail_result={"spearman_rho": float("nan"), "pass": False}, **kw)

    # 9. Lead permutation sensitivity.
    r9 = _run_test("lead_permutation", test_lead_permutation_effect,
                   model, ds, batch_size, device,
                   fail_result={"mae_orig": float("nan"), "mae_perm": float("nan"),
                                "mae_increase": float("nan"), "pass": False}, **kw)

    all_pass = (
        r1["pass"] and r2["pass"] and r3["pass"] and r3b["pass"] and
        r4["pass"] and r4b["pass"] and
        r5["pass"] and r6["pass"] and r7["pass"] and r8["pass"] and r9["pass"]
    )

    # ---------------------------------------------------------------
    # Print summary table
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("POST-TRAINING VALIDATION RESULTS")
    print("=" * 60)
    rows = [
        ("1  direction_similarity",  r1,  "mean_cosine_similarity"),
        ("2  neutral_point",         r2,  "mean_delta_a0_norm"),
        ("3  g_monotonicity",        r3,  "spearman_rho"),
        ("3b g_smoothness",          r3b, "mean_d2_sq"),
        ("4  age_leakage_linear",    r4,  "r2"),
        ("4b age_leakage_mlp",       r4b, "r2"),
        ("5  transport_invariance",  r5,  "relative_ratio"),
        ("6  canonical_leakage",     r6,  "corr_mean_abs"),
        ("7  b_collapse",            r7,  "max_mean_cos"),
        ("8  g_hat_spearman",        r8,  "spearman_rho"),
        ("9  lead_permutation",      r9,  "mae_increase"),
    ]
    for label, r, key in rows:
        val = r.get(key, float("nan"))
        status = "PASS" if r.get("pass", False) else "FAIL"
        try:
            val_str = f"{float(val):.4f}"
        except (TypeError, ValueError):
            val_str = str(val)
        print(f"  {label:<30s}  {key}={val_str:<10s}  {status}")
    print("-" * 60)
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60 + "\n")

    if logger is not None:
        logger.report_single_value("val/direction_similarity_cosine", r1.get("mean_cosine_similarity", float("nan")))
        logger.report_single_value("val/neutral_point_norm",          r2.get("mean_delta_a0_norm",     float("nan")))
        logger.report_single_value("val/g_monotonicity_spearman",     r3.get("spearman_rho",           float("nan")))
        logger.report_single_value("val/g_smoothness_d2",             r3b.get("mean_d2_sq",            float("nan")))
        logger.report_single_value("val/age_leakage_r2",              r4.get("r2",                     float("nan")))
        logger.report_single_value("val/transport_invariance_l2",     r5.get("mean_l2_difference",     float("nan")))

    return {
        "direction_similarity": r1,
        "neutral_point":        r2,
        "g_monotonicity":       r3,
        "g_smoothness":         r3b,
        "age_leakage_linear":   r4,
        "age_leakage_mlp":      r4b,
        "transport_invariance": r5,
        "canonical_leakage":    r6,
        "b_collapse_test":      r7,
        "g_hat_spearman":       r8,
        "lead_permutation":     r9,
        "all_pass":             all_pass,
    }


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Load a checkpoint and run all post-training validation tests.",
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="./models/ecg_def_mod100.pth",
        help="Path to model checkpoint (.pth), as saved by train_deformation.py. "
             "Default: %(default)s",
    )
    parser.add_argument(
        "--data",
        default="../data/ptbxl_100_T.pkl",
        metavar="PATH",
        help="Path to ECG dataset pickle (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        metavar="N",
        help="Batch size for validation loaders (default: %(default)s).",
    )
    parser.add_argument(
        "--max-beats-per-record",
        type=int,
        default=10,
        metavar="N",
        help="Cap beats sampled per record (default: %(default)s).",
    )
    parser.add_argument(
        "--beat-length",
        type=int,
        default=None,
        metavar="T",
        help="Beat length in samples. Inferred from the dataset when omitted.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string, e.g. 'cuda' or 'cpu'. Auto-detected when omitted.",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    print(f"Loading dataset from {args.data} ...")
    ds = ECG_Dataset.load(args.data)

    if args.beat_length is not None:
        beat_length = args.beat_length
    else:
        records = ds.get_data("train") or ds.get_data("test")
        if not records:
            raise ValueError("Dataset contains no train or test records.")
        beat_length = int(records[0].beat_representations.shape[1])
    print(f"Beat length: {beat_length}")

    model = ECGDeformationModel(beat_length=beat_length).to(device)

    print(f"Loading checkpoint from {args.checkpoint} ...")
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # Strip _orig_mod. prefix produced by torch.compile wrapper, if present.
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)
    print(f"Checkpoint loaded. neutral_age={float(model.neutral_age.item()):.2f}y\n")

    run_all_validation_tests(
        model=model,
        ds=ds,
        batch_size=args.batch_size,
        device=device,
        max_beats_per_record=args.max_beats_per_record,
    )


if __name__ == "__main__":
    main()