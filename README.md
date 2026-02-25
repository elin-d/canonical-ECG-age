# Canonical-Age Factorized ECG Deformation Model

A Strict 1D Monotonic Aging Axis for Amplitude-Invariant ECG Morphology
Modeling

------------------------------------------------------------------------

## 1. Abstract

This repository presents a canonical-age factorized model for 12‑lead
ECG signals that explicitly disentangles morphology from age-related
variation. The method represents aging as deformation along a single
global monotonic axis while preserving an age‑invariant morphology
latent space.

The formulation guarantees:

-   Exact neutral age constraint (zero deformation at learned canonical
    age)
-   Strict monotonicity of the age function
-   Amplitude transport invariance
-   No age leakage into morphology
-   Explicit gradient routing between objectives

Post‑training validation confirms monotonicity, transport invariance,
absence of age leakage (linear and nonlinear probes), and robust
generalization with a test MAE of 9.63 years.

------------------------------------------------------------------------

## 2. Method Overview

Given an ECG beat:

x ∈ ℝ\^{C×T}

The model factorizes:

z = Encoder(x)

x₅₀ = D_morph(z)

Δ(z,a) = g(a) · B

x_recon = x₅₀ + Δ(z,a)

Where:

-   z ∈ ℝ\^d --- age‑invariant morphology latent code\
-   B ∈ ℝ\^{C×T} --- global aging sensitivity direction\
-   g(a) ∈ ℝ --- strictly monotonic scalar age coordinate\
-   a₀ --- learnable neutral age\
-   g(a₀) = 0 ensures Δ(z,a₀) = 0 exactly

Aging is therefore constrained to a single interpretable deformation
axis.

------------------------------------------------------------------------

## 3. Amplitude-Invariant Signal Design

Amplitude-related shortcuts are explicitly prevented via:

1.  Amplitude-only augmentations (random gain, per-lead scaling,
    baseline drift).

2.  Transport invariance constraint:

    L_transport = 1 − cos(B, B_aug)

3.  Canonical leakage penalty preventing correlation between x₅₀
    statistics and age.

4.  Latent leakage penalty enforcing zero correlation between z
    dimensions and age.

5.  Age head receives signal only from axis projection.

As a result:

-   Age cannot be inferred from amplitude magnitude.
-   Morphology latent space remains age‑free.
-   The aging axis captures directional structural variation, not scale
    artifacts.

------------------------------------------------------------------------

## 4. Architecture

### Encoder

Dilated temporal convolutional network with derivative front‑end
producing:

z ∈ ℝ\^{256}

### Canonical Decoder

Transposed convolutional generator:

x₅₀ = D_morph(z)

### Global Aging Axis

Single shared learnable tensor:

B ∈ ℝ\^{1×12×T}

Normalized per batch:

B_dir = B / \|\|B\|\|

### Monotonic Age Function

Piecewise linear spline with strictly positive increments:

g(a) = g_raw(a) − g_raw(a₀)

Guarantees: - Strict monotonicity - Zero deformation at neutral age -
Smoothness via finite-difference penalty

### Age Calibration Head

Age predicted from scalar projection:

ĝ = ⟨Δ, B_dir⟩

age_pred = AgeHead(ĝ)

------------------------------------------------------------------------

## 5. Training Stages

### Stage 1 --- Canonical Pretraining

Optimize: L_rec

Freeze: age scalar, sensitivity axis, age head

### Stage 2 --- Axis Learning

Activate: - sensitivity direction - monotonic spline - neutral age - age
head

### Stage 3 --- Full Joint Training

All parameters active with leakage penalties and transport constraints.

------------------------------------------------------------------------

## 6. Loss Functions

Reconstruction:

L_rec = \|\|x_recon − x\|\|₁

Age regression:

L_age = \|\|age_pred − age\|\|₁

Neutral constraint:

L_def0 = \|\|Δ(z,a₀)\|\|

Smoothness:

L_smooth = mean((g(a+1) − 2g(a) + g(a−1))²)

Axis consistency:

L_axis = \|\|ĝ − g(a)\|\|₁

Latent leakage:

L_leak_z = mean(corr(z_k, age)²)

Canonical leakage:

L_leak_x50 = corr(\|x₅₀\|, age)²

Transport invariance:

L_transport = 1 − cos(B, B_aug)

Total loss is weighted combination of all terms.

------------------------------------------------------------------------

## 7. Dataset

-   PTB‑XL ECG dataset
-   12 leads
-   Beat‑segmented
-   Per‑lead normalization
-   Age labels in years

------------------------------------------------------------------------

## 8. Evaluation Metrics

### Age Regression

-   Mean Absolute Error (MAE)

### Axis Validity Tests

-   Direction similarity
-   Neutral point exactness
-   Spearman monotonicity of g(age)
-   Finite difference smoothness
-   Linear and MLP leakage probes
-   Transport invariance ratio
-   Canonical leakage correlation
-   Lead permutation robustness

All validation tests: PASS

Test MAE: 9.6276 years

------------------------------------------------------------------------

## 9. Reproducibility

-   Deterministic beat-level batching
-   Fixed seeds
-   Explicit gradient-routing assertions
-   Automated validation suite
-   Leakage probe testing

------------------------------------------------------------------------

## 11. Training

python train_deformation.py\
--dataset ./data/ptbxl_500_T.pkl\
--epochs 200\
--batch_size 256

------------------------------------------------------------------------

## 12. Evaluation

python train_deformation.py\
--evaluate_only\
--checkpoint models/\_ecg_deformation_model.pth


------------------------------------------------------------------------

## 14. License

MIT License

------------------------------------------------------------------------
